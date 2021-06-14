#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: trainer_condition_SN_multiscale.py
# Created Date: Saturday April 18th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 27th April 2020 11:11:28 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import  os
import  time
import  datetime

import  torch
import  torch.nn as nn
import  torch.nn.functional as F
from    torch.autograd     import Variable
from    torchvision.utils  import save_image
from    functools import partial

from    components.Transform import Transform_block
from    utilities.utilities import denorm
from    data_tools.data_loader_condition_final_new import getLoader

class Trainer(object):
    def __init__(self, config, reporter):

        self.config     = config
        # logger
        self.reporter   = reporter
        # Data loader
        # self.dataloaders= dataloaders_list

    def train(self):
        
        ckpt_dir    = self.config["projectCheckpoints"]
        sample_dir  = self.config["projectSamples"]
        total_step  = self.config["totalStep"]
        log_frep    = self.config["logStep"]
        sample_freq = self.config["sampleStep"]
        model_freq  = self.config["modelSaveStep"]
        lr_base     = self.config["gLr"]
        beta1       = self.config["beta1"]
        beta2       = self.config["beta2"]
        n_class     = len(self.config["selectedStyleDir"])
        feature_w   = self.config["featureWeight"]
        transform_w = self.config["transformWeight"]
        dStep       = self.config["dStep"]
        gStep       = self.config["gStep"]
        # total_loader= self.dataloaders

        batchSize_list = self.config["batchSize_list"] 
        switch_step_list = self.config["switch_step_list"]
        imCropSize_list = self.config["imCropSize_list"]
        redefine_dataloader_flag = True

        if self.config["useTensorboard"]:
            from utilities.utilities import build_tensorboard
            tensorboard_writer = build_tensorboard(self.config["projectSummary"])
        
        print("build models...")

        if self.config["mode"] == "train":
            package = __import__("components."+self.config["gScriptName"], fromlist=True)
            GClass  = getattr(package, 'Generator')
            package = __import__("components."+self.config["dScriptName"], fromlist=True)
            DClass  = getattr(package, 'Discriminator')
        elif self.config["mode"] == "finetune":
            print("finetune load scripts from %s"%self.config["com_base"])
            package = __import__(self.config["com_base"]+self.config["gScriptName"], fromlist=True)
            GClass  = getattr(package, 'Generator')
            package = __import__(self.config["com_base"]+self.config["dScriptName"], fromlist=True)
            DClass  = getattr(package, 'Discriminator')

        Gen     = GClass(self.config["GConvDim"], self.config["GKS"], self.config["resNum"], n_class)
        Dis     = DClass(self.config["DConvDim"], self.config["DKS"], n_class)
        
        self.reporter.writeInfo("Generator structure:")
        self.reporter.writeModel(Gen.__str__())
        # print(self.Decoder)
        self.reporter.writeInfo("Discriminator structure:")
        self.reporter.writeModel(Dis.__str__())
        
        Transform   = Transform_block().cuda()
        Gen         = Gen.cuda()
        Dis         = Dis.cuda()
        
        print("build the optimizer...")
        # Loss and optimizer
        g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                    Gen.parameters()), lr_base, [beta1, beta2])

        d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                    Dis.parameters()), lr_base, [beta1, beta2])
        L1_loss     = torch.nn.L1Loss()
        MSE_loss    = torch.nn.MSELoss()
        Hinge_loss  = torch.nn.ReLU().cuda()
        # L1_loss     = torch.nn.SmoothL1Loss()


        if self.config["mode"] == "finetune":
            # checkpoint_path = os.path.join(self.config["projectCheckpoints"], "%d_checkpoint.pth"%self.config["checkpointStep"])
            model_path = os.path.join(self.config["projectCheckpoints"], "%d_Generator.pth"%self.config["checkpointStep"])
            checkpoint = torch.load(model_path)
            Gen.load_state_dict(checkpoint['g_model'])
            g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            print('loaded trained Generator model step {}...!'.format(self.config["checkpointStep"]))
            model_path = os.path.join(self.config["projectCheckpoints"], "%d_Discriminator.pth"%self.config["checkpointStep"])
            checkpoint = torch.load(model_path)
            Dis.load_state_dict(checkpoint['d_model'])
            d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            print('loaded trained Discriminator model step {}...!'.format(self.config["checkpointStep"]))
            
        # Start with trained model
        if self.config["mode"] == "finetune":
            start = self.config["checkpointStep"]
        else:
            start = 0

        # Initialization for parameter of dataloader
        current_step_index = None
        switch_step_nums = len(switch_step_list)
        for index in range(switch_step_nums-1):
            if start >= switch_step_list[index] and start < switch_step_list[index+1]:
                current_step_index = index
        if not current_step_index:
            if start >= switch_step_list[-1]:
                current_step_index = switch_step_nums-1



        output_size = Dis.get_outputs_len()
        
        # Data iterator
        print("prepare the dataloaders...")

        print("prepare the fixed labels...")
        fix_label   = [i for i in range(n_class)]
        fix_label   = torch.tensor(fix_label).long().cuda()

        # Start time
        import datetime
        print("Start to train at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        print('Start   ======  training...')
        start_time = time.time()
        for step in range(start, total_step):

            if current_step_index != switch_step_nums-1:
                if step >= switch_step_list[current_step_index+1]:
                    current_step_index = current_step_index + 1
                    redefine_dataloader_flag = True
                else:
                    pass
            else:
                pass
            
            if redefine_dataloader_flag:
                print(' ')
                print('Current step: {}'.format(step))
                print('***Redefining the dataloader for progressive training.***')
                print('***Current spatial size is {} and batch size is {}.***'.format(imCropSize_list[current_step_index], batchSize_list[current_step_index]))
                total_loader  = getLoader(self.config["style"], self.config["content"],
                                self.config["selectedStyleDir"],self.config["selectedContentDir"],
                                imCropSize_list[current_step_index], batchSize_list[current_step_index], self.config["dataloader_workers"])
                redefine_dataloader_flag = False
                print(' ')

                
            Dis.train()
            Gen.train()
            
            # ================== Train D ================== #
            # Compute loss with real images
            for _ in range(dStep):
                content_images,style_images,label  = total_loader.next()
                label           = label.long()
                d_out = Dis(style_images,label)
                d_loss_real = 0
                for i in range(output_size):
                    temp = Hinge_loss(1 - d_out[i]).mean()
                    # temp *= prep_weights[i]
                    d_loss_real += temp

                d_loss_photo = 0
                d_out = Dis(content_images,label)
                for i in range(output_size):
                    temp = Hinge_loss(1 + d_out[i]).mean()
                    d_loss_photo += temp

                fake_image,_ = Gen(content_images,label)
                d_out = Dis(fake_image.detach(),label)
                d_loss_fake = 0
                for i in range(output_size):
                    temp = Hinge_loss(1 + d_out[i]).mean()
                    d_loss_fake += temp

                # Backward + Optimize
                d_loss = d_loss_real + d_loss_photo + d_loss_fake
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
            
            # ================== Train G ================== #
            for _ in range(gStep):
                content_images,_,_  = total_loader.next()
                fake_image,real_feature = Gen(content_images,label)
                fake_feature            = Gen(fake_image, get_feature=True)
                d_out                   = Dis(fake_image,label.long())
                
                g_feature_loss          = L1_loss(fake_feature,real_feature)
                g_transform_loss        = MSE_loss(Transform(content_images), Transform(fake_image))
                g_loss_fake = 0
                for i in range(output_size):
                    temp = -d_out[i].mean()
                    # temp *= prep_weights[i]
                    g_loss_fake += temp

                # backward & optimize
                g_loss = g_loss_fake + g_feature_loss* feature_w + g_transform_loss* transform_w
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
            

            # Print out log info
            if (step + 1) % log_frep == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                epochinformation="[{}], Elapsed [{}], Step [{}/{}], d_loss: {:.4f}, d_loss_real: {:.4f}, d_loss_photo: {:.4f}, d_loss_fake: {:.4f}, g_loss: {:.4f}, g_loss_fake: {:.4f}, g_feature_loss: {:.4f}, g_transform_loss: {:.4f}".format(self.config["version"], elapsed, step + 1, total_step, 
                            d_loss.item(), d_loss_real.item(), d_loss_photo.item(), d_loss_fake.item(), g_loss.item(), g_loss_fake.item(),\
                                 g_feature_loss.item(), g_transform_loss.item())
                print(epochinformation)
                self.reporter.write_epochInf(epochinformation)
                
                if self.config["useTensorboard"]:
                    tensorboard_writer.add_scalar('data/d_loss', d_loss.item(), (step + 1))
                    tensorboard_writer.add_scalar('data/d_loss_real', d_loss_real.item(),(step + 1))
                    tensorboard_writer.add_scalar('data/d_loss_photo', d_loss_photo.item(),(step + 1))
                    tensorboard_writer.add_scalar('data/d_loss_fake', d_loss_fake.item(),(step + 1))
                    tensorboard_writer.add_scalar('data/g_loss', g_loss.item(), (step + 1))
                    tensorboard_writer.add_scalar('data/g_loss_fake', g_loss_fake.item(), (step + 1))
                    tensorboard_writer.add_scalar('data/g_feature_loss', g_feature_loss, (step + 1))
                    tensorboard_writer.add_scalar('data/g_transform_loss', g_transform_loss, (step + 1))

            # Sample images
            if (step + 1) % sample_freq == 0:
                torch.cuda.empty_cache()
                print('Sample images {}_fake.jpg'.format(step + 1))
                Gen.eval()
                with torch.no_grad():
                    sample = content_images[0, :, :, :].unsqueeze(0)
                    saved_image1 = denorm(sample.cpu().data)
                    for index in range(n_class):
                        fake_images,_ = Gen(sample, fix_label[index].unsqueeze(0))
                        saved_image1 = torch.cat((saved_image1, denorm(fake_images.cpu().data)), 0)
                    save_image(saved_image1,
                            os.path.join(sample_dir, '{}_fake.jpg'.format(step + 1)),nrow=3)

            if (step+1) % model_freq==0:
                print("Save step %d model checkpoints!"%(step+1))
                g_save_state = {'g_model': Gen.state_dict(),
                              'g_optimizer': g_optimizer.state_dict(),
                              }

                d_save_state = {'d_model': Dis.state_dict(),
                              'd_optimizer': d_optimizer.state_dict(),
                              }
                torch.save(g_save_state,
                           os.path.join(ckpt_dir, '{}_Generator.pth'.format(step + 1)))
                torch.save(d_save_state,
                           os.path.join(ckpt_dir, '{}_Discriminator.pth'.format(step + 1)))