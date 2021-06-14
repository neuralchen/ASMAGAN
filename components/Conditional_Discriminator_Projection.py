#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Conditional_Discriminator copy.py
# Created Date: Saturday April 18th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 18th April 2020 11:26:51 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F

from torch.nn import utils

class Discriminator(nn.Module):
    def __init__(self, chn=32, k_size=3, n_class=3):
        super().__init__()
        # padding_size = int((k_size -1)/2)
        slop         = 0.2
        enable_bias  = True

        # stage 1
        self.block1 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels = 3 , out_channels = chn , kernel_size= k_size, stride = 2, padding=2,bias= enable_bias)),
            nn.LeakyReLU(slop),
            utils.spectral_norm(nn.Conv2d(in_channels = chn, out_channels = chn * 2 , kernel_size= k_size, stride = 2,padding=2, bias= enable_bias)), # 1/4
            nn.LeakyReLU(slop)
        )
        self.aux_classfier1 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels = chn * 2 , out_channels = chn , kernel_size= 5, bias=enable_bias)),
            nn.LeakyReLU(slop),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embed1 = utils.spectral_norm(nn.Embedding(n_class, chn))
        self.linear1= utils.spectral_norm(nn.Linear(chn, 1))

        # stage 2
        self.block2 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels = chn * 2 , out_channels = chn * 4 , kernel_size= k_size, stride = 2, padding=2, bias= enable_bias)),# 1/8
            nn.LeakyReLU(slop),
            utils.spectral_norm(nn.Conv2d(in_channels = chn * 4, out_channels = chn * 8 , kernel_size= k_size, stride = 2,padding=2, bias= enable_bias)),# 1/16
            nn.LeakyReLU(slop)
        )
        self.aux_classfier2 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels = chn * 8 , out_channels = chn , kernel_size= 5, bias= enable_bias)),
            nn.LeakyReLU(slop),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embed2 = utils.spectral_norm(nn.Embedding(n_class, chn))
        self.linear2= utils.spectral_norm(nn.Linear(chn, 1))

        # stage 3
        self.block3 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels = chn * 8 , out_channels = chn * 8 , kernel_size= k_size, stride = 2,padding=3, bias= enable_bias)),# 1/32
            nn.LeakyReLU(slop),
            utils.spectral_norm(nn.Conv2d(in_channels = chn * 8, out_channels = chn * 16 , kernel_size= k_size, stride = 2,padding=3, bias= enable_bias)),# 1/64
            nn.LeakyReLU(slop)
        )
        self.aux_classfier3 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels = chn * 16 , out_channels = chn, kernel_size= 5, bias= enable_bias)),
            nn.LeakyReLU(slop),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embed3 = utils.spectral_norm(nn.Embedding(n_class, chn))
        self.linear3= utils.spectral_norm(nn.Linear(chn, 1))
        self.__weights_init__()

    def __weights_init__(self):
        print("Init weights")
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                try:
                    nn.init.zeros_(m.bias)
                except:
                    print("No bias found!")

            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, input, condition):

        h = self.block1(input)
        prep1 = self.aux_classfier1(h)
        prep1 = prep1.view(prep1.size()[0], -1)
        y1 = self.embed1(condition)
        y1 = torch.sum(y1 * prep1, dim=1, keepdim=True)
        prep1 = self.linear1(prep1) + y1

        h = self.block2(h)
        prep2 = self.aux_classfier2(h)
        prep2 = prep2.view(prep2.size()[0], -1)
        y2 = self.embed2(condition)
        y2 = torch.sum(y2 * prep2, dim=1, keepdim=True)
        prep2 = self.linear2(prep2) + y2

        h = self.block3(h)
        prep3 = self.aux_classfier3(h)
        prep3 = prep3.view(prep3.size()[0], -1)
        y3 = self.embed3(condition)
        y3 = torch.sum(y3 * prep3, dim=1, keepdim=True)
        prep3 = self.linear3(prep3) + y3

        out_prep = [prep1,prep2,prep3]
        return out_prep
    
    def get_outputs_len(self):
        num = 0
        for m in self.modules():
            if isinstance(m,nn.Linear):
                num+=1
        return num

if __name__ == "__main__":
    wo = Discriminator().cuda()
    from torchsummary import summary
    summary(wo, input_size=(3, 512, 512))