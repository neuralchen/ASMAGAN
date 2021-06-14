#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: LossUtili.py
# Created Date: Thursday October 10th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 14th October 2019 5:19:31 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

def __classificationLoss__(logit, target):
    """Compute binary cross entropy loss."""
    return F.binary_cross_entropy_with_logits(logit, target, reduction='sum')/logit.size(0)

def __hingeLoss__(logit, label):
    return nn.ReLU()(1.0 - label * logit).mean()# / logit.size(0)

def getClassifierLoss(classificationLossType):
        if classificationLossType == "hinge":
            return __hingeLoss__
        elif classificationLossType == "cross-entropy":
            return __classificationLoss__

def gradientPenalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).cuda()
    dydx = torch.autograd.grad(outputs=y,
                                inputs=x,
                                grad_outputs=weight,
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = (torch.sum(dydx**2, dim=1)).sqrt()
    return ((dydx_l2norm-1)**2).mean()

def gradientPenaltyWithRelu(fakeImages, realImages):
    """Compute gradient penalty: (max(0,L2_norm(dy/dx) - 1))**2."""
    weight = torch.ones(fakeImages.size()).cuda()
    dydx = torch.autograd.grad(outputs=fakeImages,
                                inputs=realImages,
                                grad_outputs=weight,
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = (torch.sum(dydx**2, dim=1)).sqrt()
    return (nn.ReLU()(dydx_l2norm-1)**2).mean()