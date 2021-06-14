#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: GradientPenalty.py
# Created Date: Thursday October 10th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 15th April 2020 7:32:02 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

def GradientPenalty(y, x):
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

