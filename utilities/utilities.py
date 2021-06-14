#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: utilities.py
# Created Date: Monday April 6th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 7th April 2020 12:42:23 am
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


def build_tensorboard(summary_path):
    from tensorboardX import SummaryWriter
    # from logger import Logger
    # self.logger = Logger(self.log_path)
    writer = SummaryWriter(log_dir=summary_path)
    return writer

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)