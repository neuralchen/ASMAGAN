#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: StyleResize.py
# Created Date: Friday April 17th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 18th April 2020 1:39:53 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################
from PIL import Image
import torchvision.transforms.functional as F

class StyleResize(object):

    def __call__(self, images):
        th, tw = images.size # target height, width
        if max(th,tw) > 1800:
            alpha = 1800. / float(min(th,tw))
            h     = int(th*alpha)
            w     = int(tw*alpha)
            images  = F.resize(images, (h, w))
        if max(th,tw) < 800:
            # Resize the smallest side of the image to 800px
            alpha = 800. / float(min(th,tw))
            if alpha < 4.:
                h     = int(th*alpha)
                w     = int(tw*alpha)
                images  = F.resize(images, (h, w))
            else:
                images  = F.resize(images, (800, 800))
        return images

    def __repr__(self):
        return self.__class__.__name__ + '()'