import torch
from torch import nn
# from ops.Conditional_BN import Conditional_BN
from components.Adain import Adain

class Conditional_ResBlock(nn.Module):
    def __init__(self, in_channel, k_size = 3, n_class = 2, stride=1):
        super().__init__()
        padding_size = int((k_size -1)/2)
        self.same_padding1  = nn.ReplicationPad2d(padding_size)
        self.conv1          = nn.Conv2d(in_channels = in_channel , out_channels = in_channel, kernel_size= k_size, stride=stride, bias= False)
        self.adain1         = Adain(in_channel,n_class)
        self.same_padding2  = nn.ReplicationPad2d(padding_size)
        self.conv2          = nn.Conv2d(in_channels = in_channel , out_channels = in_channel, kernel_size= k_size, stride=stride, bias= False)
        self.adain2         = Adain(in_channel,n_class)


    def forward(self, input, condition):
        res = input
        h   = self.same_padding1(input)
        h   = self.conv1(h)
        h   = self.adain1(h,condition)
        h   = self.same_padding2(h)
        h   = self.conv2(h)
        h   = self.adain2(h,condition)
        out = h + res
        return out