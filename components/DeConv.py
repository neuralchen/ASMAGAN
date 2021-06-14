import torch
from torch import nn

class DeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, upsampl_scale = 2):
        super().__init__()
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=upsampl_scale)
        padding_size = int((kernel_size -1)/2)
        # self.same_padding   = nn.ReflectionPad2d(padding_size)
        self.conv           = nn.Conv2d(in_channels = in_channels ,padding=padding_size, out_channels = out_channels , kernel_size= kernel_size, bias= False)
        self.__weights_init__()

    def __weights_init__(self):
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, input):
        h   = self.upsampling(input)
        # h   = self.same_padding(h)
        h   = self.conv(h)
        return h