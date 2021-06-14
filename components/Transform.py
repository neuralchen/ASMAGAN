import torch
from torch import nn

class Transform_block(nn.Module):
    def __init__(self, k_size = 10):
        super().__init__()
        padding_size = int((k_size -1)/2)
        # self.padding = nn.ReplicationPad2d(padding_size)
        self.pool = nn.AvgPool2d(k_size, stride=1,padding=padding_size)

    def forward(self, input_image):
        # h = self.padding(input)
        out = self.pool(input_image)
        return out