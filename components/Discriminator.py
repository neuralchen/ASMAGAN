import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Discriminator(nn.Module):
    def __init__(self, chn=32, k_size=5):
        super().__init__()
        padding_size = int((k_size -1)/2)
        slop         = 0.2
        self.block0 = nn.Sequential(
            nn.Conv2d(in_channels = 3 , out_channels = chn*2, kernel_size= k_size, stride = 2, bias= False),
            nn.InstanceNorm2d(chn*2, affine=True, momentum=0),
            nn.LeakyReLU(slop)
        )
        self.aux_classfier0 = nn.Sequential(
            nn.Conv2d(in_channels = chn * 2 , out_channels = 1 , kernel_size= 5),
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = chn * 2 , out_channels = chn * 2 , kernel_size= k_size, stride = 2, bias= False),
            nn.InstanceNorm2d(chn * 2 , affine=True, momentum=0),
            nn.LeakyReLU(slop),
        )
        self.aux_classfier1 = nn.Sequential(
            nn.Conv2d(in_channels = chn * 2 , out_channels = 1, kernel_size= 10),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = chn * 2 , out_channels = chn * 4 , kernel_size= k_size, stride = 2, bias= False),
            nn.InstanceNorm2d(chn * 4, affine=True, momentum=0),
            nn.LeakyReLU(slop),
            nn.Conv2d(in_channels = chn * 4 , out_channels = chn * 8 , kernel_size= k_size, stride = 2, bias= False),
            nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
            nn.LeakyReLU(slop)
        )
        self.aux_classfier3 = nn.Sequential(
            nn.Conv2d(in_channels = chn * 8 , out_channels = 1, kernel_size= 10),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels = chn * 8 , out_channels = chn * 8 , kernel_size= k_size, stride = 2, bias= False),
            nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
            nn.LeakyReLU(slop),
            nn.Conv2d(in_channels = chn * 8, out_channels = chn * 16 , kernel_size= k_size, stride = 2, bias= False),
            nn.InstanceNorm2d(chn * 16 , affine=True, momentum=0),
            nn.LeakyReLU(slop)
        )
        self.aux_classfier5 = nn.Sequential(
            nn.Conv2d(in_channels = chn * 16 , out_channels = 1, kernel_size= 6),
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels = chn * 16 , out_channels = chn * 16 , kernel_size= k_size, stride = 2, bias= False),
            nn.InstanceNorm2d(chn * 16, affine=True, momentum=0),
            nn.LeakyReLU(slop)
        )
        self.classfier = nn.Conv2d(in_channels = chn * 16, out_channels = 1 , kernel_size= 3)
        self.__weights_init__()

    def __weights_init__(self):
        print("Init weights")
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, input):
        
        h       = self.block0(input)
        prep0   = self.aux_classfier0(h)
        h       = self.block1(h)
        prep1   = self.aux_classfier1(h)
        h       = self.block2(h)
        prep3   = self.aux_classfier3(h)
        h       = self.block4(h)
        prep5   = self.aux_classfier5(h)
        h       = self.block6(h)
        out     = self.classfier(h)
        out_prep = [prep0,prep1,prep3,prep5,out]
        return out_prep

if __name__ == "__main__":
    wo = Discriminator()