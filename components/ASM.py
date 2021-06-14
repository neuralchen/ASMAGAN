import torch
from torch import nn
import numpy as np
from components.DeConv import  DeConv



class ASM(nn.Module):
    def __init__(self, in_channels, out_channel,attr_dim, kernel_size = 3, up_scale = 4, norm='none'):
        super().__init__()
        self.n_class=attr_dim
        if norm == 'bn':

            self.gate = nn.Sequential(
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(out_channel, affine=True),
                nn.Sigmoid(),
            )

        elif norm == 'in':
            self.gate_r = nn.Sequential(
                nn.Conv2d(in_channels=out_channel * 4, out_channels=out_channel, kernel_size=kernel_size,padding=1, bias=False),
                #nn.InstanceNorm2d(out_channel),
                nn.Sigmoid(),
            )

            self.gate_z = nn.Sequential(
                nn.Conv2d(in_channels=out_channel * 4, out_channels=out_channel, kernel_size=kernel_size,padding=1, bias=False),
                #nn.InstanceNorm2d(out_channel),
                nn.Sigmoid(),
            )
            self.conv_concat = nn.Sequential(
                nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_channel),
                nn.Tanh(),
            )
            
        else:
            self.gate = nn.Sequential(
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, padding=1,
                          bias=False),
                nn.Sigmoid(),
            )

        self.dconv = DeConv(in_channels=(in_channels+attr_dim), out_channels=out_channel, upsampl_scale = up_scale)

        self.gmp = nn.AdaptiveMaxPool2d((1,1))

        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.tanh = nn.Tanh()

    def forward(self, in_data,state,classid):
        # print(type(state))

        n, _, h, w  = state.size()

        conditionnp=np.full((n, self.n_class), -1.0)
        for index , id in enumerate(classid):
            conditionnp[index][id]=1.0
        
        condition=torch.from_numpy(conditionnp).float()
        a = condition.cuda()

        attr        = a.view((n, -1, 1, 1)).expand((n, -1, h, w))
        state  = torch.cat([state, attr], dim=1)
        x_t = self.dconv(state)  # upsample and make `channel` identical to `out_channel`

        concat_1 = torch.cat([x_t, in_data], dim=1)
        concat_1_gmp = self.gmp(concat_1)
        concat_1_gap = self.gap(concat_1)
        concat_1_gmpconcatgap = torch.cat([concat_1_gmp, concat_1_gap], dim=1)

        r_t = self.gate_r(concat_1_gmpconcatgap)
        z_t = self.gate_z(concat_1_gmpconcatgap)

        c_t = r_t * x_t 

        concat_ct_x = torch.cat([c_t, in_data], dim=1)
        
        h_hat = self.conv_concat(concat_ct_x)

        h_t = z_t * h_hat + (1 - z_t) * x_t

        return h_t