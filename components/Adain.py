import torch
from torch import nn

class Adain(nn.Module):
    def __init__(self, in_channel, n_class=6):
        super().__init__()

        self.instance_norm = nn.InstanceNorm2d(in_channel,momentum=0, affine=False)

        self.embed1 = nn.Embedding(n_class, in_channel)
        self.embed2 = nn.Embedding(n_class, in_channel)
        # self.embed.weight.data[:, :in_channel] = 1
        # self.embed.weight.data[:, in_channel:] = 0
        nn.init.xavier_uniform_(self.embed1.weight)
        nn.init.xavier_uniform_(self.embed2.weight)
        
    def forward(self, input, class_id):
        out     = self.instance_norm(input)
        sigma   = self.embed1(class_id)
        mu      = self.embed2(class_id)
        # sigma, mu = embed.chunk(2, 1)
        sigma = sigma.unsqueeze(2).unsqueeze(3)
        mu = mu.unsqueeze(2).unsqueeze(3)
        out = sigma * out + mu

        return out