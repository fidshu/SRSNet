import torch.nn as nn
import torch
import torch as nd
class Spu(nn.Module):
    def __init__(self,in_channel,out_channel):
      super(Spu, self).__init__()
      self.pixel_shuffle = nn.PixelShuffle(2)
      self.conv = nn.Sequential(
              nn.Conv2d(int(in_channel / 4), out_channel, kernel_size=3, stride=1,padding = 1, bias=True),
              nn.BatchNorm2d(out_channel),
              nn.ReLU(inplace=True))
     
    def forward(self,x):
      out = self.pixel_shuffle(x)
      out = self.conv(out)
      return out