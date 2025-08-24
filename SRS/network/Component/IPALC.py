import torch.nn as nn
import torch
import torch as nd
class IPATR(nn.Module):
    def __init__(self,channel):
      super(IPATR, self).__init__()
      self.channel = channel
      self.conv1 = nn.Conv2d(self.channel,self.channel,kernel_size=1,padding=0,bias = True)
      

      self.conv2 = nn.Conv2d(self.channel,self.channel,kernel_size=3,padding=1,stride=1,bias = True)
      

      self.conv3 = nn.Conv2d(self.channel,self.channel,kernel_size=3,padding=2,stride=1,dilation=2,bias = True)
      self.sg = torch.nn.Sigmoid()

      self.tr = nn.Conv2d(3 * self.channel,self.channel,kernel_size=3,padding=2,stride=1,dilation=2,bias = True)
      self.tr_1 =nn.Conv2d(2 * self.channel,self.channel,kernel_size=3,padding=2,stride=1,dilation=2,bias = True)
    def ipatr(self,x,choice):
      if(choice ==1):
        conv = self.conv1
      elif(choice == 2):
        conv = self.conv2
      else:
        conv = self.conv3
      mu = conv(x)
      segma = self.sg(conv(x-mu))
      mscn = torch.div(x-mu+0.001,segma+0.001)
      # print(segma)
      return mscn
    def forward(self,x):
      x1 = self.ipatr(x,choice = 1)
      x2 = self.ipatr(x,choice = 2)
      x3 = self.ipatr(x,choice = 3)
      mpcm =torch.div(x1+x2+x3,3)


      # mpcm =torch.cat((x1,x2,x3),dim = 1)
      # mpcm = self.tr(mpcm)
      mpcm = self.sg(mpcm)
      mpcm = x+mpcm*x

      return mpcm