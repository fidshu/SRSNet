import torch.nn as nn
import torch
import torch as nd
import torchvision.transforms as F
from CALayer import ECA_Layer
from IPALC import IPATR

from SPD import Spu
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Dense_BasicBlock_MALC(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1,batch_size =8):
        super(Dense_BasicBlock_MALC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.CONV = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        

        out =self.CONV(out)
        
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out



class ResNetInput(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 bs=4):
        super(ResNetInput, self).__init__()
        self.OutChannel = 16
        self.conv1 = nn.Sequential(
          nn.Conv2d(3,self.OutChannel,stride = 1 ,padding =1 ,kernel_size=3,bias=True),
          nn.BatchNorm2d(self.OutChannel),
          nn.ReLU(inplace=True)
        )
        nb_filter = [8,16,32,64]
        self.nb = nb_filter
        self.bs = bs
        self.conv0_0 = self._make_layer(block,self.OutChannel,nb_filter[0],blocks_num[0])
        self.conv1_0 = self._make_layer(block,nb_filter[0],nb_filter[1],blocks_num[1])
        self.conv2_0 = self._make_layer(block,nb_filter[1],nb_filter[2],blocks_num[2])
        self.conv3_0 = self._make_layer(block,nb_filter[2],nb_filter[3],blocks_num[3])

        self.conv0_1 = self._make_layer(block,nb_filter[0]+nb_filter[1],nb_filter[0],blocks_num[0])
        self.conv1_1 = self._make_layer(block,nb_filter[1]+nb_filter[2]+nb_filter[0],nb_filter[1],blocks_num[1])     
        self.conv2_1 = self._make_layer(block,nb_filter[2]+nb_filter[3]+nb_filter[1],nb_filter[2],blocks_num[2])    

        self.conv0_2 = self._make_layer(block,nb_filter[0]+nb_filter[0]+nb_filter[1],nb_filter[0],blocks_num[0])
        self.conv1_2 = self._make_layer(block,nb_filter[1]+nb_filter[1]+nb_filter[2]+nb_filter[0],nb_filter[1],blocks_num[1])    

        self.conv0_3 = self._make_layer(block,nb_filter[0]+nb_filter[0]+nb_filter[1],nb_filter[0],blocks_num[0])
        
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.pool  = nn.MaxPool2d(2, 2)

        self.bottomuplocal_fpn_3 = BLAM(channel=nb_filter[2])
        self.bottomuplocal_fpn_2 = BLAM(channel=nb_filter[1])
        self.bottomuplocal_fpn_1 = BLAM(channel=nb_filter[0])

        self.IPATR2 = IPATR(nb_filter[2])
        self.IPATR1 = IPATR(nb_filter[1])
        self.IPATR0 = IPATR(nb_filter[0])

        self.fup3 =  Spu(nb_filter[3],nb_filter[2])
        self.fup2 =  Spu(nb_filter[2],nb_filter[1])
        self.fup1 =  Spu(nb_filter[1],nb_filter[0])
        
        self.finalconv = self._make_layer(block,8*3,nb_filter[0])
        
        self.mask_head_1x1 = nn.Conv2d(nb_filter[0],3,kernel_size = 1,bias = True)
        
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)        
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1,DyFactor =0):
        layers = []
        layers.append(block(input_channels, output_channels,batch_size = self.bs))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels,batch_size = self.bs))
        return nn.Sequential(*layers)

    def UpSC(self,feature):
        _,channel,_,_, = feature.shape
        if(channel == self.nb[3]):
          out = self.fup3(feature)
        elif(channel == self.nb[2]):
          out = self.fup2(feature)
        elif(channel == self.nb[1]):
          out = self.fup1(feature)
        return out

    def forward(self,x):
        _, _, orig_hei, orig_wid = x.shape
        x = self.conv1(x)

        ################### D E N S E##############################################
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x0_1 = self.conv0_1(torch.cat([x0_0,self.up(x1_0)],dim = 1))
        x1_1 = self.conv1_1(torch.cat([x1_0,self.up(x2_0),self.pool(x0_1)],dim = 1))
        x2_1 = self.conv2_1(torch.cat([x2_0,self.up(x3_0),self.pool(x1_1)],dim = 1))

        x0_2 = self.conv0_2(torch.cat([x0_1,self.up(x1_1),x0_0],dim = 1))
        x1_2 = self.conv1_2(torch.cat([x1_1,self.up(x2_1),x1_0,self.pool(x0_2)],dim = 1))

        x0_3 = self.conv0_3(torch.cat([x0_2,self.up(x1_2),x0_1],dim = 1))
        ###################### F U S E ##########################################

        f_3 = self.bottomuplocal_fpn_3(self.IPATR2(x2_1),self.UpSC(x3_0))
        f_2 = self.bottomuplocal_fpn_2(self.IPATR1(x1_2),self.UpSC(f_3))
        f_1 = self.bottomuplocal_fpn_1(self.IPATR0(x0_3),self.UpSC(f_2))

        
        final = torch.cat([f_1,self.up(self.conv0_1_1x1(f_2)),self.up_4(self.conv0_2_1x1(f_3))],1)
        final = self.finalconv(final)
        
        pre   = self.mask_head_1x1(final)
        
        #############################################################################
        # return x3_0
        return pre
class BLAM(nn.Module):
    def __init__(self,
            channel):
      super(BLAM, self).__init__()
      self.channel = channel
      inter_channel = int(channel // 1)
      self.bn1 = nn.BatchNorm2d(self.channel)
      self.bn2 = nn.BatchNorm2d(self.channel)

      self.conv1 = nn.Conv2d(channel,inter_channel,kernel_size=1,padding=0,stride=1,bias = True)
      self.bn3  = nn.BatchNorm2d(inter_channel)
      self.relu1  = nn.ReLU(inplace=True)
      self.conv2 = nn.Conv2d(inter_channel,channel,kernel_size=1,padding=0,stride=1,bias = True)
      self.bn4  = nn.BatchNorm2d(channel)
      self.relu2 = nn.ReLU(inplace=True)


      self.ECALayer = ECA_Layer(channel)

      self.finalconv = nn.Sequential(
        nn.Conv2d(2 * channel,channel,kernel_size=1,stride=1,bias = True),
        nn.BatchNorm2d(channel),
        nn.ReLU(inplace=True)
      )
      
    def forward(self,low,residual):
      temp = low

      low = self.conv1(low)
      low = self.bn3(low)
      low = self.relu1(low)
      low = self.conv2(low)
      low = self.bn4(low)
      low = self.relu2(low)

      combine = torch.cat((self.ECALayer(residual),low), dim =1)
      blam = self.finalconv(combine)

      return blam

def SRS_R(bs = 4):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    print("hello SRS_R ")
    return ResNetInput(Dense_BasicBlock_MALC, [5, 5, 5, 5],bs)