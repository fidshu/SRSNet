import torch.nn as nn
import torch
import torch.nn.functional as F
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


class PKernel_D2(nn.Module):
    def __init__(self,input_channel = 8,output_channel = 1,kernel_size =1,batch_size=8,infor_channel = 64):
        super(PKernel_D2, self).__init__()
        
        self.output_channel = output_channel
        self.input_channel = input_channel
        self.kernel_size = kernel_size
        #这里原本是batch * input_channel*output_channel*kernel_size*kernel_size
        #但是avg自带是对后2项进行avg。所以天然自带batch和input_channel
        #外加hidden planes 自带batch，所以,hidden planes = input_channel*output_channel*kernel_size*kernel_size
        self.channel_need = output_channel * kernel_size * kernel_size
        self.hidden_planes = input_channel*output_channel*kernel_size*kernel_size
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv = nn.Sequential(
            nn.Conv2d(infor_channel,infor_channel,kernel_size=1,bias = True), #这里过后tensor[batch,self.channel_need,xxx,xxx]
            self.avgpool,#这里过后tensor[batch,self.channel_need,1,1]
            nn.Conv2d(infor_channel, self.hidden_planes, kernel_size=1, bias=True,groups = infor_channel),#这里过后tensor = [batch,input_channel*output_channel*kernel_size*kernel_size]
        )
        self.ca = ChannelAttention(infor_channel)
        self.sa = SpatialAttention()

    def forward(self, x1, weight):
        weight =self.sa(self.ca(weight)*weight)*weight
        # X1 分割网络输入 weight 重建网络输入
        kernel = self.Conv(weight)
        #####################################################################################################
        batch_size, in_planes, height, width = x1.size()
        kernel = kernel.view(batch_size*self.output_channel,self.input_channel,self.kernel_size , self.kernel_size)
        x = x1.view(1, -1, height, width)# 变化成一个维度进行组卷积
        out = F.conv2d(x,kernel,bias=None,padding=int((self.kernel_size-1)/2),stride = 1, groups = batch_size)
        out = out.view(batch_size, self.output_channel, out.size(-2), out.size(-1))
        #####################################################################################################
        return out


 # self.PKernel_D2 = PKernel_D2(input_channel = nb_filter[3],output_channel = nb_filter[3], kernel_size = 5,batch_size=self.bs,infor_channel = nb_filter[3])