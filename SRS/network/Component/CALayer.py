import torch.nn as nn
import torch
import torch as nd
class ECA_Layer(nn.Module):
    """
    Constructs a ECA module writed by NBK.
    """
    def __init__(self, channel, k_size=3):
        super(ECA_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.sigmoid(x)
        # y=x
        y = self.avg_pool(y)

        # 1, 512, 1, 1 -> 1, 512, 1 -> 1, 1, 512 -> 1, 1, 512 -> 1, 512, 1 -> 1, 512, 1, 1
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)


        return x * y.expand_as(x)