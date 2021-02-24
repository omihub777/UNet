import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=0, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)
        self.bn = nn.BatchNorm2d(out_c)
        
    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        return out

class UpConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=2, s=2, p=0, bias=False):
        super(UpConvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        # Need bn & act?
        out = F.relu(self.bn(self.conv(x)))
        return out

# class ResBlock(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(ResBlock, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_c)
#         self.conv1 = nn.Conv2d()

if __name__=="__main__":
    x = torch.randn(2,1,32,32)
    # n = ConvBlock(1, 16)
    n = UpConvBlock(1, 16)
    out = n(x)
    print(out.shape)