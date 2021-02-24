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

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, s=2):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=s, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)

        if in_c!=out_c or s!=1:
            self.skip = nn.Conv2d(in_c, out_c, kernel_size=1, stride=s, bias=False)
        else:
            self.skip = nn.Sequential()

    def forward(self, x):
        x = F.relu(self.bn1(x))
        out = self.conv1(x)
        out = self.conv2(F.relu(self.bn2(out)))

        return out + self.skip(x)


if __name__=="__main__":
    x = torch.randn(2,1,32,32)
    # n = ConvBlock(1, 16)
    n = UpConvBlock(1, 16)
    out = n(x)
    print(out.shape)