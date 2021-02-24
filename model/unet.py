import sys, os
sys.path.append(os.path.abspath("model"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

from layers import ConvBlock, UpConvBlock, ResBlock

class VanillaUNet(nn.Module):
    """Different input/output size"""
    def __init__(self, in_c, out_c):
        super(VanillaUNet, self).__init__()
        self.enc1 = nn.Sequential(
            ConvBlock(in_c, 64),
            ConvBlock(64, 64),
        )
        self.enc2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128)
        )
        self.enc3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256)
        )
        self.enc4 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512)
        )
        self.trans = nn.Sequential(
            ConvBlock(512, 1024),
            ConvBlock(1024, 1024),
        )
        self.upconv1 = UpConvBlock(1024, 512)
        self.dec1 = nn.Sequential(
            ConvBlock(1024, 512),
            ConvBlock(512, 512)
        )
        self.upconv2 = UpConvBlock(512, 256)
        self.dec2 = nn.Sequential(
            ConvBlock(512, 256),
            ConvBlock(256, 256)
        )
        self.upconv3 = UpConvBlock(256, 128)
        self.dec3 = nn.Sequential(
            ConvBlock(256, 128),
            ConvBlock(128, 128)
        )
        self.upconv4 = UpConvBlock(128, 64)
        self.dec4 = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 64)
        )
        self.last = nn.Conv2d(64, out_c, kernel_size=1)

    def forward(self, x):
        out1 = self.enc1(x)
        out2 = self.enc2(F.max_pool2d(out1, 2))
        out3 = self.enc3(F.max_pool2d(out2, 2)) 
        out4 = self.enc4(F.max_pool2d(out3, 2))

        out = self.trans(F.max_pool2d(out4, 2))

        out = self.upconv1(out)
        out = self.dec1(self._crop_cat(out4, out))
        out = self.upconv2(out)
        out = self.dec2(self._crop_cat(out3, out))
        out = self.upconv3(out)
        out = self.dec3(self._crop_cat(out2, out))
        out = self.upconv4(out)
        out = self.dec4(self._crop_cat(out1, out))
        out = self.last(out)
        return out

    @staticmethod
    def _crop_cat(data1, data2):
        _, _, h, w = data1.size()
        _, _, h_ref, w_ref = data2.size()
        x1 = int((w-w_ref)/2)
        x2 = x1 + w_ref
        y1 = int((h-h_ref)/2)
        y2 = y1 + h_ref
        out = torch.cat([data1[:, :, y1:y2, x1:x2], data2], dim=1)
        return out


class UNet(nn.Module):
    def __init__(self, in_c: int, out_c: int , filters: int=44):
        super(UNet, self).__init__()
        self.enc1 = nn.Sequential(
            ConvBlock(in_c, filters, p=1),
            ConvBlock(filters, filters, p=1),
        )
        self.enc2 = nn.Sequential(
            ConvBlock(filters, filters*2, p=1),
            ConvBlock(filters*2, filters*2, p=1)
        )
        self.enc3 = nn.Sequential(
            ConvBlock(filters*2, filters*4, p=1),
            ConvBlock(filters*4, filters*4, p=1)
        )
        self.enc4 = nn.Sequential(
            ConvBlock(filters*4, filters*8, p=1),
            ConvBlock(filters*8, filters*8, p=1)
        )
        self.trans = nn.Sequential(
            ConvBlock(filters*8, filters*16, p=1),
            ConvBlock(filters*16, filters*16, p=1),
        )
        self.upconv1 = UpConvBlock(filters*16, filters*8)
        self.dec1 = nn.Sequential(
            ConvBlock(filters*16, filters*8, p=1),
            ConvBlock(filters*8, filters*8, p=1)
        )
        self.upconv2 = UpConvBlock(filters*8, filters*4)
        self.dec2 = nn.Sequential(
            ConvBlock(filters*8, filters*4, p=1),
            ConvBlock(filters*4, filters*4, p=1)
        )
        self.upconv3 = UpConvBlock(filters*4, filters*2)
        self.dec3 = nn.Sequential(
            ConvBlock(filters*4, filters*2, p=1),
            ConvBlock(filters*2, filters*2, p=1)
        )
        self.upconv4 = UpConvBlock(filters*2, filters)
        self.dec4 = nn.Sequential(
            ConvBlock(filters*2, filters, p=1),
            ConvBlock(filters, filters, p=1)
        )
        self.last = nn.Conv2d(filters, out_c, kernel_size=1)

    def forward(self, x):
        out1 = self.enc1(x)
        out2 = self.enc2(F.max_pool2d(out1, 2))
        out3 = self.enc3(F.max_pool2d(out2, 2)) 
        out4 = self.enc4(F.max_pool2d(out3, 2))

        out = self.trans(F.max_pool2d(out4, 2))

        out = self.upconv1(out)
        out = self.dec1(torch.cat([out4, out], dim=1))
        out = self.upconv2(out)
        out = self.dec2(torch.cat([out3, out], dim=1))
        out = self.upconv3(out)
        out = self.dec3(torch.cat([out2, out], dim=1))
        out = self.upconv4(out)
        out = self.dec4(torch.cat([out1, out], dim=1))
        out = self.last(out)
        return out


class ResUNet(nn.Module):
    """Residual-modified UNet. Note that we don't use a residual block in the inital block.
    This is because the architecture of ResNet dosen't use residual block in the very first layer either.
    """
    def __init__(self, in_c: int, out_c: int , filters: int=44):
        super(ResUNet, self).__init__()
        self.enc1 = nn.Sequential(
            ConvBlock(in_c, filters, p=1),
            ConvBlock(filters, filters, p=1),
        )
        self.enc2 = ResBlock(filters, filters*2)
        self.enc3 = ResBlock(filters*2, filters*4)
        self.enc4 = ResBlock(filters*4, filters*8)

        self.trans = ResBlock(filters*8, filters*16)

        self.upconv1 = UpConvBlock(filters*16, filters*8)
        self.dec1 = ResBlock(filters*16, filters*8, s=1)
        self.upconv2 = UpConvBlock(filters*8, filters*4)
        self.dec2 = ResBlock(filters*8, filters*4, s=1)
        self.upconv3 = UpConvBlock(filters*4, filters*2)
        self.dec3 = ResBlock(filters*4, filters*2, s=1)
        self.upconv4 = UpConvBlock(filters*2, filters)
        self.dec4 = ResBlock(filters*2, filters, s=1)
        self.last = nn.Conv2d(filters, out_c, kernel_size=1)

    def forward(self, x):
        out1 = self.enc1(x)
        out2 = self.enc2(out1)
        out3 = self.enc3(out2) 
        out4 = self.enc4(out3)

        out = self.trans(out4)

        out = self.upconv1(out)
        out = self.dec1(torch.cat([out4, out], dim=1))
        out = self.upconv2(out)
        out = self.dec2(torch.cat([out3, out], dim=1))
        out = self.upconv3(out)
        out = self.dec3(torch.cat([out2, out], dim=1))
        out = self.upconv4(out)
        out = self.dec4(torch.cat([out1, out], dim=1))
        out = self.last(out)
        return out


if __name__=="__main__":
    b, c, h, w = 2,1,224,224
    x = torch.randn(b, c, h, w)
    # n = UNet(c, 1)
    n = ResUNet(c, 1)
    out = n(x)
    torchsummary.summary(n, (c, h, w))
