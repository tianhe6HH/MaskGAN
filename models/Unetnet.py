from mpmath.functions.signals import sigmoid
from torch import nn
from torch.nn import functional as F
import torch
"""
Unet分为三个模块，卷积模块，下采样和上采样模块
"""
class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            # first conv
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(0.2),
            # second conv
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(0.2),
        )

    def forward(self,x):
        return self.layer(x)

class DownSample(nn.Module):
    # conv 代替 maxpool，减少特征损失
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            # stride = 2可以将图像的长宽变为1/2
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.layer(x)

class UpSample(nn.Module):  #降通道
    def __init__(self,channel):
        super(UpSample,self).__init__()
        self.layer=nn.Conv2d(channel, channel//2,1,1)  # #1*1的卷积是为了降通道

    def forward(self,x,feature_map):
        up = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True)
        out= self.layer(up)
        return torch.cat([out,feature_map],1)  # NCHW,在C通道进行的，所以是1

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.c1=Conv_Block(3,64)
        self.d1= DownSample(64)
        self.c2=Conv_Block(64,128)
        self.d2= DownSample(128)
        self.c3=Conv_Block(128,256)
        self.d3= DownSample(256)
        self.c4=Conv_Block(256,512)
        self.d4= DownSample(512)
        self.c5=Conv_Block(512,1024)
        self.u1=UpSample(1024)
        self.c6=Conv_Block(1024,512)
        self.u2=UpSample(512)
        self.c7=Conv_Block(512,256)
        self.u3=UpSample(256)
        self.c8=Conv_Block(256,128)
        self.u4=UpSample(128)
        self.c9=Conv_Block(128,64)
        self.out=nn.Conv2d(64,3,1,1)
        self.Th=nn.Sigmoid() # 每个像素点是否有颜色，二分类

    def forward(self,x):
        R1=self.c1(x)
        R2=self.c2(self.d1(R1))
        R3=self.c3(self.d2(R2))
        R4=self.c4(self.d3(R3))
        R5=self.c5(self.d4(R4))
        O1=self.c6(self.u1(R5,R4))
        O2=self.c7(self.u2(O1,R3))
        O3=self.c8(self.u3(O2,R2))
        O4=self.c9(self.u4(O3,R1))
        return self.Th(self.out(O4))

if __name__ == '__main__':
    x=torch.randn(2,3,256,256)
    net=Unet()
    print(net(x).shape)






