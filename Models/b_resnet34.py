import torch
import torch.nn as nn
import torch.nn.functional as F

#https://github.com/yuhuixu1993/BNET/blob/main/classification/imagenet/models/resnet.py
IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]
class RGB(nn.Module):
    def __init__(self,):
        super(RGB, self).__init__()
        self.register_buffer('mean', torch.zeros(1,3,1,1))
        self.register_buffer('std', torch.ones(1,3,1,1))
        self.mean.data = torch.FloatTensor(IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(IMAGE_RGB_STD).view(self.std.shape)

    def forward(self, x):
        x = (x-self.mean)/self.std
        return x



# Batch Normalization with Enhanced Linear Transformation
class EnBatchNorm2d(nn.Module):
    def __init__(self, in_channel, k=3, eps=1e-5):
        super(EnBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_channel, eps=1e-5,affine=False)
        self.conv = nn.Conv2d(in_channel, in_channel,
                              kernel_size=k,
                              padding=(k - 1) // 2,
                              groups=in_channel,
                              bias=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x

###############################################################################
class ConvEnBn2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1):
        super(ConvEnBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn   = EnBatchNorm2d(out_channel, eps=1e-5)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvBn2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn   = nn.BatchNorm2d(out_channel, eps=1e-5)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x



# bottleneck type C
class EnBasic(nn.Module):
    def __init__(self, in_channel, out_channel, channel, kernel_size=3, stride=1, is_shortcut=False):
        super(EnBasic, self).__init__()
        self.is_shortcut = is_shortcut

        self.conv_bn1 = ConvEnBn2d(in_channel,    channel, kernel_size=kernel_size, padding=kernel_size//2, stride=stride)
        self.conv_bn2 = ConvEnBn2d(   channel,out_channel, kernel_size=kernel_size, padding=kernel_size//2, stride=1)

        if is_shortcut:
            self.shortcut = ConvBn2d(in_channel, out_channel, kernel_size=1, padding=0, stride=stride)


    def forward(self, x):
        z = F.relu(self.conv_bn1(x),inplace=True)
        z = self.conv_bn2(z)

        if self.is_shortcut:
            x = self.shortcut(x)

        z += x
        z = F.relu(z,inplace=True)
        return z




class EnResNet34(nn.Module):

    def __init__(self, num_class=1000 ):
        super(EnResNet34, self).__init__()

        self.block0  = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block1  = nn.Sequential(
             EnBasic( 64, 64, 64, kernel_size=3, stride=1, is_shortcut=False,),
          * [EnBasic( 64, 64, 64, kernel_size=3, stride=1, is_shortcut=False,) for i in range(1,3)],
        )
        self.block2  = nn.Sequential(
             EnBasic( 64,128,128, kernel_size=3, stride=2, is_shortcut=True, ),
          * [EnBasic(128,128,128, kernel_size=3, stride=1, is_shortcut=False,) for i in range(1,4)],
        )
        self.block3  = nn.Sequential(
             EnBasic(128,256,256, kernel_size=3, stride=2, is_shortcut=True, ),
          * [EnBasic(256,256,256, kernel_size=3, stride=1, is_shortcut=False,) for i in range(1,6)],
        )
        self.block4 = nn.Sequential(
             EnBasic(256,512,512, kernel_size=3, stride=2, is_shortcut=True, ),
          * [EnBasic(512,512,512, kernel_size=3, stride=1, is_shortcut=False,) for i in range(1,3)],
        )
        self.logit = nn.Linear(512,num_class)
        self.rgb = RGB()

    def forward(self, x):
        batch_size = len(x)
        x = self.rgb(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        logit = self.logit(x)
        return logit
