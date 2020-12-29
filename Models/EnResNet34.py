import sys
sys.path.append("/home/shaozc/Project/Kaggle-PANDA/Kaggle-PANDA-Solution/") 
from common import *
from Configs import *
from External import *
from Models import *
from Myloss import *
from Utils import *

#unet ################################################################

def resize_like(x, reference, mode='nearest'):
    if x.shape[2:] !=  reference.shape[2:]:
        if mode=='bilinear': #上下采样函数
            x = F.interpolate(x, size=reference.shape[2:],mode='bilinear',align_corners=False)
        if mode=='nearest':
            x = F.interpolate(x, size=reference.shape[2:],mode='nearest')
    return x

class SqueezeExcite(nn.Module): #相当于一个注意力模块
    def __init__(self, in_channel, reduction=16):
        super().__init__()

        #channel squeeze excite
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // reduction, in_channel, 1),
            nn.Sigmoid(),
        )

        #spatial squeeze excite
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channel, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.channel(x) + x * self.spatial(x)
        return x


class ResDecode(nn.Module): #decode
    def __init__( self, in_channel, out_channel ):
        super().__init__()
        self.attent1 = SqueezeExcite(in_channel)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1,bias=False),
            EnBatchNorm2d(out_channel), #nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attent2 = SqueezeExcite(out_channel)

    def forward(self, x):

        x = torch.cat(x, 1) #横着拼接
        x = self.attent1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attent2(x)
        return x


class CustomEnResNet34(nn.Module):
    def load_pretrain(self, skip=['logit.',], is_print=True):
        load_pretrain(self, skip, pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=is_print)

    def __init__( self, ):
        super(CustomEnResNet34, self).__init__()
        e = EnResNet34()
        # self.rgb = RGB()

        self.block0 = e.block0
        self.block1 = e.block1
        self.block2 = e.block2
        self.block3 = e.block3
        self.block4 = e.block4
        e = None  #dropped

        #---
        # self.center = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(256, 256, kernel_size=3, padding=4, dilation=4, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )

        self.center = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.decode1 = ResDecode(256+512, 256)
        self.decode2 = ResDecode(128+256, 128)
        self.decode3 = ResDecode( 64+128,  64)
        self.decode4 = ResDecode( 64+ 64,  32)
        self.decode5 = ResDecode(     32,  16)
        #---

        self.logit = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, image):
        batch_size,C,H,W = image.shape
        #x = self.rgb(image)
        x = image

        x0 = self.block0(x)         #;print('block0',x0.shape)
        x1 = self.block1(x0)        #;print('block1',x1.shape)
        x2 = self.block2(x1)        #;print('block2',x2.shape)
        x3 = self.block3(x2)        #;print('block3',x3.shape)
        x4 = self.block4(x3)        #;print('block4',x4.shape)

        skip = [x0,x1,x2,x3]

        #----
        z = self.center(x4)
        z = self.decode1([skip[-1], resize_like(z, skip[-1])])  #; print('d1',x.size())
        z = self.decode2([skip[-2], resize_like(z, skip[-2])])  #; print('d2',x.size())
        z = self.decode3([skip[-3], resize_like(z, skip[-3])])  #; print('d3',x.size())
        z = self.decode4([skip[-4], resize_like(z, skip[-4])])  #; print('d4',x.size())
        z = self.decode5([resize_like(z, x)])

        logit = self.logit(z)
        return logit