# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from fastai.vision.all import *
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models
# from functools import partial
# nonlinearity = partial(F.relu, inplace=True)
# class DACblock(nn.Module):
#     def __init__(self, channel):
#         super(DACblock, self).__init__()
#         self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
#         self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
#         self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
#         self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#     def forward(self, x):
#         dilate1_out = nonlinearity(self.dilate1(x))
#         dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
#         dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
#         dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
#         out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
#         return out
# class SPPblock(nn.Module):
#     def __init__(self, in_channels):
#         super(SPPblock, self).__init__()
#         self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
#         self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
#         self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
#         self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

#         self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

#     def forward(self, x):
#         self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
#         self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=True)
#         self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=True)
#         self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=True)
#         self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=True)

#         out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

#         return out
# class FPN(nn.Module):
#     def __init__(self, input_channels:list, output_channels:list):
#         super().__init__()
#         self.convs = nn.ModuleList(
#             [nn.Sequential(nn.Conv2d(in_ch, out_ch*2, kernel_size=3, padding=1),
#              nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch*2),
#              nn.Conv2d(out_ch*2, out_ch, kernel_size=3, padding=1))
#             for in_ch, out_ch in zip(input_channels, output_channels)])
        
#     def forward(self, xs:list, last_layer):
#         hcs = [F.interpolate(c(x),scale_factor=2**(len(self.convs)-i),mode='bilinear') 
#                for i,(c,x) in enumerate(zip(self.convs, xs))]
#         hcs.append(last_layer)
#         return torch.cat(hcs, dim=1)

# class UnetBlock(Module):
#     def __init__(self, up_in_c:int, x_in_c:int, nf:int=None, blur:bool=False,
#                  self_attention:bool=False, **kwargs):
#         super().__init__()
#         self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, **kwargs)
#         self.bn = nn.BatchNorm2d(x_in_c)
#         ni = up_in_c//2 + x_in_c
#         nf = nf if nf is not None else max(up_in_c//2,32)
#         self.conv1 = ConvLayer(ni, nf, norm_type=None, **kwargs)
#         self.conv2 = ConvLayer(nf, nf, norm_type=None,
#             xtra=SelfAttention(nf) if self_attention else None, **kwargs)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, up_in:Tensor, left_in:Tensor) -> Tensor:
#         s = left_in
#         up_out = self.shuf(up_in)
#         cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
#         return self.conv2(self.conv1(cat_x))

# from torch import nn
# from pytorch_toolbelt.modules import encoders as E
# from pytorch_toolbelt.modules import decoders as D
# import segmentation_models_pytorch as smp
# class CustomUneXt50_C(nn.Module):
#     def __init__(self, stride=1, **kwargs):
#         super().__init__()
#         #encoder
#         m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',
#                            'resnext50_32x4d_ssl')
#         self.enc0 = nn.Sequential(m.conv1, m.bn1, nn.ReLU(inplace=True))
#         self.enc1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),m.layer1) #256
#         self.enc2 = m.layer2 #512
#         self.enc3 = m.layer3 #1024
#         self.enc4 = m.layer4 #2048
#         #aspp with customized dilatations
#         self.dblock = DACblock(2048)
#         self.spp = SPPblock(2048)
#         self.mid_conv = nn.Conv2d(2052, 512, 1, bias=False)
#         #decoder
#         self.dec4 = UnetBlock(512,1024,256)
#         self.dec3 = UnetBlock(256,512,128)
#         self.dec2 = UnetBlock(128,256,64)
#         self.dec1 = UnetBlock(64,64,32)
#         self.fpn = FPN([512,256,128,64],[16]*4)
#         self.drop = nn.Dropout2d(0.1)
#         self.final_conv = ConvLayer(32+16*4, 1, ks=1, norm_type=None, act_cls=None)
        
#     def forward(self, x):
#         enc0 = self.enc0(x)
#         enc1 = self.enc1(enc0)
#         enc2 = self.enc2(enc1)
#         enc3 = self.enc3(enc2)
#         enc4 = self.enc4(enc3)
#         # print("4: ",enc4.shape)
#         enc4 = self.dblock(enc4)
#         enc4 = self.spp(enc4)
#         enc5 = self.mid_conv(enc4)
#         # print("5: ",enc5.shape)
#         dec3 = self.dec4(enc5,enc3)
#         dec2 = self.dec3(dec3,enc2)
#         dec1 = self.dec2(dec2,enc1)
#         dec0 = self.dec1(dec1,enc0)
#         x = self.fpn([enc5, dec3, dec2, dec1], dec0)
#         x = self.final_conv(self.drop(x))
#         x = F.interpolate(x,scale_factor=2,mode='bilinear')
#         return x
# if __name__ == '__main__':
#     data = torch.randn((4,3,512,512)).cuda()
#     net = CustomUneXt50_C().cuda()
#     output = net(data)
#     print(output.shape)

########################################################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        #self.bn_qk = nn.BatchNorm2d(groups)
        #self.bn_qr = nn.BatchNorm2d(groups)
        #self.bn_kr = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AxialAttentionNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.5):
        super(AxialAttentionNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], stride=1, kernel_size=80)
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=80,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=40,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=20,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(1024 * block.expansion * s), num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                if isinstance(m, qkv_transform):
                    pass
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, AxialBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, 
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        print(x.shape)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)


        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def axial26s(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [1, 2, 4, 1], s=0.5, **kwargs)
    return model


def axial50s(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3], s=0.5, **kwargs)
    return model


def axial50m(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3], s=0.75, **kwargs)
    return model


def axial50l(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3], s=1, **kwargs)
    return model
# ax_net = axial50m().cuda()

# total = sum([param.nelement() for param in ax_net.parameters()])
# print("Number of parameter: %.2fM" % (total/1e6))
# data = torch.randn((3,3,320,320)).cuda()
# output = ax_net(data)
# output.shape

from fastai.vision.all import *
class FPN(nn.Module):
    def __init__(self, input_channels:list, output_channels:list):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_ch, out_ch*2, kernel_size=3, padding=1),
             nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch*2),
             nn.Conv2d(out_ch*2, out_ch, kernel_size=3, padding=1))
            for in_ch, out_ch in zip(input_channels, output_channels)])
        
    def forward(self, xs:list, last_layer):
        hcs = [F.interpolate(c(x),scale_factor=2**(len(self.convs)-i),mode='bilinear') 
               for i,(c,x) in enumerate(zip(self.convs, xs))]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)

class UnetBlock(Module):
    def __init__(self, up_in_c:int, x_in_c:int, nf:int=None, blur:bool=False,
                 self_attention:bool=False, **kwargs):
        super().__init__()
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, **kwargs)
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = up_in_c//2 + x_in_c
        nf = nf if nf is not None else max(up_in_c//2,32)
        self.conv1 = ConvLayer(ni, nf, norm_type=None, **kwargs)
        self.conv2 = ConvLayer(nf, nf, norm_type=None,
            xtra=SelfAttention(nf) if self_attention else None, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in:Tensor, left_in:Tensor) -> Tensor:
        s = left_in
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))
        
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                stride=1, padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[6, 12, 18, 24], out_c=None):
        super().__init__()
        self.aspps = [_ASPPModule(inplanes, mid_c, 1, padding=0, dilation=1)] + \
            [_ASPPModule(inplanes, mid_c, 3, padding=d, dilation=d,groups=4) for d in dilations]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                        nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
                        nn.BatchNorm2d(mid_c), nn.ReLU())
        out_c = out_c if out_c is not None else mid_c
        self.out_conv = nn.Sequential(nn.Conv2d(mid_c*(2+len(dilations)), out_c, 1, bias=False),
                                    nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(mid_c*(2+len(dilations)), out_c, 1, bias=False)
        self._init_weight()

    def forward(self, x):
        x0 = self.global_pool(x)
        xs = [aspp(x) for aspp in self.aspps]
        x0 = F.interpolate(x0, size=xs[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class CustomUneXt50_A(nn.Module):
    def __init__(self, stride=1, **kwargs):
        super().__init__()
        #encoder
        m = axial50m()
        self.enc0 = nn.Sequential(m.conv1, m.bn1, nn.ReLU(inplace=True)) #48
        self.enc1 = nn.Sequential(m.maxpool,m.layer1) #192
        
        self.enc2 = m.layer2 #384
        
        self.enc3 = m.layer3 #768
        
        self.enc4 = m.layer4 #1536
        
        #aspp with customized dilatations
        self.aspp = ASPP(1536,256,out_c=384,dilations=[stride*1,stride*2,stride*3,stride*4])
        self.drop_aspp = nn.Dropout2d(0.5)
        #decoder
        self.dec4 = UnetBlock(384,768,192)
        self.dec3 = UnetBlock(192,384,96)
        self.dec2 = UnetBlock(96,192,48)
        self.dec1 = UnetBlock(48,48,32)
        self.fpn = FPN([384,192,96,48],[16]*4)
        self.drop = nn.Dropout2d(0.1)
        self.final_conv = ConvLayer(32+16*4, 1, ks=1, norm_type=None, act_cls=None)
        
    def forward(self, x):
        enc0 = self.enc0(x)
        # print("enc0:",enc0.shape)
        enc1 = self.enc1(enc0)
        # print("enc1:",enc1.shape)
        enc2 = self.enc2(enc1)
        # print("enc2:",enc2.shape)
        enc3 = self.enc3(enc2)
        # print("enc3:",enc3.shape)
        
        enc4 = self.enc4(enc3)
        # print("enc4:",enc4.shape)

        enc5 = self.aspp(enc4)
        # print("enc5:",enc5.shape)

        dec3 = self.dec4(self.drop_aspp(enc5),enc3)
        # print("dec3:",dec3.shape)

        dec2 = self.dec3(dec3,enc2)
        # print("dec2:",dec2.shape)
        dec1 = self.dec2(dec2,enc1)
        # print("dec1:",dec1.shape)
        dec0 = self.dec1(dec1,enc0)
        # print("dec0:",dec0.shape)
        x = self.fpn([enc5, dec3, dec2, dec1], dec0)
        x = self.final_conv(self.drop(x))
        x = F.interpolate(x,scale_factor=2,mode='bilinear')
        return x
if __name__=='__main__':
        net = CustomUneXt50_A().cuda()
        data = torch.randn((3,3,320,320)).cuda()
        total = sum([param.nelement() for param in net.parameters()])
        print("Number of parameter: %.2fM" % (total/1e6))
        output = net(data)