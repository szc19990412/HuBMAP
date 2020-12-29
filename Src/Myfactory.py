import albumentations as alb
import albumentations.pytorch as albp
import torch
import torch.nn as nn
import torch_optimizer as toptim
import yaml
from addict import Dict
import json

import sys
sys.path.append("/home/shaozc/Project/Kaggle-PANDA/Kaggle-PANDA-Solution/") 
from common import *
from Configs import *
from External import *
from Models import *
from Myloss import *
from Utils import *



"""
Functions for converting config to each object
"""


def get_transform(conf_augmentation):
    # For multi-channel mask
    additional_targets = {
        "mask0": "mask",
        "mask1": "mask",
        "mask2": "mask",
        "mask3": "mask",
        "mask4": "mask",
    }

    def get_object(trans):
        if trans.name in {"Compose", "OneOf"}:
            augs_tmp = [get_object(aug) for aug in trans.member]
            return getattr(alb, trans.name)(augs_tmp, **trans.params)

        if hasattr(alb, trans.name):
            return getattr(alb, trans.name)(**trans.params)
        else:
            return eval(trans.name)(**trans.params)

    if conf_augmentation is None:
        augs = list()
    else:
        augs = [get_object(aug) for aug in conf_augmentation]
    augs.append(albp.ToTensorV2())
    return alb.Compose(augs, additional_targets=additional_targets)


def get_model(cfg_model):
    net = None
    if "seresnext" in cfg_model.base:
        net = CustomSEResNeXt(
            in_ch=cfg_model.in_channel,
            out_ch=cfg_model.out_channel,
            pool_type=cfg_model.pool_type,
            pretrained=cfg_model.pretrained,
        )
    elif "resnest" in cfg_model.base:
        net = CustomResnest(
            base=cfg_model.base,
            in_ch=cfg_model.in_channel,
            out_ch=cfg_model.out_channel,
            pool_type=cfg_model.pool_type,
            pretrained=cfg_model.pretrained,
        )
    elif "resnet" in cfg_model.base:
        net = CustomResNet(
            base="resnet34",
            target_size=cfg_model.out_channel,
            in_ch=cfg_model.in_channel,
            pretrained=cfg_model.pretrained,
        )
    elif "eunet" == cfg_model.base:
        net = EUNet(
            base=cfg_model.encoder,
            cls_out_ch=cfg_model.out_channel,
            seg_out_ch=cfg_model.out_channel_seg,
            pretrained=cfg_model.pretrained,
        )
    elif "eunet-mini" == cfg_model.base:
        net = EUNetMini(
            base=cfg_model.encoder,
            cls_out_ch=cfg_model.out_channel,
            seg_out_ch=cfg_model.out_channel_seg,
            pretrained=cfg_model.pretrained,
            fp16=cfg_model.fp16,
        )
    elif "eunet-mini2" == cfg_model.base:
        net = EUNetMini2(
            base=cfg_model.encoder,
            cls_out_ch=cfg_model.out_channel,
            seg_out_ch=cfg_model.out_channel_seg,
            pretrained=cfg_model.pretrained,
            fp16=cfg_model.fp16,
        )
    elif "eunet-mini3" == cfg_model.base:
        net = EUNetMini3(
            base=cfg_model.encoder,
            cls_out_ch=cfg_model.out_channel,
            seg_out_ch=cfg_model.out_channel_seg,
            pretrained=cfg_model.pretrained,
            fp16=cfg_model.fp16,
        )
    elif "efficientnet" in cfg_model.base:
        net = CustomEfficientNet(
            base=cfg_model.base,
            in_ch=cfg_model.in_channel,
            out_ch=cfg_model.out_channel,
            pretrained=cfg_model.pretrained,
        )
    elif "CustomUneXt50" in cfg_model.base:
        net = CustomUneXt50()
    elif "CustomEnResNet34" in cfg_model.base:
        net = CustomEnResNet34()
    assert net is not None
    return net


def get_loss(conf):
    conf_loss = conf.base_loss
    # assert hasattr(nn, conf_loss.name) or hasattr(loss, conf_loss.name)
    loss = None
    if hasattr(nn, conf_loss.name):
        loss = getattr(nn, conf_loss.name) #获取对象的属性值
    elif hasattr(myloss, conf_loss.name):
        loss = getattr(myloss, conf_loss.name)

    return loss(**conf_loss.params)


def get_optimizer(conf):
    conf_optim = conf.Optimizer
    name = conf_optim.optimizer.name
    if hasattr(torch.optim, name):
        optimizer_cls = getattr(torch.optim, name)
    else:
        optimizer_cls = getattr(toptim, name)

    if hasattr(conf_optim, "lr_scheduler"):
        scheduler_cls = getattr(torch.optim.lr_scheduler, conf_optim.lr_scheduler.name)
    else:
        scheduler_cls = None
    return optimizer_cls, scheduler_cls

def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

def read_tiff(image_file):
    image = tiff.imread(image_file)
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
        image = np.ascontiguousarray(image)
    return image

def read_mask(mask_file):
    mask = np.array(PIL.Image.open(mask_file))
    return mask

def read_json_as_df(json_file):
   with open(json_file) as f:
       j = json.load(f)
   df = pd.json_normalize(j) #json格式变dataframe
   return df

#-------------------Metric-----------------------#
def np_dice_score(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)
    ths = np.arange(0.46,0.54,0.01)
    union = torch.zeros(len(ths))
    overlap = torch.zeros(len(ths))
    for i,th in enumerate(ths):
        pt = (p>th)
        tt = (t>th)
        union[i] = pt.sum() + tt.sum()
        overlap[i] = (pt*tt).sum()

    dice = torch.where(union>0, 2*overlap/(union+0.001), torch.zeros_like(union))
    return dice, ths

def np_accuracy(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)
    p = p>0.5
    t = t>0.5
    tp = (p*t).sum()/(t).sum()
    tn = ((1-p)*(1-t)).sum()/(1-t).sum()
    return tp, tn

# --draw ------------------------------------------
def mask_to_inner_contour(mask):
    mask = mask>0.5
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = mask & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour


def draw_contour_overlay(image, mask, color=(0,0,255), thickness=1):
    contour =  mask_to_inner_contour(mask)
    if thickness==1:
        image[contour] = color
    else:
        r = max(1,thickness//2)
        for y,x in np.stack(np.where(contour)).T:
            cv2.circle(image, (x,y), r, color, lineType=cv2.LINE_4 )
    return image

if __name__ == "__main__":
    d = read_yaml()
    print(type(d))
    print(d.Augmentation)
    print(d["Augmentation"])
    loss = get_loss(d.Loss)
    print(loss)
