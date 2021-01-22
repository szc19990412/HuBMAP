import albumentations as alb
import albumentations.pytorch as albp
import torch
import torch.nn as nn
import torch_optimizer as toptim
import yaml
from addict import Dict
import json

import sys
sys.path.append("./") 
from common import *
from Configs import *
from External import *
from Models import *
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
    if "CustomUneXt50" in cfg_model.base:
        net = CustomUneXt50()
    elif "CE_Net_binary" in cfg_model.base:
        net = CE_Net_binary()
    elif "A_UneXt50" in cfg_model.base:
        net = CustomUneXt50_A()
    elif "EnResNet34" in cfg_model.base:
        net = CustomEnResNet34()
    elif "DoubleUnet" in cfg_model.base:
        base_model = models.vgg19_bn(pretrained=True)
        net = DoubleUnet(base_model)
    assert net is not None
    return net


def get_loss(conf):
    conf_loss = conf.base_loss
    # assert hasattr(nn, conf_loss.name) or hasattr(loss, conf_loss.name)
    loss = None
    if hasattr(nn, conf_loss.name): #判断对象是否包含对应的属性
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

def read_tiff(image_file:str)->np.ndarray:
    """read tiff

    Args:
        image_file (str): [description]

    Returns:
        ndarray: return HWC ndarray image
    """
    image = tiff.imread(image_file)
    if image.shape[:2] == (1, 1):
        image = image.squeeze(0).squeeze(0)
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


if __name__ == "__main__":
    d = read_yaml()
    print(type(d))
    print(d.Augmentation)
    print(d["Augmentation"])
    loss = get_loss(d.Loss)
    print(loss)
