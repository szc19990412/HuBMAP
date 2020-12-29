# import random
# from collections import OrderedDict
# import os
# # os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# import numpy as np
# import pytorch_lightning as pl
# from pytorch_lightning.metrics import Metric
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from warmup_scheduler import GradualWarmupScheduler
# from albumentations import *
# import cv2
# from tqdm import tqdm
# import pandas as pd

# import sys
# sys.path.append('../Utils')
# sys.path.append('../Myloss')
# sys.path.append('./factory')
# sys.path.append('./LightningMetric')
# sys.path.append("./dataset")
# sys.path.append('../Models')
# from dataset import TrainDataset, TestDataset, img2tensor
# from factory import get_model, get_optimizer, get_transform, get_loss
# from Myloss import lovasz_hinge
# from Models import CustomUneXt50
# from Utils import (
#     cutmix_tile,
#     mixup_criterion,
#     mixup_data_same_provider,
#     quadratic_weighted_kappa,
# )
# import sys
# sys.path.append('/home/shaozc/Project/Kaggle-PANDA/Kaggle-PANDA-Solution/Myloss')
# from myloss import dice
import sys
sys.path.append("/home/shaozc/Project/Kaggle-PANDA/Kaggle-PANDA-Solution/") 
from common import *
from Configs import *
from External import *
from Models import *
from Myloss import *
from Propressing import *
from Utils import *
from Src import get_model, get_optimizer, get_transform, get_loss,\
                img2tensor, TrainDataset, TestDataset, null_collate,\
                np_dice_score, np_accuracy, draw_contour_overlay,\
                read_tiff


#--------------------------Metric-------------------------#
from pytorch_lightning.metrics import Metric
class Dice_th_pred(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.add_state("dice", default=torch.tensor(0.0), dist_reduce_fx="mean")  

    def update(self, preds, target):
        smooth = 1e-7
        self.inter = torch.zeros(len(self.ths)) #清零
        self.union = torch.zeros(len(self.ths))
        preds_flat = preds.reshape(-1)
        preds_flat = 1/(1+np.exp(-preds_flat))
        targets_flat = target.reshape(-1)
        p = (preds_flat > 0.5)
        self.inter[i] += (p*targets_flat).sum().item() #.item()把张量变成一个浮点数
        self.union[i] += (p+targets_flat).sum().item()

    def compute(self):
        dice = 2.0*self.inter/(self.union+smooth)
        return dice


#-------------------------Loss------------------------#
def symmetric_lovasz(outputs, targets):
    return 0.5*(lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1.0 - targets))
#-------------------------Transform--------------------#
# def get_aug(p=1):
#     return Compose([
#         RandomRotate90(), #随机旋转
#         Flip(), #水平翻转或垂直翻转
#         Transpose(), #行列转置
#         ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, 
#                          border_mode=cv2.BORDER_REFLECT), #仿射变换：线性变换+平移
#         OneOf([
#             MedianBlur(blur_limit=3, p=0.3), #中心模糊
#             Blur(blur_limit=3, p=0.3), #模糊图像
#         ], p=0.3),
#         OneOf([
#             OpticalDistortion(p=0.3), #光学畸变
#             IAAPiecewiseAffine(p=0.3), #形态畸变
#         ], p=0.3),
#         OneOf([
#             IAASharpen(), #锐化
#             IAAEmboss(), #类似于锐化
#             RandomContrast(limit=0.5), #对比度变化
#         ], p=0.3),
#         OneOf([ #HSV变换
# 		    HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
#         # CLAHE(clip_limit=2),
#             RandomBrightnessContrast(), #随机亮度和对比度变化 
#          ], p=0.8),
#     ], p=p)
def get_aug(p=1.0):
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, 
                         border_mode=cv2.BORDER_REFLECT),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
        OneOf([
            HueSaturationValue(10,15,10),
            # CLAHE(clip_limit=2),
            RandomBrightnessContrast(),            
        ], p=0.3),
    ], p=p)


#-------------------------------

class LightningModuleReg(pl.LightningModule):
#-------Init部分-------#
    def __init__(self, cfg):
        super(LightningModuleReg, self).__init__()
        self.cfg = cfg
        self.frozen_bn = cfg['General']['frozen_bn'] #是否反向传播
        self.net = self.get_net()
        # self.net = AmpNet()
        self.phase = cfg['Model']['phase']
        self.output = OrderedDict()
        self.dice = Dice_th_pred() #Metric
        self.tta = True
        self.test_dir = self.cfg['Data']['testdataset']['test_dir']
        self.submit_dir = self.cfg['Data']['testdataset']['submit_dir']
        self.submit_id = pd.read_csv(self.cfg['Data']['testdataset']['label_dir']).id.values
        self.kfold = self.cfg['Data']['dataset']['fold']
        #----
        self.valid_probability = []
        self.valid_mask = []

    def get_net(self):
        return get_model(self.cfg['Model'])
    
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
#-----------------forward---------------#
    def forward(self, x):
        return self.net(x)

#-----------------training_step----------#

    def training_step(self, batch, batch_nb):
        batch = null_collate(batch)
        # batch_size = len(batch['index'])
        mask = batch['mask'].cuda()
        image = batch['image'].cuda()

        logit = self.net(image)
        # loss = criterion_binary_cross_entropy(logit, mask)
        loss = symmetric_lovasz(logit, mask)

        # train_probability.append(logit.cpu().detach().numpy())
        # train_mask.append(mask.cpu().detach().numpy())

        # probability = np.concatenate(train_probability)
        # mask = np.concatenate(train_mask)#连起来变成numpy
        # loss = criterion_binary_cross_entropy(probability, mask)
        # dice = np_dice_score(probability, mask)
        # # tp, tn = np_accuracy(probability, mask)
        # self.log('train-loss', loss, prog_bar=True)
        # self.log('dice', dice)
        # self.log('tp', tp)
        # self.log('tn', tn)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        pass




#----------------validation_step-------------#
    def validation_step(self, batch, batch_nb):
        batch = null_collate(batch)
        batch_size = len(batch['index'])
        mask = batch['mask'].cuda()
        image = batch['image'].cuda()

        logit = self.net(image)
        # probability = torch.sigmoid(logit)

        self.valid_probability.append(logit.data.cpu().numpy())
        self.valid_mask.append(mask.data.cpu().numpy())

        # tp, tn = np_accuracy(probability, mask)
        # self.log('tp', tp, prog_bar=True)
        # self.log('tn', tn, prog_bar=True)

        # output = OrderedDict({"pred": self.valid_probability})
        # output["mask"] = self.valid_mask


    def validation_epoch_end(self, outputs):
        probability = np.concatenate(self.valid_probability)
        mask = np.concatenate(self.valid_mask)#连起来

        loss = np_binary_cross_entropy_loss(probability, mask)
        dices, ths = np_dice_score(probability, mask)
        best_dice = dices.max()
        best_thr = ths[dices.argmax()]
        self.log('val-loss', loss, prog_bar=True)
        self.log('thr', best_thr, prog_bar=True)
        self.log('dice', best_dice, prog_bar=True)
        self.valid_probability = []
        self.valid_mask = []
        

#----------------configure_optimizers----------------#
    def configure_optimizers(self):
        optimizer_cls, scheduler_cls = get_optimizer(self.cfg)

        conf_optim = self.cfg.Optimizer
        # optimizer = optimizer_cls(self.parameters(), **conf_optim.optimizer.params)
        optimizer = Lookahead(RAdam(filter(lambda p: p.requires_grad, self.parameters()),lr=1e-3))
        if scheduler_cls is None:
            return [optimizer]
        else:
            scheduler_default = scheduler_cls(
                optimizer, **conf_optim.lr_scheduler.params
            )
            scheduler = GradualWarmupScheduler( #学习率的warmup操作
                optimizer,
                multiplier=10,
                total_epoch=1, #target learning rate 第几步达到
                after_scheduler=scheduler_default,
            )
        return [optimizer], [scheduler]

#-------------------test_step-----------------------#
    def mask_to_csv(self, image_id, submit_dir):
        predicted = []
        for id in tqdm(image_id): #此处比较慢，需要加tqdm
            image_file = '%s/%s.tiff' % (self.test_dir,id)
            image = read_tiff(image_file)

            height, width = image.shape[:2]
            predict_file = submit_dir + '/%s.predict.png' % id
            # predict = cv2.imread(predict_file, cv2.IMREAD_GRAYSCALE)
            predict = np.array(PIL.Image.open(predict_file))
            predict = cv2.resize(predict, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
            predict = (predict > 128).astype(np.uint8) * 255 

            p = rle_encode(predict)
            predicted.append(p)

        df = pd.DataFrame()
        df['id'] = image_id
        df['predicted'] = predicted
        return df

    def test_step(self, batch, batch_nb):
        pass
        # image, structure, ids = batch #此处取出的类型是tensor
        # mask = None
        # submit_dir = Path(self.submit_dir) / str(self.kfold)
        # submit_dir.mkdir(exist_ok=True)
        # submit_dir = str(submit_dir)

        # tile_size = 512
        # tile_average_step = 256
        # tile_scale = 0.25
        # tile_min_score = 0.25
        # #dataloader 取出的是tensor，且第一维度为batch数，cv要求输入格式（HWC）
        # image = image.cpu().numpy().squeeze(0)
        # structure = structure.cpu().numpy().squeeze(0)

        # tile = to_tile(image, mask, structure, tile_scale, tile_size, tile_average_step, tile_min_score)
        # tile_image = tile['tile_image'] #切片集
        # tile_image = np.stack(tile_image)[..., ::-1] #沿最后一个轴叠加 == np.stack(tile_image)
        # tile_image = np.ascontiguousarray(tile_image.transpose(0,3,1,2)) #连续存储，速度更快
        # tile_image = tile_image.astype(np.float32)/255

        # tile_probability = []
        # batch = np.array_split(tile_image, len(tile_image)//4) #切分，64份
        # for t,m in enumerate(batch):
        #     m = torch.from_numpy(m).cuda()
            
        #     p = []
        #     logit = self(m)
        #     p.append(torch.sigmoid(logit))

        #     #----
        #     if self.tta: #tta
        #         logit = self(m.flip(dims=(2,)))
        #         p.append(torch.sigmoid(logit.flip(dims=(2,))))

        #         logit = self(m.flip(dims=(3,)))
        #         p.append(torch.sigmoid(logit.flip(dims=(3,))))
        #     #----

        #     p = torch.stack(p).mean(0) #相加求平均
        #     tile_probability.append(p.data.cpu().numpy())

        # tile_probability = np.concatenate(tile_probability).squeeze(1)
        # height, width = tile['image_small'].shape[:2]
        # probability = to_mask(tile_probability, tile['coord'], height, width,
        #                       tile_scale, tile_size, tile_average_step, tile_min_score,
        #                       aggregate='mean')

        # #------ show results -------
        # truth = np.zeros((height, width), np.float32)

        # overlay = np.dstack([ #沿深度方向做拼接
        #     np.zeros_like(truth),
        #     probability, #green
        #     truth, #red
        # ])
        # image_small = tile['image_small'].astype(np.float32)/255
        # predict = (probability>0.5).astype(np.float32)
        # overlay1 = 1-(1-image_small)*(1-overlay)
        # overlay2 = image_small.copy()
        # overlay2 = draw_contour_overlay(overlay2, tile['structure_small'], color=(1, 1, 1), thickness=3)
        # overlay2 = draw_contour_overlay(overlay2, truth, color=(0, 0, 1), thickness=8)
        # overlay2 = draw_contour_overlay(overlay2, probability, color=(0, 1, 0), thickness=3)

        # if 1:
        #     cv2.imwrite(submit_dir+'/%s.image_small.png'%ids, (image_small*255).astype(np.uint8))
        #     cv2.imwrite(submit_dir+'/%s.probability.png'%ids, (probability*255).astype(np.uint8))
        #     cv2.imwrite(submit_dir+'/%s.predict.png'%ids, (predict*255).astype(np.uint8))
        #     cv2.imwrite(submit_dir+'/%s.overlay.png'%ids, (overlay*255).astype(np.uint8))
        #     cv2.imwrite(submit_dir+'/%s.overlay1.png'%ids, (overlay1*255).astype(np.uint8))
        #     cv2.imwrite(submit_dir+'/%s.overlay2.png'%ids, (overlay2*255).astype(np.uint8))

        # # #---

    def test_epoch_end(self, outputs):
        submit_dir = Path(self.submit_dir) / str(self.kfold)
        submit_dir = str(submit_dir)
        self.submit_id = ['afa5e8098']
        csv_file = submit_dir +'/submission-%s-%s.csv'%(str(self.kfold),IDENTIFIER)
        df = self.mask_to_csv(list(self.submit_id), submit_dir)
        df.to_csv(csv_file, index=False)
        print(df)


                    





#--------------------DataLoader-------------------------#
    def get_ds(self, phase):
        assert phase in {"train", "valid"} 
        return TrainDataset(
            conf_dataset=self.cfg.Data.dataset,
            phase=phase,
            transform=get_aug() if phase == "train" else None,
        )   

    def get_loader(self, phase):
        assert phase in {"train", "valid"}
        dataset = self.get_ds(phase=phase)

        cfg_dataloader = self.cfg.Data.dataloader
        return DataLoader(
            dataset,
            batch_size=cfg_dataloader.batch_size if phase == "train" else 8,
            shuffle=False if phase == "train" else False,
            num_workers=cfg_dataloader.num_workers,
            drop_last=True if phase == "train" else False,
            pin_memory  = True,
        )

    def train_dataloader(self):
        return self.get_loader(phase="train")
  
    def val_dataloader(self):
        return self.get_loader(phase="valid")

    def test_dataloader(self):
        dataset = TestDataset(self.cfg.Data.testdataset) 
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8
        )

