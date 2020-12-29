import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import cv2
import gc
import random
import torch
from albumentations import *
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/shaozc/Project/Kaggle-PANDA/Kaggle-PANDA-Solution/") 
from Src import read_tiff, read_json_as_df
from Propressing import *



def img2tensor(img,dtype:np.dtype=np.float32):
    # if img.ndim==2 : img = np.expand_dims(img,2)
    # img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))

def null_collate(batch):
    batch_size = len(batch)
    index = []
    mask = []
    image = []
    for r in batch.items():
        if r[0] in 'index':
            index.append(r[1].cpu().numpy())
        if r[0] in 'mask':
            mask.append(r[1].cpu().numpy())
        if r[0] in 'image':
            image.append(r[1].cpu().numpy())
    index = index[0]

    image = np.stack(image) #把数组组合一起
    image = image[...,::-1]
    image = image.squeeze(0)
    image = image.transpose(0,3,1,2)
    image = np.ascontiguousarray(image)
    # print("null_collate")

    mask  = np.stack(mask)
    mask = mask.squeeze(0)
    mask  = np.ascontiguousarray(mask)

    #---
    image = torch.from_numpy(image).contiguous().float()
    mask  = torch.from_numpy(mask).contiguous().unsqueeze(1)
    mask  = (mask>0.5).float()

    return {
        'index' : index,
        'mask' : mask,
        'image' : image,
    }

def make_image_id (mode):
    train_image_id = {
        0 : '2f6ecfcdf',
        1 : 'aaa6a05cc',
        2 : 'cb2d976f4',
        3 : '0486052bb',
        4 : 'e79de561c',
        5 : '095bf7a1f',
        6 : '54f2eec69',
        7 : '1e2425f28',
    }
    test_image_id = {
        0 : 'b9a3865fc',
        1 : 'b2dc8411c',
        2 : '26dc41664',
        3 : 'c68fe75ea',
        4 : 'afa5e8098',
    }
    if 'pseudo-all'==mode:
        test_id = [ test_image_id[i] for i in [0,1,2,3,4] ]
        return test_id

    if 'test-all'==mode:
        test_id = [ test_image_id[i] for i in [0,1,2,3,4] ] # list(test_image_id.values()) #
        return test_id


    if 'train-all'==mode:
        train_id = [ train_image_id[i] for i in [0,1,2,3,4,5,6,7] ] # list(test_image_id.values()) #
        return train_id

    if 'valid' in mode or 'train' in mode:
        fold = int(mode[-1])
        valid = [fold,]
        train = list({0,1,2,3,4,5,6,7}-{fold,})
        valid_id = [ train_image_id[i] for i in valid ]
        train_id = [ train_image_id[i] for i in train ]

        if 'valid' in mode: return valid_id
        if 'train' in mode: return train_id

class TrainDataset(Dataset):
    def __init__(self, conf_dataset, phase, transform=None):
        assert phase in {"train", "valid"}
        self.conf_dataset = conf_dataset
        self.phase = phase
        self.transform = transform
        # self.mean = np.array([0.63234259,0.43992372,0.67714315])
        # self.std = np.array([0.37777001,0.46452386,0.34882929])

        #Get from config
        self.nfolds = conf_dataset.nfolds
        self.fold = conf_dataset.fold
        self.seed = conf_dataset.seed 
        self.train_dir = conf_dataset.train_dir #img数据的路径
        self.mask_dir = conf_dataset.mask_dir 
        self.label_dir = conf_dataset.label_dir #csv文件的路径
        self.data_dir = conf_dataset.data_dir #总的一个路径
        self.image_dir = conf_dataset.image_dir #文件夹的名字

        # ids = pd.read_csv(self.label_dir).id.values
        # kf = KFold(n_splits=self.nfolds, random_state=self.seed)
        # ids = set(ids[list(kf.split(ids))[self.fold][0 if phase=="train" else 1]])
        ids = make_image_id('%s-%d'%(self.phase,self.fold))
        ids_list = list(ids)

        tile_id = []
        for id in ids_list: #对于tiff的ids进行遍历
            df = pd.read_csv(self.data_dir + '%s/%s.csv'% (self.image_dir,id))
            tile_id += ('%s/%s/'%(self.image_dir,id) + df.tile_id).tolist()
        self.tile_id = tile_id #所有的tile的集合
        self.len = len(self.tile_id)

        print('\n--- [Dataset] %s\n' % ('-' * 64))
        print('\tphase  = %s'%self.phase)
        print('\tnfolds = %d'%self.nfolds)
        print('\t fold  = %d'%self.fold)
        print('\t  ids  = %s'%str(ids))
        print('\t  len  = %d\n'%self.len)


    def __len__(self):
        return self.len
    

    def __getitem__(self, index):
        id = self.tile_id[index] #存储了一个路径
        image = cv2.imread(self.data_dir + '%s.png'%(id), cv2.IMREAD_COLOR)
        mask  = cv2.imread(self.data_dir + '%s.mask.png'%(id), cv2.IMREAD_GRAYSCALE)
        # image = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA) 
        # mask = cv2.resize(mask, (256,256), interpolation=cv2.INTER_AREA) 

        image = image.astype(np.float32) / 255
        mask  = mask.astype(np.float32) / 255

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        # image = (image/255.0 - self.mean)/self.std #色彩均一化
        r = {
            'index' : index,
            'tile_id' : id,
            'mask' : img2tensor(mask),
            'image' : img2tensor(image),
        }

        return r

class TestDataset(Dataset):
    def __init__(self, conf_dataset, transform=None):
        self.conf_dataset = conf_dataset
        self.label_dir = self.conf_dataset.label_dir
        self.test_dir = self.conf_dataset.test_dir
        self.ids_all = pd.read_csv(self.label_dir).id.values

    def __len__(self):
        return len(self.ids_all)
        # return 1
    
    def __getitem__(self,idx): #返回一个tiff
        ids = self.ids_all[idx]
        image_file = self.test_dir + '/%s.tiff' % ids
        json_file  = self.test_dir + '/%s-anatomical-structure.json' % ids

        image = read_tiff(image_file)
        height, width = image.shape[:2]
        #通过给的json文件填充了一个mask区域，后续利用这个区域来去除背景
        # structure = draw_strcuture(read_json_as_df(json_file), height, width, structure=['Cortex'])
        structure = draw_strcuture_from_hue(image, fill=255, scale=1/32)
        return image, structure, ids

