
__author__ = 'shaozc'

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

#--------------------------------------------#

import numpy as np
import pandas as pd

import cv2
import skimage
import skimage.io
import tifffile as tiff
import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import sys #路径
from timeit import default_timer as timer
import random #随机
from pathlib import Path #路径
import argparse #自定参数
import zipfile #压缩文件
from tqdm import tqdm #进度条
import shutil #文件拷贝和删除
from collections import defaultdict, OrderedDict#字典
from datetime import datetime #时间模块
IDENTIFIER = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
import ssl #下载模块
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from warmup_scheduler import GradualWarmupScheduler #优化器
from albumentations import * #transform

import pytorch_lightning as pl
from pytorch_lightning.metrics import Metric
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

#--------------------------------------------#



