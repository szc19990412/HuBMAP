import sys
sys.path.append("./") 
from common import *
from Configs import *
from External import *
from Models import *
from Utils import *
from Src import get_model, TrainDataset, TestDataset, null_collate,\
                read_tiff, np_dice_score, np_accuracy, quadratic_weighted_kappa\
                create_optimizer, EtcTrainDataset, EtcTestDataset, create_loss
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
import tifffile
import segmentation_models_pytorch as smp
from pytorch_toolbelt.inference import tta
import colorcorrect.algorithm as cca

def save_pyramid(img, path):
        tif = tifffile.TiffWriter(path, bigtiff=True)
        options = dict(tile=(256, 256), compress='jpeg')
        # print('start saving')
        tif.save(img, subifds=2, **options)
        tif.save(img[::2, ::2], subfiletype=1, **options)
        tif.save(img[::4, ::4], subfiletype=1, **options)
        tif.save(img[::8, ::8], subfiletype=1, **options)
        tif.save(img[::16, ::16], subfiletype=1, **options)
#-------------------------Loss------------------------#
def symmetric_lovasz(outputs, targets):
    return 0.5*(lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1.0 - targets))

def np_binary_cross_entropy_loss(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)

    #---
    logp = -np.log(np.clip(p,1e-6,1))
    logn = -np.log(np.clip(1-p,1e-6,1))
    loss = t*logp +(1-t)*logn
    loss = loss.mean()
    return loss

def criterion_binary_cross_entropy(logit, mask): #BCEWithLogitsLoss
    logit = logit.reshape(-1)
    mask = mask.reshape(-1)
    # logit = torch.from_numpy(logit).cuda()
    # logit = logit.requires_grad_(True)
    # mask = torch.from_numpy(mask).cuda()

    loss = F.binary_cross_entropy_with_logits(logit, mask)
    # loss = loss.requires_grad_(True)
    return loss
#-------------------------Transform--------------------#
def get_aug(p=1): #增强方案一
    return Compose([
        RandomRotate90(), #随机旋转
        Flip(), #水平翻转或垂直翻转
        Transpose(), #行列转置
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, 
                         border_mode=cv2.BORDER_REFLECT), #仿射变换：线性变换+平移
        OneOf([
            MedianBlur(blur_limit=3, p=0.3), #中心模糊
            Blur(blur_limit=3, p=0.3), #模糊图像
        ], p=0.3),
        OneOf([
            OpticalDistortion(p=0.3), #光学畸变
            IAAPiecewiseAffine(p=0.3), #形态畸变
        ], p=0.3),
        OneOf([
            IAASharpen(), #锐化
            IAAEmboss(), #类似于锐化
            RandomContrast(limit=0.5), #对比度变化
        ], p=0.3),
        OneOf([ #HSV变换
		    HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
        # CLAHE(clip_limit=2),
            RandomBrightnessContrast(), #随机亮度和对比度变化 
         ], p=0.8),
    ], p=p)

# def train_augment(record): #增强方案二
#     image = record['image']
#     mask  = record['mask']
#     image_size = 256
#     for fn in np.random.choice([
#         lambda image, mask : do_random_rotate_crop(image, mask, size=image_size, mag=45),
#         lambda image, mask : do_random_scale_crop(image, mask, size=image_size, mag=0.075),
#         lambda image, mask : do_random_crop(image, mask, size=image_size),
#     ],1): image, mask = fn(image, mask)

#     for fn in np.random.choice([
#         lambda image, mask : (image, mask),
#         lambda image, mask : do_random_contast(image, mask, mag=0.8),
#         lambda image, mask : do_random_gain(image, mask, mag=0.9),
#         #lambda image, mask : do_random_hsv(image, mask, mag=[0.1, 0.2, 0]),
#         lambda image, mask : do_random_noise(image, mask, mag=0.1),
#     ],2): image, mask =  fn(image, mask)

#     image, mask = do_random_hsv(image, mask, mag=[0.1, 0.2, 0])
#     image, mask = do_random_flip_transpose(image, mask)

#     record['mask'] = mask
#     record['image'] = image
#     return record

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        Lambda(image=preprocessing_fn),
        # Lambda(image=to_tensor, mask=to_tensor),
    ]
    return Compose(_transform)

#-------------------------------

class LightningModuleReg(pl.LightningModule):
#-------Init部分-------#
    def __init__(self, cfg):
        super(LightningModuleReg, self).__init__()
        self.cfg = cfg
        self.frozen_bn = cfg['General']['frozen_bn'] #是否反向传播
        self.net = self.get_net()
        self.loss = create_loss(self.cfg['Loss'])
        self.phase = cfg['Model']['phase']
        self.tta = True
        self.test_dir = self.cfg['Data']['testdataset']['test_dir']
        self.submit_dir = self.cfg['Data']['testdataset']['submit_dir']
        self.submit_id = pd.read_csv(self.cfg['Data']['testdataset']['label_dir']).id.values
        self.kfold = self.cfg['Data']['dataset']['fold']
        #----
        self.valid_probability = []
        self.valid_mask = []
        #----
        # ENCODER = 'se_resnext50_32x4d'
        # ENCODER_WEIGHTS = 'imagenet'
        # self.preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

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
        label = batch['label'].cuda()

        logit, x_class = self.net(image)
        #-----分类加分割loss
        # loss = symmetric_lovasz(logit, mask) + criterion_binary_cross_entropy(x_class, label)
        #-----分类loss
        loss = criterion_binary_cross_entropy(x_class, label)
        # loss = self.loss(logit, mask)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        pass




#----------------validation_step-------------#
    def validation_step(self, batch, batch_nb):
        batch = null_collate(batch)
        batch_size = len(batch['index'])
        mask = batch['mask'].cuda()
        image = batch['image'].cuda()
        label = batch['label'].cuda()

        logit, x_class = self.net(image)
        loss = criterion_binary_cross_entropy(x_class, label)

        # Accuracy and Kappa score
        all_preds = torch.cat([x_class for i in x_class]).cpu().numpy()
        all_targets = torch.cat([label for i in label]).cpu().numpy()
        v_kappa = quadratic_weighted_kappa(all_targets, all_preds)
        # acc = (all_preds == all_targets).mean() * 100.0
        # tp, tn = np_accuracy(x_class.data.cpu().numpy(), label.data.cpu().numpy())
        self.log('val-loss', loss, prog_bar=True)
        # self.log('tp', tp, prog_bar=True)
        # self.log('tn', tn, prog_bar=True)

        # self.valid_probability.append(logit.data.cpu().numpy())
        # self.valid_mask.append(mask.data.cpu().numpy())


    # def validation_epoch_end(self, outputs):
    #     probability = np.concatenate(self.valid_probability)
    #     mask = np.concatenate(self.valid_mask)#连起来
    #     # loss = self.loss(probability, mask)
    #     # loss = np_binary_cross_entropy_loss(probability, mask)
    #     # loss = criterion_binary_cross_entropy(x_class, label)
    #     dices, ths = np_dice_score(probability, mask)
    #     best_dice = dices.max()
    #     # best_thr = ths[dices.argmax()]
    #     # self.log('val-loss', loss, prog_bar=True)
    #     self.log('dice', best_dice, prog_bar=True)
    #     self.valid_probability = []
    #     self.valid_mask = []
        

#----------------configure_optimizers----------------#
    def configure_optimizers(self):
        optimizer = create_optimizer(self.cfg['Optimizer'], self.net)
        # lr_scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=3)
        return [optimizer] #, [lr_scheduler]

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
        image, structure, ids = batch #此处取出的类型是tensor
        mask = None
        submit_dir = Path(self.submit_dir) / str(self.kfold)
        submit_dir.mkdir(exist_ok=True)
        submit_dir = str(submit_dir)

        tile_size = self.cfg['Data']['testdataset']['tile_size']
        tile_average_step = self.cfg['Data']['testdataset']['tile_average_step']
        tile_scale = self.cfg['Data']['testdataset']['tile_scale']
        tile_min_score = self.cfg['Data']['testdataset']['tile_min_score']
        #dataloader 取出的是tensor，且第一维度为batch数，cv要求输入格式（HWC）
        image = image.cpu().numpy().squeeze(0)
        structure = structure.cpu().numpy().squeeze(0)

        tile = to_tile(image, mask, structure, tile_scale, tile_size, tile_average_step, tile_min_score)
        tile_image = tile['tile_image'] #切片集
        for i in range(len(tile_image)):
            tile_image[i] = cca.stretch(tile_image[i])
        tile_image = np.stack(tile_image)[..., ::-1] #沿最后一个轴叠加 == np.stack(tile_image)
        tile_image = np.ascontiguousarray(tile_image.transpose(0,3,1,2)) #连续存储，速度更快
        tile_image = tile_image.astype(np.float32)/255

        tile_probability = []
        batch = np.array_split(tile_image, len(tile_image)//4) #切分，64份
        for t,m in enumerate(batch):
            m = torch.from_numpy(m).cuda()
            
            p = []
            logit, _ = self(m)
            p.append(torch.sigmoid(logit))

            #----
            if self.tta: #tta
                logit, _ = self(m.flip(dims=(2,)))
                p.append(torch.sigmoid(logit.flip(dims=(2,))))

                logit, _ = self(m.flip(dims=(3,)))
                p.append(torch.sigmoid(logit.flip(dims=(3,))))
            #----

            p = torch.stack(p).mean(0) #相加求平均
            # p = tta.d4_image2mask(self, m)
            tile_probability.append(p.data.cpu().numpy())

        tile_probability = np.concatenate(tile_probability).squeeze(1)
        height, width = tile['image_small'].shape[:2]
        probability = to_mask(tile_probability, tile['coord'], height, width,
                              tile_scale, tile_size, tile_average_step, tile_min_score,
                              aggregate='mean')

        #------ show results -------
        truth = np.zeros((height, width), np.float32)

        overlay = np.dstack([ #沿深度方向做拼接
            np.zeros_like(truth),
            probability, #green
            truth, #red
        ])
        image_small = tile['image_small'].astype(np.float32)/255
        predict = (probability>0.5).astype(np.float32)
        overlay1 = 1-(1-image_small)*(1-overlay)
        overlay2 = image_small.copy()
        overlay2 = draw_contour_overlay(overlay2, tile['structure_small'], color=(1, 1, 1), thickness=3)
        overlay2 = draw_contour_overlay(overlay2, truth, color=(0, 0, 1), thickness=8)
        overlay2 = draw_contour_overlay(overlay2, probability, color=(0, 1, 0), thickness=3)

        if 1:
            cv2.imwrite(submit_dir+'/%s.image_small.png'%ids, (image_small*255).astype(np.uint8))
            cv2.imwrite(submit_dir+'/%s.probability.png'%ids, (probability*255).astype(np.uint8))
            cv2.imwrite(submit_dir+'/%s.predict.png'%ids, (predict*255).astype(np.uint8))
            overlay = (overlay*255).astype(np.uint8)
            overlay1 = (overlay1*255).astype(np.uint8)
            overlay2 = (overlay2*255).astype(np.uint8)
            save_pyramid(overlay,submit_dir+'/%s.overlay.png'%ids)
            save_pyramid(overlay1,submit_dir+'/%s.overlay1.png'%ids)
            save_pyramid(overlay2,submit_dir+'/%s.overlay2.png'%ids)
            # cv2.imwrite(submit_dir+'/%s.overlay.png'%ids, (overlay*255).astype(np.uint8))
            # cv2.imwrite(submit_dir+'/%s.overlay1.png'%ids, (overlay1*255).astype(np.uint8))
            # cv2.imwrite(submit_dir+'/%s.overlay2.png'%ids, (overlay2*255).astype(np.uint8))

        # # #---

    def test_epoch_end(self, outputs):
        submit_dir = Path(self.submit_dir) / str(self.kfold)
        submit_dir = str(submit_dir)
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
            # preprocessing=get_preprocessing(self.preprocessing_fn),
            preprocessing=None,
        )   

    def get_loader(self, phase):
        assert phase in {"train", "valid"}
        dataset = self.get_ds(phase=phase)

        cfg_dataloader = self.cfg.Data.dataloader
        return DataLoader(
            dataset,
            batch_size=cfg_dataloader.batch_size if phase == "train" else 8,
            shuffle=False if phase == "train" else False,
            sampler = RandomSampler(dataset) if phase == "train" else SequentialSampler(dataset),
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
            num_workers=8,
        )

