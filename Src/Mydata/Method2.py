__author__ = 'shaozc'
#先切片完，然后根据像素点以及饱和度筛选

import os
import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import skimage
import skimage.io
# import tifffile as tiff
# img = tiff.imread(os.path.join(DATA,index+'.tiff'))
from tqdm import tqdm


def get_tiles(img, tile_size, n_tiles, mask):
    t_sz = tile_size
    h, w, c = img.shape
    pad_h = (t_sz - h % t_sz) % t_sz #做一个padding
    pad_w = (t_sz - w % t_sz) % t_sz 

    img = np.pad(img,[[pad_h//2,pad_h-pad_h//2],[pad_w//2,pad_w-pad_w//2],[0,0]],
                constant_values=0)
    mask = np.pad(mask,[[pad_h//2,pad_h-pad_h//2],[pad_w//2,pad_w-pad_w//2]],
                constant_values=0)
    #split image and mask into tiles using the reshape+transpose trick
    # img = cv2.resize(img,(img.shape[1]//reduce,img.shape[0]//reduce),
    #                     interpolation = cv2.INTER_AREA)
    img = img.reshape(img.shape[0]//t_sz,t_sz,img.shape[1]//t_sz,t_sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,t_sz,t_sz,3)

    # mask = cv2.resize(mask,(mask.shape[1]//reduce,mask.shape[0]//reduce),
    #                     interpolation = cv2.INTER_NEAREST)
    mask = mask.reshape(mask.shape[0]//t_sz,t_sz,mask.shape[1]//t_sz,t_sz)
    mask = mask.transpose(0,2,1,3).reshape(-1,t_sz,t_sz)
    
    return img, mask


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--tile-num", type=int, default=64)
    arg("--tile-size", type=int, default=1024)
    arg("--res-level", type=int, default=0, help="0:High, 1:Middle, 2:Low")
    arg("--resize", type=int, default=None)
    arg("--saturation", type=int, default=40)
    arg("--pixel", type=int, default=200)
    return parser.parse_args()

#解析train.csv
def enc2mask(encs, shape):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for m,enc in enumerate(encs):
        if isinstance(enc,np.float) and np.isnan(enc):continue
        s = enc.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1 + m
    return img.reshape(shape).T

def mask2enc(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1,n+1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0: encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs

#--------------------------------
def main():
    #Config
    args = make_parse()
    n_tiles = args.tile_num
    tile_size = args.tile_size
    res_level = args.res_level
    resize_size = args.resize
    s_th = args.saturation
    p_th = 200*tile_size//256
    #-----------------
    root = Path("/newdata/shaozc/Kaggle-HuBMAP/")
    img_dir = root / "train"
    mask_dir = root / "train.csv"
    df_masks = pd.read_csv(mask_dir).set_index('id')
    out_dir = ( #输出文件夹的命名
        root / f"numtile-{n_tiles}-tilesize-{tile_size}-res-{res_level}"
    )
    out_dir.mkdir(exist_ok=True)
    out_train_zip = str(out_dir / "train.zip")
    out_mask_zip = str(out_dir / "mask.zip")
    
    print("Tiling slides into super-patches...")
    x_tot, x2_tot = [], []
    with zipfile.ZipFile(out_train_zip, "w") as img_out, zipfile.ZipFile( #创建一个压缩文件
        out_mask_zip, "w"
    ) as mask_out:
        for index, encs in tqdm(df_masks.iterrows(),total=len(df_masks)):
            img_path = str(img_dir / (index + ".tiff"))
            image = skimage.io.MultiImage(img_path)[res_level]
            if image.shape[-1] != 3:image = np.transpose(image.squeeze(), (1,2,0)) #假如不按照RGB排列

            mask = enc2mask(encs,(image.shape[1],image.shape[0]))

            tiles, masks = get_tiles(
                img=image, tile_size=tile_size, n_tiles=n_tiles, mask=mask
            )

            if resize_size is not None:
                tiles = [cv2.resize(t, (resize_size, resize_size)) for t in tiles]
                masks = [cv2.resize(m, (resize_size, resize_size)) for m in masks]

            for i, (img,m) in enumerate(zip(tiles,masks)):
                #remove black or gray images based on saturation check

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)

                if (s>s_th).sum() <= p_th or img.sum() <= p_th: continue

                # RGB
                x_tot.append((img / 255.0).reshape(-1, 3).mean(0))
                x2_tot.append(((img / 255.0) ** 2).reshape(-1, 3).mean(0))


                # if read with PIL RGB turns into BGR
                # We get CRC error when unzip if not cv2.imencode 图像压缩一波
                img = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
                img_out.writestr(f"{index}_{i}.png", img)

                m = cv2.imencode(".png", m)[1]
                mask_out.writestr(f"{index}_{i}.png", m)

    # image stats
    img_avr = np.array(x_tot).mean(0)
    img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr ** 2)
    print("mean:", img_avr, ", std:", np.sqrt(img_std))
            

    
if __name__ == '__main__':
    main()