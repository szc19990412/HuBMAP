import cv2
import os, glob
import tqdm
import pandas as pd
import numpy as np
from skimage import data, io, transform
from skimage.color import rgb2gray, rgb2hsv
from skimage.util import img_as_ubyte, view_as_windows
from skimage import img_as_ubyte
def thresSaturation(img, t=15):
    # typical t = 15
    img = rgb2hsv(img)
    h, w, c = img.shape
    sat_img = img[:, :, 1]
    sat_img = img_as_ubyte(sat_img)
    ave_sat = np.sum(sat_img) / (h * w)
    return ave_sat >= t

if __name__ == "__main__":
    data_dir = '/data/shaozc/Kaggle-HuBMAP/etc-data/images_1024/'
    img_name = glob.glob(data_dir+'*.png') 
    img_save = []
    for i in range(len(img_name)):
        image = cv2.imread('%s'%(img_name[i]), cv2.IMREAD_COLOR)
        if thresSaturation(image):
            img_save.append(img_name[i])
        print('Finish '+ img_name[i])
    df = pd.DataFrame(img_save)
    df.to_csv(data_dir+'process.csv', index=False)


