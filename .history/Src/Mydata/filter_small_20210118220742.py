import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from histomicstk.segmentation.label import area_open

def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


#https://www.kaggle.com/bguberfain/memory-aware-rle-encoding
def rle_encode_less_memory(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    This simplified method requires first and last pixel to be zero
    '''
    pixels = img.T.flatten()
    
    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)

def rle_decode(rle, height, width , fill=255):
    s = rle.split()
    start, length = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    start -= 1
    mask = np.zeros(height*width, dtype=np.uint8)
    for i, l in zip(start, length):
        mask[i:i+l] = fill
    mask = mask.reshape(width,height).T
    mask = np.ascontiguousarray(mask)
    return mask


def rle_encode(mask):
    m = mask.T.flatten()
    m = np.concatenate([[0], m, [0]])
    run = np.where(m[1:] != m[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle =  ' '.join(str(r) for r in run)
    return rle

def image_size_dict(img_id, x, y):
    image_id = [thing[:-5] for thing in img_id]
    x_y = [(x[i], y[i]) for i in range(0, len(x))]    
    return dict(zip(image_id, x_y))

def get_Several_MinMax_Array(np_arr, several):
    """
    获取numpy数值中最大或最小的几个数
    :param np_arr:  numpy数组
    :param several: 最大或最小的个数（负数代表求最大，正数代表求最小）
    :return:
        several_min_or_max: 结果数组
    """
    if several > 0:
        several_min_or_max = np_arr[np.argpartition(np_arr,several)[:several]]
    else:
        several_min_or_max = np_arr[np.argpartition(np_arr, several)[several:]]
    return several_min_or_max

# post process ---
# https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv/42812226
def filter_small(mask, min_size):

    m = (mask*255).astype(np.uint8)
    #num_comp
    #comp：图像上每一像素的标记，用数字1、2、3…表示（不同的数字表示不同的连通域）
    #stat：每一个标记的统计信息，是一个5列的矩阵，每一行对应每个连通区域的外接矩形的x、y、width、height和面积，示例如下： 0 0 720 720 291805
    #centroid：连通域的中心点
    num_comp, comp, stat, centroid = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_comp==1: return mask

    # filtered = np.zeros(comp.shape,dtype=np.uint8)
    area = stat[:, -1]
    # print(get_Several_MinMax_Array(area,8))
    for i in range(1, num_comp):
        if area[i] <= min_size:
            m[comp == i] = 0
    return m

if __name__ == '__main__':
    min_size = 0
    local_file = '/data/shaozc/Kaggle-HuBMAP/submit/Ensemble/submission-2021-01-18_09-53-58.csv'
    local_path = Path(local_file)
    dfpred = pd.read_csv(local_file)
    image_id = [thing for thing in dfpred['id']]
    dfinfo = pd.read_csv('/data/shaozc/Kaggle-HuBMAP/HuBMAP-20-dataset_information.csv')
    dfsample = pd.read_csv('/data/shaozc/Kaggle-HuBMAP/sample_submission.csv')
    size_dict = image_size_dict(dfinfo.image_file, dfinfo.width_pixels, dfinfo.height_pixels)  #dict which contains image sizes mapped to id's
    for i, ids in tqdm(enumerate(image_id)):
        TARGET_ID = ids
        mask_shape = size_dict.get(TARGET_ID)
        taridx = dfpred[dfpred['id']==TARGET_ID].index.values[0]  #row of TARGET_ID in dfpred
        maskpred = rle2mask(dfpred.iloc[taridx]['predicted'], mask_shape)
        # maskpred1 = maskpred.copy()
        # maskpred1[maskpred1>0]=1
        #去除小的连通域
        mask_filter = maskpred
        # mask_filter = filter_small(maskpred1, min_size)
        newrle = rle_encode_less_memory(mask_filter)
        dfpred.at[taridx, 'predicted'] = newrle
    mydict = dict(zip(dfpred['id'], dfpred['predicted']))
    dfsample['predicted'] = dfsample['id'].map(mydict).fillna(dfsample['predicted'])
    dfsample = dfsample.replace(np.nan, '', regex=True)
    dfsample.to_csv(local_path.parent/(local_path.stem+'_filter'+local_path.suffix) ,index=False)

    61115381 34 61146675 36 61177970 36 61209264 38 61240545 70 61271839 72 61303134 72 61334428 74 61365713 90 61397007 92 61428302 92 61459596 94 61490889 102 61522183 104 61553478 104 61584772 106 61616061 118 61647355 120 61678650 120 61709944 122 61741237 130 61772531 132 61803826 132 61835120 134 61866409 142 61897703 144 61928998 144 61960292 146 61991585 154 62022879 156 62054174 156 62085468 158 62116761 162 62148055 164 62179350 164 62210644 166 62241937 170 62273231 172 62304526 172 62335820 174 62367113 182 62398407 184 62429702 184 62460996 186 62492289 190 62523583 192 62554878 192 62586173 193 62617468 199 62648763 200 62680058 200 62711352 202 62742645 206 62773939 208 62805234 208 62836528 210 62867821 214 62899115 216 62930410 216 62961705 217 62993000 219 63024295 220 63055590 220 63086884 222 63118177 226 63149471 228 63180766 228 63212061 229 63243356 231 63274651 232 63305946 232