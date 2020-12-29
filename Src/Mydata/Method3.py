__author__ = 'shaozc'
#根据提供的json文件，确定mask的区域，后做切片

import sys
sys.path.append("/home/shaozc/Project/Kaggle-PANDA/Kaggle-PANDA-Solution/") 
from common import *
from Configs import *
from External import *
from Models import *
from Myloss import *
from Propressing import *
from Utils import *
from Src import read_tiff

def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--tile-scale", type=int, default=0.25)
    arg("--tile-min-score", type=int, default=0.25)
    arg("--tile-size", type=int, default=1024)
    arg("--tile-average-step", type=int, default=512)
    arg("--data-dir", type=str, default='/newdata/shaozc/Kaggle-HuBMAP')
    arg("--train-tile-dir", type=str, default=f'/newdata/shaozc/Kaggle-HuBMAP/0.25_1024_512_train_corrected')
    return parser.parse_args()

'''
'/%s-anatomical-structure.csv'%id

df.columns
Index(['type', 'id', 'geometry.type', 'geometry.coordinates',
       'properties.classification.name', 'properties.classification.colorRGB',
       'properties.isLocked', 'properties.measurements'],
      dtype='object')


Cortex
Outer Stripe
Inner medulla
Outer Medulla

'''
def draw_strcuture(df, height, width, fill=255, structure=[]):
    mask = np.zeros((height, width), np.uint8)
    for row in df.values:
        type  = row[2]  #geometry.type
        coord = row[3]  # geometry.coordinates
        name  = row[4]   # properties.classification.name

        if structure !=[]:
            if not any(s in name for s in structure): continue


        if type=='Polygon':
            pt = np.array(coord).astype(np.int32)
            #cv2.polylines(mask, [coord.reshape((-1, 1, 2))], True, 255, 1)
            cv2.fillPoly(mask, [pt.reshape((-1, 1, 2))], fill) #填充多边形

        if type=='MultiPolygon':
            for pt in coord:
                pt = np.array(pt).astype(np.int32)
                cv2.fillPoly(mask, [pt.reshape((-1, 1, 2))], fill)

    return mask

def draw_strcuture_from_hue(image, fill=255, scale=1/32):

    height, width, _ = image.shape
    vv = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    vv = cv2.cvtColor(vv, cv2.COLOR_RGB2HSV)
    # image_show('v[0]', v[:,:,0])
    # image_show('v[1]', v[:,:,1])
    # image_show('v[2]', v[:,:,2])
    # cv2.waitKey(0)
    mask = (vv[:, :, 1] > 32).astype(np.uint8) #相当于做一个过滤，把小于32的去除
    mask = mask*fill
    mask = cv2.resize(mask, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

    return mask

# --- rle ---------------------------------
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


# --- tile ---------------------------------
def to_tile(image, mask, structure, scale, size, step, min_score):

    half = size//2 #image类型numpy
    image_small = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    height, width, _ = image_small.shape

    #make score
    structure_small = cv2.resize(structure, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    vv = structure_small.astype(np.float32)/255


    #make coord
    xx = np.linspace(half, width  - half, int(np.ceil((width  - size) / step)))
    yy = np.linspace(half, height - half, int(np.ceil((height - size) / step)))
    xx = [int(x) for x in xx]
    yy = [int(y) for y in yy]

    coord  = []
    reject = []
    for cy in yy:
        for cx in xx:
            cv = vv[cy - half:cy + half, cx - half:cx + half].mean()
            if cv>min_score:
                coord.append([cx,cy,cv]) #去除背景元素
            else:
                reject.append([cx,cy,cv])
    #-----
    if 1:
        tile_image = []
        for cx,cy,cv in coord:
            t = image_small[cy - half:cy + half, cx - half:cx + half]
            assert (t.shape == (size, size, 3))
            tile_image.append(t)

    if mask is not None:
        mask_small = cv2.resize(mask, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        tile_mask = []
        for cx,cy,cv in coord:
            t = mask_small[cy - half:cy + half, cx - half:cx + half]
            assert (t.shape == (size, size))
            tile_mask.append(t)
    else:
        mask_small = None
        tile_mask  = None

    return {
        'image_small': image_small,
        'mask_small' : mask_small,
        'structure_small' : structure_small,
        'tile_image' : tile_image,
        'tile_mask'  : tile_mask,
        'coord'  : coord,
        'reject' : reject,
    }


def to_mask(tile, coord, height, width, scale, size, step, min_score, aggregate='mean'):

    half = size//2
    mask  = np.zeros((height, width), np.float32)

    if 'mean' in aggregate:
        w = np.ones((size,size), np.float32)

        #if 'sq' in aggregate:
        if 1:
            #https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
            y,x = np.mgrid[-half:half,-half:half] #生成2D
            y = half-abs(y)
            x = half-abs(x)
            w = np.minimum(x,y) #在对应位置取较小值，高斯滤波
            w = w/w.max()#*2.5
            w = np.minimum(w,1)

        #--------------
        count = np.zeros((height, width), np.float32) #利用坐标去做拼接
        for t, (cx, cy, cv) in enumerate(coord):
            mask [cy - half:cy + half, cx - half:cx + half] += tile[t]*w
            count[cy - half:cy + half, cx - half:cx + half] += w
               # see unet paper for "Overlap-tile strategy for seamless segmentation of arbitrary large images"
        m = (count != 0)
        mask[m] /= count[m]

    if aggregate=='max':
        for t, (cx, cy, cv) in enumerate(coord):
            mask[cy - half:cy + half, cx - half:cx + half] = np.maximum(
                mask[cy - half:cy + half, cx - half:cx + half], tile[t] )

    return mask


def run_make_train_tile():
    df_train = pd.read_csv(data_dir + '/train.csv')
    print('-'*20 + 'df_train' + '-'*60)
    print(df_train)
    print('-'*20 + 'df_train.shape' + '-'*60)
    print('\ndf_train.shape = '+str(df_train.shape)+'\n')
    print('-'*80)
    os.makedirs(train_tile_dir, exist_ok=True)
    print("\nTiling slides into super-patches...")
    x_tot, x2_tot = [], []
    for i in tqdm(range(0,len(df_train))):
        id, encoding = df_train.iloc[i]

        image_file = data_dir + '/train/%s.tiff' % id
        image = read_tiff(image_file)
        height, width = image.shape[:2]
        mask = rle_decode(encoding, height, width, 255)

        #根据hue值做一次清洗
        structure = draw_strcuture_from_hue(image, fill=255, scale=tile_scale/32)
        #make tile
        tile = to_tile(image, mask, structure, tile_scale, tile_size, tile_average_step, tile_min_score)

        coord = np.array(tile['coord'])
        df_image = pd.DataFrame()
        df_image['cx']=coord[:,0].astype(np.int32)
        df_image['cy']=coord[:,1].astype(np.int32)
        df_image['cv']=coord[:,2]

        # --- save ---
        os.makedirs(train_tile_dir+'/%s'%id, exist_ok=True)

        tile_id =[]
        num = len(tile['tile_image'])
        for t in range(num):
            cx,cy,cv   = tile['coord'][t]
            s = 'y%08d_x%08d' % (cy, cx)
            tile_id.append(s)

            tile_image = tile['tile_image'][t]
            tile_mask  = tile['tile_mask'][t]
            cv2.imwrite(train_tile_dir + '/%s/%s.png' % (id, s), tile_image)
            cv2.imwrite(train_tile_dir + '/%s/%s.mask.png' % (id, s), tile_mask)

            # image_show('tile_image', tile_image)
            # image_show('tile_mask', tile_mask)
            # cv2.waitKey(1)
            # RGB
            x_tot.append((tile_image / 255.0).reshape(-1, 3).mean(0))
            x2_tot.append(((tile_image / 255.0) ** 2).reshape(-1, 3).mean(0))


        df_image['tile_id']=tile_id
        df_image[['tile_id','cx','cy','cv']].to_csv(train_tile_dir+'/%s.csv'%id, index=False)

    # image stats
    img_avr = np.array(x_tot).mean(0)
    img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr ** 2)
    print("mean:", img_avr, ", std:", np.sqrt(img_std))

if __name__ == "__main__":
    #Config
    args = make_parse()
    tile_scale = args.tile_scale
    tile_min_score = args.tile_min_score
    tile_size = args.tile_size
    tile_average_step = args.tile_average_step
    data_dir = args.data_dir
    train_tile_dir = (data_dir) + '/' + f'{tile_min_score}_{tile_size}_{tile_average_step}_train_corrected'
    print(train_tile_dir)
#----------------------------------------#
    run_make_train_tile()