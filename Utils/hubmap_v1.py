import sys
sys.path.append("./")
from common import *

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


## augmentation ######################################################################
#flip
def do_random_flip_transpose(image, mask):
    if np.random.rand()>0.5:
        image = cv2.flip(image,0)
        mask = cv2.flip(mask,0)
    if np.random.rand()>0.5:
        image = cv2.flip(image,1)
        mask = cv2.flip(mask,1)
    if np.random.rand()>0.5:
        image = image.transpose(1,0,2)
        mask = mask.transpose(1,0)

    image = np.ascontiguousarray(image)
    mask = np.ascontiguousarray(mask)
    return image, mask

#geometric
def do_random_crop(image, mask, size):
    height, width = image.shape[:2]
    x = np.random.choice(width -size)
    y = np.random.choice(height-size)
    image = image[y:y+size,x:x+size]
    mask  = mask[y:y+size,x:x+size]
    return image, mask

def do_random_scale_crop(image, mask, size, mag):
    height, width = image.shape[:2]

    s = 1 + np.random.uniform(-1, 1)*mag
    s =  int(s*size)

    x = np.random.choice(width -s)
    y = np.random.choice(height-s)
    image = image[y:y+s,x:x+s]
    mask  = mask[y:y+s,x:x+s]
    if s!=size:
        image = cv2.resize(image, dsize=(size,size), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask, dsize=(size,size), interpolation=cv2.INTER_LINEAR)
    return image, mask

def do_random_rotate_crop(image, mask, size, mag=30 ):
    angle = 1+np.random.uniform(-1, 1)*mag

    height, width = image.shape[:2]
    dst = np.array([
        [0,0],[size,size], [size,0], [0,size],
    ])

    c = np.cos(angle/180*2*PI)
    s = np.sin(angle/180*2*PI)
    src = (dst-size//2)@np.array([[c, -s],[s, c]]).T
    src[:,0] -= src[:,0].min()
    src[:,1] -= src[:,1].min()

    src[:,0] = src[:,0] + np.random.uniform(0,width -src[:,0].max())
    src[:,1] = src[:,1] + np.random.uniform(0,height-src[:,1].max())

    if 0: #debug
        def to_int(f):
            return (int(f[0]),int(f[1]))

        cv2.line(image, to_int(src[0]), to_int(src[1]), (0,0,1), 16)
        cv2.line(image, to_int(src[1]), to_int(src[2]), (0,0,1), 16)
        cv2.line(image, to_int(src[2]), to_int(src[3]), (0,0,1), 16)
        cv2.line(image, to_int(src[3]), to_int(src[0]), (0,0,1), 16)
        image_show_norm('image', image, min=0, max=1)
        cv2.waitKey(1)


    transform = cv2.getAffineTransform(src[:3].astype(np.float32), dst[:3].astype(np.float32))
    image = cv2.warpAffine( image, transform, (size, size), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    mask  = cv2.warpAffine( mask, transform, (size, size), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask

#warp/elastic deform ...
#<todo>

#noise
def do_random_noise(image, mask, mag=0.1):
    height, width = image.shape[:2]
    noise = np.random.uniform(-1,1, (height, width,1))*mag
    image = image + noise
    image = np.clip(image,0,1)
    return image, mask


#intensity
def do_random_contast(image, mask, mag=0.3):
    alpha = 1 + random.uniform(-1,1)*mag
    image = image * alpha
    image = np.clip(image,0,1)
    return image, mask

def do_random_gain(image, mask, mag=0.3):
    alpha = 1 + random.uniform(-1,1)*mag
    image = image ** alpha
    image = np.clip(image,0,1)
    return image, mask

def do_random_hsv(image, mask, mag=[0.15,0.25,0.25]):
    image = (image*255).astype(np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0].astype(np.float32)  # hue
    s = hsv[:, :, 1].astype(np.float32)  # saturation
    v = hsv[:, :, 2].astype(np.float32)  # value
    h = (h*(1 + random.uniform(-1,1)*mag[0]))%180
    s =  s*(1 + random.uniform(-1,1)*mag[1])
    v =  v*(1 + random.uniform(-1,1)*mag[2])

    hsv[:, :, 0] = np.clip(h,0,180).astype(np.uint8)
    hsv[:, :, 1] = np.clip(s,0,255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(v,0,255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    image = image.astype(np.float32)/255
    return image, mask