__author__ = 'shaozc'
#根据提供的json文件，确定mask的区域，后做切片

import sys
sys.path.append("./") 
from common import *
from Configs import *
from External import *
from Models import *
from Utils import *
from Src import read_tiff

def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--tile-scale", type=int, default=0.25)
    arg("--tile-min-score", type=int, default=0.25)
    arg("--tile-size", type=int, default=256)
    arg("--tile-average-step", type=int, default=128)
    arg("--data-dir", type=str, default='/newdata/shaozc/Kaggle-HuBMAP')
    arg("--train-tile-dir", type=str, default=f'/newdata/shaozc/Kaggle-HuBMAP/0.25_256_128_train_corrected')
    return parser.parse_args()



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