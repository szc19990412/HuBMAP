import sys
sys.path.append("./") 
from common import *
from Models import *
from Utils import *
from main import MyModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from Src import read_tiff, save_pyramid

import torch.cuda.amp as amp
is_mixed_precision = True  # True #True #

min_size = 50

fold0 = '/data/shaozc/Kaggle-HuBMAP/output/model/10/10_CustomUneXt50_kfold_0_bestloss.ckpt'
fold1 = '/data/shaozc/Kaggle-HuBMAP/output/model/11/11_CustomUneXt50_kfold_1_bestloss.ckpt'
fold2 = '/data/shaozc/Kaggle-HuBMAP/output/model/12/12_CustomUneXt50_kfold_2_bestloss.ckpt'
fold3 = '/data/shaozc/Kaggle-HuBMAP/output/model/13/13_CustomUneXt50_kfold_3_bestloss.ckpt'

img_path = '/data/shaozc/Kaggle-HuBMAP/test/'
label_dir = '/data/shaozc/Kaggle-HuBMAP/submission.csv'
submit = '/data/shaozc/Kaggle-HuBMAP/submit/Ensemble_fold_320/'

# 把集成的模型放到这个里面
models_to_consider = {  '0': fold0, 
                        '1': fold1, 
                        # '2': fold2,
                        '3': fold3,
                     }

def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--wsi-path", default="/data/shaozc/Kaggle-HuBMAP/test/", help="test data")
    arg("--batch-size", default=1, type=int, help="config path")
    arg("--tile-size", default=640, type=int, help="tile size")
    arg("--stride-size", default=320, type=int, help="stride size")
    arg("--tile-scale", default=0.25, type=float)
    arg("--tile-min-score", default=0.25, type=float)
    return parser

def mask_to_csv( image_id, submit_dir):
        predicted = []
        for id in tqdm(image_id): #此处比较慢，需要加tqdm
            image_file = '%s/%s.tiff' % (img_path,id)
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

# load_trained_models函数，加载三个模型
def load_trained_models(model, path):
    device = torch.device('cuda:0')
    if model == ('0'):
        checkpoint = torch.load(path)
        checkpoint = checkpoint['state_dict']
        checkpoint = {k.replace('net.',''):v for k,v in checkpoint.items()}
        net = CustomUneXt50()
        net.load_state_dict(checkpoint)
        net.to(device)
        net.half()
        net = net.eval()
        print ("Loaded fold0")
        return net
    elif model == ('1'):
        checkpoint = torch.load(path)
        checkpoint = checkpoint['state_dict']
        checkpoint = {k.replace('net.',''):v for k,v in checkpoint.items()}
        net = CustomUneXt50()
        net.load_state_dict(checkpoint)
        net.to(device)
        net.half()
        net = net.eval()
        print ("Loaded fold1")
        return net
    elif model == ('2'):        
        checkpoint = torch.load(path)
        checkpoint = checkpoint['state_dict']
        checkpoint = {k.replace('net.',''):v for k,v in checkpoint.items()}
        net = CustomUneXt50()
        net.load_state_dict(checkpoint)
        net.to(device)
        net.half()
        net = net.eval()
        print ("Loaded fold2")
        return net
    elif model == ('3'):        
        checkpoint = torch.load(path)
        checkpoint = checkpoint['state_dict']
        checkpoint = {k.replace('net.',''):v for k,v in checkpoint.items()}
        net = CustomUneXt50()
        net.load_state_dict(checkpoint)
        net.to(device)
        net.half()
        print ("Loaded fold3")
        return net
    # elif model == 'all':
    #     return [load_models(mdl) for mdl in ['inception', 'dense', 'deeplabv3']]


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

import torch
import numpy as np


# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(Dataset):
	# 初始化函数，得到数据
    def __init__(self):
        self.conf_dataset = img_path
        self.label_dir = label_dir
        self.ids_all = pd.read_csv(self.label_dir).id.values
        # self.data = data_root
        # self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, idx):
        ids = self.ids_all[idx]
        image_file = self.conf_dataset + '/%s.tiff' % ids
        image = read_tiff(image_file)
        height, width = image.shape[:2]
        #通过给的json文件填充了一个mask区域，后续利用这个区域来去除背景
        # structure = draw_strcuture(read_json_as_df(json_file), height, width, structure=['Cortex'])
        structure = draw_strcuture_from_hue(image, fill=255, scale=1/32)
        return image, structure, ids
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.ids_all)

def main():
    args = make_parse().parse_args()
    # 1 加载三个模型
    models = dict()
    for i, model_name in enumerate(models_to_consider.keys()):
        models[model_name] = load_trained_models(model_name, 
                                                models_to_consider[model_name])
    # 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
    torch_data = GetLoader()



    # 读取数据
    datas = DataLoader(torch_data, batch_size=args.batch_size, num_workers=8, drop_last=True)


    tile_probability = []
    for ii, (image, structure, ids) in enumerate(datas):
        mask = None
        submit_dir = Path(submit)
        submit_dir.mkdir(exist_ok=True)
        submit_dir = str(submit_dir)
        tile_size = args.tile_size
        tile_average_step = args.stride_size
        tile_scale = args.tile_scale
        tile_min_score = args.tile_min_score

        #dataloader 取出的是tensor，且第一维度为batch数，cv要求输入格式（HWC）
        image = image.cpu().numpy().squeeze(0)
        structure = structure.cpu().numpy().squeeze(0)

        tile = to_tile(image, mask, structure, tile_scale, tile_size, tile_average_step, tile_min_score)
        tile_image = tile['tile_image'] #切片集
        tile_image = np.stack(tile_image)[..., ::-1] #沿最后一个轴叠加 == np.stack(tile_image)
        tile_image = np.ascontiguousarray(tile_image.transpose(0,3,1,2)) #连续存储，速度更快
        tile_image = tile_image.astype(np.float32)/255

        tile_probability = []
        batch = np.array_split(tile_image, len(tile_image)//2) #降低GPU
        for t,m in enumerate(batch):
            m = torch.from_numpy(m).cuda().half()
            
            p = []
            for j, model_name in enumerate(models.keys()):
                
                logit = models[model_name](m)
                p.append(torch.sigmoid(logit))

                #----
                if 1: #tta
                    logit = models[model_name](m.flip(dims=(2,)))
                    p.append(torch.sigmoid(logit.flip(dims=(2,))))

                    logit = models[model_name](m.flip(dims=(3,)))
                    p.append(torch.sigmoid(logit.flip(dims=(3,))))
                #----

            p = torch.stack(p).mean(0) #相加求平均
            # p = torch.stack(p)
            # p,_ = torch.max(p, 0)
                # p = tta.d4_image2mask(self, m)
            tile_probability.append(p.data.cpu().numpy()) #p.data去除梯度信息
            # tile_probability.append(p.numpy())
        # tile_probability = (tile_probability[0]+tile_probability[1]+tile_probability[2])/3

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
        predict = filter_small(predict,min_size)

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

    submit_dir = str(submit)
    csv_file = submit_dir +'/submission-%s.csv'%(IDENTIFIER)
    submit_id = pd.read_csv(label_dir).id.values
    df = mask_to_csv(list(submit_id), submit_dir)
    df.to_csv(csv_file, index=False)
    print(df)
        # # #---

if __name__ == "__main__":
    main()