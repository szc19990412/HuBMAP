import sys
sys.path.append("./") 
from common import *
from Configs import *
from External import *
from Models import *
from Utils import *
from Src import read_yaml,LightningModuleReg
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime

def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--debug", action="store_true", help="debug")
    arg("--config", default="./Configs/21.yaml", type=str, help="config path") #14.yaml,单模型准确率0.871
    arg("--gpus", default="0", type=str, help="gpu numbers")
    # arg("--kfold", type=int, default=1)
    return parser

def make_output_path(output_path: Path, debug: bool) -> Path:
    if debug:
        name_tmp = output_path.name
        output_path = Path("data/shaozc/Kaggle-HuBMAP/output/tmp") / name_tmp
        cnt = 0
        while output_path.exists():
            output_path = output_path.parent / (name_tmp + f"_{cnt}")
            cnt += 1

    output_path.mkdir(parents=True, exist_ok=True)
    output_path.chmod(0o777)
    return output_path

#-------------------------------------------------------------#

class MyModelCheckpoint(ModelCheckpoint):
    def __init__(
        self, model_name: str, kfold: int, cfg_name: str, filepath: str, **args
    ):
        super(MyModelCheckpoint,self).__init__(**args)
        self.monitor = "dice"
        self.model_name = model_name
        self.kfold = kfold
        self.cfg_name = cfg_name
        self.file_path = filepath
        self.latest_path = (
            f"{self.file_path}/{cfg_name}_{model_name}_kfold_{kfold}_latest.ckpt"
        )
        self.bestloss_path = (
            f"{self.file_path}/{cfg_name}_{model_name}_kfold_{kfold}_bestloss.ckpt"
        )
        self.mode = "max" #loss取小，acc取大
        if self.mode == "min":
            self.monitor_op = np.less
            self.best_model_score = np.Inf
        elif self.mode == "max":
            self.monitor_op = np.greater
            self.best_model_score = -np.Inf

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        # Save latest
        if self.verbose > 0:
            epoch = trainer.current_epoch
            print(f'\nEpoch: {epoch}: Saving latest model to {self.latest_path}')
        if os.path.exists(self.latest_path):
            os.remove(self.latest_path)
        # self.filename = self.latest_path
        trainer.save_checkpoint(self.latest_path)

        metrics = trainer.callback_metrics #获取指标，log部分
        if metrics.get(self.monitor) is not None: #0.0
            current = metrics.get(self.monitor)
            if self.monitor_op(current, self.best_model_score):
                self.best_model_score = current
                if self.verbose > 0:
                    print(f'Saving best model to {self.bestloss_path}')
                if os.path.exists(self.bestloss_path):
                    os.remove(self.bestloss_path)
                # self.filename = self.bestloss_path
                trainer.save_checkpoint(self.bestloss_path)
#-----------------------------------------------------------------#
class MyLogger(LightningLoggerBase):
    def __init__(self, logger_df_path: Path):
        super(MyLogger, self).__init__()
        self.all_metrics = defaultdict(list)
        self.df_path = logger_df_path
    @property
    def name(self):
        return 'MyLogger'

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return '0.1'

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        if step is None:
            step = len(self.all_metrics["step"])

        if "created_at" not in metrics:
            metrics["created_at"] = str(datetime.utcnow())

        if "dice" in metrics:
            self.all_metrics["step"].append(step)
            for k, v in metrics.items():
                self.all_metrics[k].append(v)
            metrics_df = pd.DataFrame(self.all_metrics)
            metrics_df = metrics_df[sorted(metrics_df.columns)]
            metrics_df.to_csv(self.df_path, index=False)

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
#--------------------------------------------------------------------------#
def train_a_kfold(cfg, cfg_name: str, output_path: Path) -> None:
    # Checkpoint callback
    kfold = cfg['Data']['dataset']['fold']
    server = cfg['General']['server']
    checkpoint_callback = MyModelCheckpoint(
        model_name=cfg['Model']['base'],
        kfold=kfold,
        cfg_name=cfg_name,
        filepath=str(output_path),
    )

    # Logger
    logger_name = f"kfold_{str(kfold).zfill(2)}.csv"
    mylogger = MyLogger(logger_df_path=output_path /logger_name)
    # mylogger = TensorBoardLogger(save_dir=output_path, name='%s-%s-%s'%(cfg_name,kfold,IDENTIFIER),log_graph=True)
    print('\n--- [START %s] %s\n' % (IDENTIFIER, '-' * 64))


    # Trainer
    seed_torch(cfg['General']['seed'])
    seed_everything(cfg['General']['seed'])
    debug = cfg['General']['debug']
    trainer = Trainer(
        logger=mylogger,
        max_epochs=5 if debug else cfg['General']['epoch'],
        checkpoint_callback=checkpoint_callback,
        gpus=cfg['General']['gpus'],
        amp_level=cfg['General']['amp_level'], #优化等级
        precision=16, #半精度训练
        accelerator=cfg['General']['multi_gpu_mode'],
        accumulate_grad_batches=cfg['General']['grad_acc'],
        deterministic=True,
        check_val_every_n_epoch=1,
        # resume_from_checkpoint = '/data/shaozc/Kaggle-HuBMAP/output/model/5/5_CustomUneXt50_kfold_1_latest.ckpt', #加载预训练模型
        # limit_train_batches=0.01, #调试代码用
    )

    # Lightning module and start training
    model = LightningModuleReg(cfg)
    # 训练train阶段
    if server == 'local':
        trainer.fit(model)
    #------------------------------------------------#
    # 测试test阶段
    if server == 'kaggle':
        # model_dir = [f'/newdata/shaozc/Kaggle-HuBMAP/output/model/1/1_CustomUneXt50_kfold_{i}_bestloss.ckpt' 
        #             for i in range(1)]
        model_dir = ['/data/shaozc/Kaggle-HuBMAP/output/model/14/14_CustomUneXt50_kfold_1_bestloss.ckpt']
        for path in model_dir:
            new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            trainer.test(model=new_model)
            # models.append(new_model)
    


#-------------------------------------#

def main():
    args = make_parse().parse_args()

    #Read Config
    cfg = read_yaml(fpath=args.config)
    # cfg['Data']['dataset']['kfold'] = args.kfold
    # cfg['General']['debug'] = args.debug

    # Set gpu
    device_count = torch.cuda.device_count() # 可用的GPU数量
    current_device = torch.cuda.current_device() #返回当前所选设备的索引
    device_name = torch.cuda.get_device_name(current_device) #返回设备名，默认值为当前设备
    device_capability = torch.cuda.get_device_capability(current_device) # 设备的最大和最小的cuda容量, 默认值为当前设备
    device_properties = torch.cuda.get_device_properties(current_device)
    # device_properties = torch.cuda.get_device_properties(0)
    is_available = torch.cuda.is_available() # 当前CUDA是否可用
    device_cuda = torch.device("cuda") # GPU设备

    print('-' * 120)
    print('\tdevice_count:        {device_count}'.format(device_count=device_count))
    print('\tcurrent_device:      {current_device}'.format(current_device=current_device))
    print('\tdevice_name:         {device_name}'.format(device_name=device_name))
    print('\tdevice_capability:   {device_capability}'.format(device_capability=device_capability))
    print('\tdevice_properties:   {device_properties}'.format(device_properties=device_properties))
    print('\tis_available:        {is_available}'.format(is_available=is_available))
    print('\tdevice_cuda:         {device_cuda}'.format(device_cuda=device_cuda))
    print('-' * 120)
    print('\n')
    
    cfg['General']['gpus'] = list(map(int, args.gpus.split(",")))

    # Make output path
    output_path = Path("/data/shaozc/Kaggle-HuBMAP/output/model/%s"%Path(args.config).stem)
    output_path.mkdir(exist_ok=True)
    # Source code backup
    shutil.copy2(args.config, str(output_path / Path(args.config).name))
    src_backup_path = output_path / "src_backup"
    src_backup_path.mkdir(exist_ok=True)
    src_backup(input_dir=Path("./"), output_dir=src_backup_path)

    # Train start
    train_a_kfold(cfg, Path(args.config).stem, output_path)


if __name__ == "__main__":
    main()