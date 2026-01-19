from __future__ import print_function
import argparse

from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_MIL_Dataset
import torch
import pandas as pd
import numpy as np
import os
import logging

from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')

parser.add_argument('--cancer', type=str, default='ESCA', help='the type of cancer')
parser.add_argument('--data_root_dir', type=str, default='/homeb/yuehl/Haley/TCGA_feature/ESCA', help='data directory')
parser.add_argument('--data_folder_s', type=str, default='20/features_conch_v1', help='dir under data directory' )
parser.add_argument('--data_folder_l', type=str, default='10/features_conch_v1', help='dir under data directory' )
parser.add_argument('--data_folder_f', type=str, default='5/features_conch_v1', help='dir under data directory' )
parser.add_argument('--max_epochs', type=int, default=200, help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=4, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='/homeb/yuehl/Haley/ViLa-MIL/result', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default='/homeb/yuehl/Haley/ViLa-MIL/splits/TCGA_ESCA')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--hard_or_soft', default=False, help='False_hard; True_soft')
parser.add_argument('--early_stopping',action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd', 'ranger'], default='ranger')
parser.add_argument('--drop_out', action='store_true', default=True, help='enabel dropout (p=0.25)')
parser.add_argument('--model_type', type=str, choices=['ViLa_MIL','TransMIL','AMIL','WiKG','RRTMIL','PatchGCN','surformer','MambaMIL','DSMIL','S4MIL'], default='TransMIL', help='type of model')
parser.add_argument('--mode', type=str, choices=['transformer'], default='transformer')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'focal'], default='ce')
parser.add_argument('--task', default='task_tcga_esca_subrisk',type=str)
parser.add_argument("--text_prompt", type=str, default=None)
parser.add_argument("--text_prompt_path", type=str, default='/homeb/yuehl/Haley/ViLa-MIL/text_prompt/TCGA_esca_two_scale_text_prompt.csv')
parser.add_argument("--prototype_number", type=int, default=16)

args = parser.parse_args()
args.text_prompt = np.array(pd.read_csv(args.text_prompt_path, header=None)).squeeze()

now = datetime.now()
year = now.year
month = now.month
day = now.day
hour = now.hour
minute = now.minute


log_file = '/homeb/yuehl/Haley/ViLa-MIL/logging/'+'20X/'+args.cancer+'/'+args.model_type+'/'
if not os.path.exists(log_file):
    os.makedirs(log_file)
label = "cancer_={}  lr_={}   max-epoch_={}  drop-out={} hard_or_soft={} in {}-{}-{} {}:{}".format(args.cancer,args.lr,args.max_epochs, args.drop_out,args.hard_or_soft,year,month,day,hour,minute) 
   
            # 创建一个Logger对象
logger = logging.getLogger()
logger.setLevel(logging.INFO)

    # 创建一个处理器将日志消息写入文件
file_handler = logging.FileHandler(log_file + label +'.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)

    # 创建一个处理器将消息打印到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

    # 设置日志消息的格式
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 将处理器添加到Logger对象
logger.addHandler(file_handler)
logger.addHandler(console_handler)



def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'lr': args.lr,
            'experiment': args.exp_code,
            'label_frac': args.label_frac,
            'seed': args.seed,
            'model_type': args.model_type,
            'mode': args.mode,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

logging.info('\nLoad Dataset')

if args.task == 'task_tcga_rcc_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/TCGA_RCC_subtyping.csv',
                                  mode = args.mode,
                                  data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
                                  data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'CCRCC':0, 'PRCC':1, 'CRCC':2},
                                  patient_strat= False,
                                  ignore=[])
                                  
elif args.task == 'task_tcga_lung_subtyping':
    args.n_classes=4
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/TCGA_Lung_subtyping.csv',
                                  mode = args.mode,
                                  data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
                                  data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                  shuffle = False,
                                  print_info = True,
                                  label_dict = {'LUAD':0, 'LUSC':1},
                                  patient_strat= False,
                                  ignore=[])
    


elif args.task == 'task_tcga_esca_subrisk':
    args.n_classes=4
    dataset = Generic_MIL_Dataset(csv_path = '/homeb/yuehl/Haley/ViLa-MIL/datasets_csv/ESCA_new.csv',
                                  mode = args.mode,
                                  data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s),
                                  data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l),
                                  data_dir_f= os.path.join(args.data_root_dir, args.data_folder_f),
                                  shuffle = False,
                                  print_info = False,
                                  label_dict = {'low':0,'Moderate':1, 'Elevated':2, 'high':3},
                                  patient_strat= False,
                                  ignore=[])

else:
    raise NotImplementedError





if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

logging.info('split_dir: {}'.format(args.split_dir))
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


logging.info("################# Settings ###################")
for key, val in settings.items():
    logging.info("{}:  {}".format(key, val))






def main(args):
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_c_index = []
    all_val_c_index = []
    

    all_fold_train = []
    all_fold_val = []
    all_fold_test = []

    all_fold_test_early_stop = []

    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, i)) 
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_loss,c_index_test, all_epoch_train_ci, all_epoch_val_ci, all_epoch_test_ci = train(datasets, i, args)
        all_fold_test_early_stop.append(c_index_test)
        all_fold_train.append(all_epoch_train_ci)
        all_fold_val.append(all_epoch_val_ci)
        all_fold_test.append(all_epoch_test_ci)

    all_fold_train_array = np.array(all_fold_train)
    all_fold_val_array = np.array(all_fold_val)
    all_fold_test_array = np.array(all_fold_test)

    mean_epoch_train = np.mean(all_fold_train_array,axis=0)
    mean_epoch_val = np.mean(all_fold_val_array,axis=0)
    mean_epoch_test = np.mean(all_fold_test_array,axis=0)


    for i in folds:
        logging.info('the training cindex of {} fold as follows:{}'.format(i, all_fold_train[i]))
    logging.info('mean epoch train cindex:{}'.format(mean_epoch_train))

    for i in folds:
        logging.info('the validation cindex of {} fold as follows:{}'.format(i, all_fold_val[i]))
    logging.info('mean epoch val cindex:{}'.format(mean_epoch_val))

    for i in folds:
        logging.info('the testing cindex of {} fold as follows:{}'.format(i, all_fold_test[i]))
    logging.info('mean epoch test cindex:{}'.format(mean_epoch_test))
    logging.info('max epoch is {}, cindex is {}'.format(np.max(mean_epoch_test),np.argmax(mean_epoch_test)))

    
    all_fold_test_early_stop = np.array(all_fold_test_early_stop)
    logging.info('each fold early stop {}'.format(all_fold_test_early_stop))
    logging.info('average cindex of each fold early stop {}'.format(np.mean(all_fold_test_early_stop,axis=0)))

    


if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


