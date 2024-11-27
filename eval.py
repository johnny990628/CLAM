from __future__ import print_function

import numpy as np
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits, Generic_MIL_Survival_Dataset
import h5py
from utils.eval_utils import *
from lifelines import KaplanMeierFitter

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./RESULTS',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'clam_survival'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'task_tp53_mutation', 'task_survival'])
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--embed_dim', type=int, default=512)
parser.add_argument('--multi_scale', action='store_true', default=False)
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join(args.save_exp_code)
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

if args.multi_scale:
    args.embed_dim = int(args.embed_dim)*2

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.models_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'task_tp53_mutation':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tp53_mutation_529.csv',
                            data_dir= os.path.join(args.data_root_dir),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'normal':0, 'mutation':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_survival':
    args.n_classes = 1  # 生存分析輸出為一個連續值
    dataset = Generic_MIL_Survival_Dataset(csv_path = 'dataset_csv/survival_335.csv',
                        data_dir= os.path.join(args.data_root_dir),
                        shuffle = False, 
                        print_info = True,
                        time_col = 'time',
                        event_col = 'event',
                        patient_strat=True)

else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

def find_optimal_threshold(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    youden_index = tpr - fpr
    optimal_threshold = thresholds[np.argmax(youden_index)]
    return optimal_threshold

def gen_cm_matrics(df, auc, save_dir):
    y_true = df['Y']
    y_pred_prob = df['Y_hat']
    # 根據ground truth選擇正確的概率列
    # y_prob = np.where(y_true == 1, df['p_1'], df['p_0'])
    y_prob = df['p_1']
    threshold = find_optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    print(f"Optimal Threshold: {threshold:.4f}")
    print(f"Sensitivity: {sensitivity}, Specificity: {specificity}, F1 Score: {f1_score}")
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix')
    plt.savefig(os.path.join(save_dir, f'cm.png'))
    plt.close()
    return auc, sensitivity, specificity, f1_score, threshold

def gen_km_curve(df, save_dir):
    kmf = KaplanMeierFitter()
    median_risk = df['Y_hat'].median()
    groups = df['Y_hat'].map(lambda x: 'High Risk' if x >= median_risk else 'Low Risk')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for group in ['Low Risk', 'High Risk']:
        mask = groups == group
        kmf.fit(df['survival_time'][mask], df['event'][mask], label=group)
        kmf.plot(ax=ax)
    
    plt.title('Kaplan-Meier Curve')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.savefig(os.path.join(save_dir, 'km_curve.png'))
    plt.close()

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    all_cindex = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        if args.task == 'task_survival':
            model, patient_results, test_error, cindex, df  = eval(split_dataset, args, ckpt_paths[ckpt_idx])
            all_results.append(all_results)
            all_cindex.append(cindex)
            df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)
            gen_km_curve(df, args.save_dir)
            metrics_df = pd.DataFrame({'C Index': cindex}, index=[0])
            metrics_df.to_csv(os.path.join(args.save_dir,'metrics.csv'))
        else:
            model, patient_results, test_error, auc, df  = eval(split_dataset, args, ckpt_paths[ckpt_idx])
            all_results.append(all_results)
            all_auc.append(auc)
            all_acc.append(1-test_error)
            df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)
            auc, sensitivity, specificity, f1_score, threshold = gen_cm_matrics(df, auc, args.save_dir)
            metrics_df = pd.DataFrame({'AUC': auc, 'Sensitivity': sensitivity, 'Specificity': specificity, 
            'F1 Score': f1_score, 'Youden Threshold': threshold}, index=[0])
            metrics_df.to_csv(os.path.join(args.save_dir,'metrics.csv'))
            final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
            if len(folds) != args.k:
                save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
            else:
                save_name = 'summary.csv'
            final_df.to_csv(os.path.join(args.save_dir, save_name))
