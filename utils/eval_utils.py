import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB, CLAM_Survival
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored


def initiate_model(args, ckpt_path, device='cuda'):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, "n_classes": args.n_classes, "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb', 'clam_survival']:
        model_dict.update({"size_arg": args.model_size})
    if args.task == 'task_survival':
        model_dict.update({"n_classes": 1})

    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type == 'clam_survival':
        model = CLAM_Survival(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    _ = model.to(device)
    _ = model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset, args)
    if args.task == 'task_survival':
        patient_results, test_error, cindex, df, _ = survival_summary(model, loader, args)
        print('C Index: ', cindex)
        return model, patient_results, test_error, cindex, df
    else:
        patient_results, test_error, auc, df, _ = summary(model, loader, args)
        print('test_error: ', test_error)
        print('auc: ', auc)
        return model, patient_results, test_error, auc, df

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger

def survival_summary(model, loader, args):
    model.eval()
    all_risk_scores = []
    all_survival_times = []
    all_events = []
    all_slide_ids = []

    patient_results = {}
    
    for batch_idx, (data, survival_time, event) in enumerate(loader):
        data = data.to(device)
        survival_time = survival_time.to(device)
        event = event.to(device)
        slide_id = loader.dataset.slide_data['slide_id'].iloc[batch_idx]

        with torch.no_grad():
            risk_pred = model(data)
        
        all_risk_scores.append(risk_pred.cpu().numpy().item())  # Convert to scalar
        all_survival_times.append(survival_time.cpu().numpy().item())  # Convert to scalar
        all_events.append(event.cpu().numpy().item())  # Convert to scalar
        all_slide_ids.append(slide_id)
        
        patient_results.update({slide_id: {'slide_id': slide_id, 
                                           'risk_score': risk_pred.item(), 
                                           'survival_time': survival_time.item(), 
                                           'event': event.item()}})

    all_risk_scores = np.array(all_risk_scores)
    all_survival_times = np.array(all_survival_times)
    all_events = np.array(all_events, dtype=bool)
    
    c_index = concordance_index_censored(
        event_indicator=all_events,  
        event_time=all_survival_times,
        estimate=all_risk_scores  # No need for negative sign since higher score = higher risk
    )
    c_index = c_index[0]

    print('\nTest Set, c-index: {:.4f}'.format(c_index))
    
    results_dict = {'slide_id': all_slide_ids, 
                    'Y': all_events,  # event is treated as label
                    'Y_hat': all_risk_scores,  # risk score is treated as prediction
                    'survival_time': all_survival_times,
                    'event': all_events}
    
    # Ensure all arrays are 1-dimensional
    for key in results_dict:
        if isinstance(results_dict[key], np.ndarray):
            results_dict[key] = results_dict[key].flatten()
    
    df = pd.DataFrame(results_dict)
    return patient_results, 0.0, c_index, df, None  # 0.0 is a placeholder for test_error