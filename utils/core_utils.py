import numpy as np
import torch
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits, save_splits_survival
from utils.loss_utils import NLLLoss, SurvPLE
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB, CLAM_Survival, GIGAPATH_Survival
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
from torch.cuda.amp import GradScaler, autocast

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, stop_epoch=5, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score - self.delta < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

class EarlyStoppingSurvival:
    """Early stops the training if validation C-index doesn't improve after a given patience."""
    def __init__(self, patience=10, stop_epoch=5, verbose=False, delta=0.0):
        """
        Args:
            patience (int): How long to wait after last time validation C-index improved.
            stop_epoch (int): Earliest epoch possible for stopping.
            verbose (bool): If True, prints a message for each C-index improvement.
            delta (float): Minimum change in C-index to qualify as an improvement.
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.val_cindex_max = -np.inf

    def __call__(self, epoch, val_cindex, model, ckpt_name='checkpoint_cindex.pt'):
        
        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_cindex, model, ckpt_name)
        elif score < self.best_score :
            self.counter += 1
            print(f'EarlyStoppingCIndex counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_cindex, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_cindex, model, ckpt_name):
        """Saves model when validation C-index increases."""
        if self.verbose:
            print(f'Validation C-index increased ({self.val_cindex_max:.6f} --> {val_cindex:.6f}). Saving model...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_cindex_max = val_cindex



def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    
    if args.task == 'task_survival':
        save_splits_survival(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    else:
        save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.task == 'task_survival':
        loss_fn = NLLLoss()
        # loss_fn = SurvPLE()
    elif args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
    else:
        loss_fn = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        loss_fn = loss_fn.cuda()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes,
                  "embed_dim": args.embed_dim*2 if args.multi_scale else args.embed_dim,
                  }
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb', 'clam_survival', 'gigapath_survival']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
        else:
            instance_loss_fn = nn.CrossEntropyLoss()

        if device.type == 'cuda':
            instance_loss_fn = instance_loss_fn.cuda()

        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_survival':
            model = CLAM_Survival(**model_dict)
        elif args.model_type == 'gigapath_survival':
            model = GIGAPATH_Survival(**model_dict)
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    is_survival = args.task == 'task_survival'
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample, survival=is_survival)
    val_loader = get_split_loader(val_split,  testing = args.testing, survival=is_survival)
    test_loader = get_split_loader(test_split, testing = args.testing, survival=is_survival)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStoppingSurvival(patience = 30, stop_epoch=10, verbose=True) if args.task == 'task_survival' else EarlyStopping(patience = 30, stop_epoch=10, verbose = True)
        # early_stopping = EarlyStopping(patience = 20, stop_epoch=10, verbose = True, delta=1e-9)
    else:
        early_stopping = None
    print('Done!')

    print('\nSetup Learning Rate Scheduler', end=' ')
    if args.lr_scheduler:
        max_epochs = 50
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
        scheduler = SequentialLR(optimizer, 
                                schedulers=[warmup_scheduler, cosine_scheduler],
                                milestones=[args.warmup_epochs])
    else:
        scheduler = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        elif args.model_type == 'clam_survival' or args.model_type == 'gigapath_survival':
            train_loop_survival(epoch, model, train_loader, optimizer, writer, loss_fn, args.batch_size)
            stop = validate_survival(cur, epoch, model, val_loader,
                early_stopping, writer, loss_fn, args.results_dir)
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break
        if scheduler:
            scheduler.step()
            writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch)
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.10f}")

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    
    if args.task == 'task_survival':
         # For validation
        _, val_cindex = summary_survival(model, val_loader)
        print('Val C-index: {:.4f}'.format(val_cindex))

        # For testing
        results_dict, test_cindex = summary_survival(model, test_loader)
        print('Test C-index: {:.4f}'.format(test_cindex))
        # 記錄到tensorboard
        if writer:
            writer.add_scalar('final/val_c_index', val_cindex, 0)
            writer.add_scalar('final/test_c_index', test_cindex, 0)
            
            # 如果需要記錄其他統計資訊，可以在這裡添加
            # 例如：記錄中位生存時間、事件比例等
            event_ratio = np.mean([v['event'] for v in results_dict.values()])
            median_time = np.median([v['survival_time'] for v in results_dict.values()])
            
            writer.add_scalar('final/event_ratio', event_ratio, 0)
            writer.add_scalar('final/median_survival_time', median_time, 0)
            
            writer.close()
        return results_dict, test_cindex, val_cindex, 0, 0

    else:
        _, val_error, val_auc, _= summary(model, val_loader, args.n_classes)
        print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

        results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
        print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

        for i in range(args.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

            if writer:
                writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
        if writer:
            writer.add_scalar('final/val_error', val_error, 0)
            writer.add_scalar('final/val_auc', val_auc, 0)
            writer.add_scalar('final/test_error', test_error, 0)
            writer.add_scalar('final/test_auc', test_auc, 0)
            writer.close()
        return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros((len(loader), n_classes))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        label = label.float()  # 確保標籤是浮點型
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.cpu().numpy()

        slide_id = slide_ids.iloc[batch_idx]
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.cpu().numpy()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels[:, 1], all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = all_labels
        for class_idx in range(n_classes):
            if class_idx in binary_labels.argmax(axis=1):
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger


def train_loop_survival(epoch, model, loader, optimizer, writer=None, loss_fn=None, batch_size=16):
    model.train()

    all_loss = []
    all_risk_scores = []
    all_survival_times = []
    all_events = []
    all_cindex = []
    
    for batch_idx, (data, survival_time, event) in enumerate(loader):
        data = data.to(device)
        survival_time = survival_time.to(device)
        event = event.to(device)
        risk_pred = model(data)

        all_risk_scores.append(risk_pred)
        all_survival_times.append(survival_time)
        all_events.append(event)

        if (batch_idx + 1) % batch_size == 0 or (batch_idx + 1) == len(loader):
            batch_risk_scores = torch.cat(all_risk_scores, dim=0)  # [batch_size]
            batch_survival_times = torch.cat(all_survival_times, dim=0)  # [batch_size]
            batch_events = torch.cat(all_events, dim=0)  # [batch_size]

            batch_risk_scores = batch_risk_scores.view(-1)

            loss = loss_fn(batch_risk_scores, batch_survival_times, batch_events)
            all_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            c_index = concordance_index_censored(
                batch_events.cpu().numpy().astype(bool),
                batch_survival_times.cpu().numpy(),
                -batch_risk_scores.detach().cpu().numpy()
            )[0]

            all_cindex.append(c_index)

            all_risk_scores = []
            all_survival_times = []
            all_events = []

            print('Processed {} WSI, current loss: {:.10f}'.format(batch_idx + 1, loss.item()))

    avg_loss = sum(all_loss) / len(all_loss)
    avg_cindex = sum(all_cindex)/ len(all_cindex)

    if writer:
        writer.add_scalar('train/loss', avg_loss, epoch) 
        writer.add_scalar('train/cindex', avg_cindex, epoch)

    print('Epoch: {}, train_loss: {:.10f}, train_cindex: {:.4f}'.format(epoch, avg_loss, avg_cindex))

def validate_survival(cur, epoch, model, loader, early_stopping=None, writer=None, loss_fn=None, results_dir=None):
    model.eval()
    all_loss = []
    all_risk_scores = []
    all_survival_times = []
    all_events = []
    
    with torch.no_grad():
        for batch_idx, (data, survival_time, event) in enumerate(loader):
            data = data.to(device)
            survival_time = survival_time.to(device)
            event = event.to(device)
            risk_pred = model(data)

            all_risk_scores.append(risk_pred)
            all_survival_times.append(survival_time)
            all_events.append(event)
    
    all_risk_scores = torch.cat(all_risk_scores, dim=0).view(-1)  # [total_samples]
    all_survival_times = torch.cat(all_survival_times, dim=0)  # [total_samples]
    all_events = torch.cat(all_events, dim=0)  # [total_samples]
    
    # 計算損失
    loss = loss_fn(all_risk_scores, all_survival_times, all_events)
    avg_loss = loss.item()
    
    # 計算 C-index
    c_index = concordance_index_censored(
        all_events.cpu().numpy().astype(bool),
        all_survival_times.cpu().numpy(),
        -all_risk_scores.cpu().numpy().squeeze()
    )[0]
    
    if writer:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/c_index', c_index, epoch)
    
    print('\nVal Set, val_loss: {:.10f}, c-index: {:.4f}'.format(avg_loss, c_index))
    
    if early_stopping:
        assert results_dir
        early_stopping(epoch, c_index, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True
    
    return False

def summary_survival(model, loader):
    model.eval()
    all_risk_scores = []
    all_survival_times = []
    all_events = []
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    with torch.no_grad():
        for batch_idx, (data, survival_time, event) in enumerate(loader):
            data = data.to(device)
            risk_pred = model(data)
            
            slide_id = slide_ids.iloc[batch_idx]
            patient_results.update({
                slide_id: {
                    'slide_id': np.array(slide_id),
                    'risk_score': risk_pred.cpu().numpy(),
                    'survival_time': survival_time.numpy(),
                    'event': event.numpy()
                }
            })
            
            all_risk_scores.append(risk_pred.cpu().numpy())
            all_survival_times.append(survival_time.numpy())
            all_events.append(event.numpy())
    
    all_events = np.concatenate(all_events).astype(bool)
    all_survival_times = np.concatenate(all_survival_times)
    all_risk_scores = np.concatenate(all_risk_scores).squeeze()
    
    c_index = concordance_index_censored(all_events,all_survival_times,-all_risk_scores)[0]
    
    return patient_results, c_index

def train_loop_survival_amp(epoch, model, loader, optimizer, writer=None, loss_fn=None, batch_size=16, use_fp16=True):
    model.train()

    all_loss = []
    all_risk_scores = []
    all_survival_times = []
    all_events = []
    all_cindex = []

    # 初始化 GradScaler（仅在 FP16 模式下启用）
    scaler = GradScaler() if use_fp16 else None

    for batch_idx, (data, survival_time, event, coords) in enumerate(loader):
        data = data.to(device)
        survival_time = survival_time.to(device)
        event = event.to(device)
        coords = coords.to(device)

        # 开启自动混合精度上下文
        with autocast(enabled=use_fp16):
            risk_pred = model(data, coords)

            all_risk_scores.append(risk_pred)
            all_survival_times.append(survival_time)
            all_events.append(event)

            # 在每个 batch 或 loader 的末尾计算 loss 和优化
            if (batch_idx + 1) % batch_size == 0 or (batch_idx + 1) == len(loader):
                batch_risk_scores = torch.cat(all_risk_scores, dim=0)  # [batch_size]
                batch_survival_times = torch.cat(all_survival_times, dim=0)  # [batch_size]
                batch_events = torch.cat(all_events, dim=0)  # [batch_size]

                batch_risk_scores = batch_risk_scores.view(-1)

                loss = loss_fn(batch_risk_scores, batch_survival_times, batch_events)
                all_loss.append(loss.item())

                # 优化器零梯度
                optimizer.zero_grad()

                # 在 FP16 模式下放大损失并计算梯度
                if use_fp16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # 清理缓存
                torch.cuda.empty_cache()

                # 计算 C-index
                c_index = concordance_index_censored(
                    batch_events.cpu().numpy().astype(bool),
                    batch_survival_times.cpu().numpy(),
                    -batch_risk_scores.detach().cpu().numpy()
                )[0]

                all_cindex.append(c_index)

                # 重置临时变量
                all_risk_scores = []
                all_survival_times = []
                all_events = []

                print('Processed {} WSI, current loss: {:.10f}'.format(batch_idx + 1, loss.item()))

    avg_loss = sum(all_loss) / len(all_loss)
    avg_cindex = sum(all_cindex) / len(all_cindex)

    if writer:
        writer.add_scalar('train/loss', avg_loss, epoch) 
        writer.add_scalar('train/cindex', avg_cindex, epoch)

    print('Epoch: {}, train_loss: {:.10f}, train_cindex: {:.4f}'.format(epoch, avg_loss, avg_cindex))