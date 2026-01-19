import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc as calc_auc
from utils.loss_utils import FocalLoss
from sklearn.utils import check_consistent_length, check_array
import numpy
import logging



class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
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
    def __init__(self, patience=20, stop_epoch=0, verbose=False):
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
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            logging.info('EarlyStopping counter: {}, out of {}'.format(self.counter,self.patience))


            if self.counter >= self.patience and epoch > self.stop_epoch:
                
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

            logging.info('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min,val_loss))

        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train(datasets, cur, args):




    """   
        train for a single fold
    """
    logging.info('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    logging.info('\nInit train/val/test splits...')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    logging.info("Training on {} samples".format(len(train_split)))
    logging.info("Validating on {} samples".format(len(val_split)))
    logging.info("Testing on {} samples".format(len(test_split)))

    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'focal':
        loss_fn = FocalLoss().cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()

    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.model_type == 'ViLa_MIL':
        import ml_collections
        from models.model_ViLa_MIL import ViLa_MIL_Model
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.hard_or_soft = args.hard_or_soft
        model_dict = {'config': config, 'num_classes':args.n_classes}
        model = ViLa_MIL_Model(**model_dict)

    elif args.model_type == 'TransMIL':
        import ml_collections
        from models.TransMIL import TransMIL
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.hard_or_soft = args.hard_or_soft
        model_dict = {'config': config, 'num_classes':args.n_classes}
        model = TransMIL(**model_dict)
    
# def __init__(self, config, n_classes, gate = False):
    elif args.model_type == 'AMIL':
        import ml_collections
        from models.AMIL import AMIL
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.hard_or_soft = args.hard_or_soft
        model_dict = {'config': config, 'n_classes':args.n_classes,'gate':True}
        model = AMIL(**model_dict)
    
    elif args.model_type == 'WiKG':
        import ml_collections
        from models.WiKG import WiKG
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.hidden_size = 384
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.hard_or_soft = args.hard_or_soft
        model_dict = {'config': config, 'dim_in':512,'dim_hidden':384,'topk':6, 'n_classes':4,'agg_type':'bi-interaction', 'dropout':0.3, 'pool':'attn'}
        model = WiKG(**model_dict)
    
    elif args.model_type == 'RRTMIL':
        import ml_collections
        from models.RRT import RRTMIL
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.hidden_size = 384
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.hard_or_soft = args.hard_or_soft
        model_dict = {'config': config, 'input_dim':512,'mlp_dim':256,'act':'relu','n_classes':4}
        model = RRTMIL(**model_dict)

    elif args.model_type == 'surformer':
        import ml_collections
        from models.surformer import surformer
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.hidden_size = 384
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.hard_or_soft = args.hard_or_soft
        model_dict = {'config': config, 'size_arg':"small", 'dropout':0.25, 'n_classes':4}
        model = surformer(**model_dict)

    
    elif args.model_type == 'DSMIL':
        import ml_collections
        from models.DSMIL import MILNet
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.hidden_size = 384
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.hard_or_soft = args.hard_or_soft
        model_dict = {'config': config, 'in_size':512, 'dropout':0.25, 'num_class':4}
        model = MILNet(**model_dict)

    
    elif args.model_type == 'S4MIL':
        import ml_collections
        from models.S4MIL import S4Model
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.hidden_size = 384
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.hard_or_soft = args.hard_or_soft
        model_dict = {'config': config, 'in_dim':512, 'dropout':0.25, 'n_classes':4,'act':'gelu','d_model':64, 'd_state':32}
        model = S4Model(**model_dict)




    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)


    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))

    optimizer = get_optim(model, args)


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample, mode=args.mode)
    val_loader = get_split_loader(val_split,  testing = args.testing, mode=args.mode)
    test_loader = get_split_loader(test_split, testing = args.testing, mode=args.mode)
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=8, stop_epoch=0, verbose=True)
    else:
        early_stopping = None
    
    all_epoch_train_ci = []
    all_epoch_val_ci = []
    all_epoch_test_ci = []


    for epoch in range(args.max_epochs):
        train_ci = train_loop(args, epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
        stop, val_ci  = validate(args, cur, epoch, model, val_loader, args.n_classes,early_stopping, writer, loss_fn, args.results_dir)
        results_dict, test_loss, c_index_test  = summary(args.mode, model, test_loader, args.n_classes)
        logging.info('Epoch: {}, Test_loss: {:.4f}, c_index: {:.4f}'.format(epoch, test_loss,c_index_test))
        all_epoch_train_ci.append(train_ci)
        all_epoch_val_ci.append(val_ci)
        all_epoch_test_ci.append(c_index_test)

        if stop: 
            break



    if args.early_stopping: 
      
        results_dir_pt = args.results_dir+'/'+args.cancer
        model.load_state_dict(torch.load(os.path.join(results_dir_pt, "cancer={}_lr={}_max-epoch={}_drop-out={}_hard_or_soft={}-s_{}_checkpoint.pt".format(args.cancer,args.lr,args.max_epochs, args.drop_out,args.hard_or_soft,cur))))
    else:
        torch.save(model.state_dict(), os.path.join(results_dir_pt, "cancer={}_lr={}_max-epoch={}_drop-out={}_hard_or_soft={}-s_{}_checkpoint.pt".format(args.cancer,args.lr,args.max_epochs, args.drop_out,args.hard_or_soft,cur)))



    _, val_loss, c_index_val = summary(args.mode, model, val_loader, args.n_classes)
    logging.info('********************************Validate************************************')
    logging.info('Final: validate_loss: {:.4f}, val_c_index: {:.4f}'.format(val_loss,c_index_val))

    results_dict, test_loss,c_index_test1  = summary(args.mode, model, test_loader, args.n_classes)
    logging.info('********************************Testing*************************************')
    logging.info('Final: testing_loss: {:.4f}, test_c_index: {:.4f}'.format(test_loss,c_index_test1))
    

     
    return results_dict, test_loss,c_index_test1, all_epoch_train_ci, all_epoch_val_ci, all_epoch_test_ci






def _calculate_risk(h):
    r"""
    Take the logits of the model and calculate the risk for the patient 
    
    Args: 
        - h : torch.Tensor 
    
    Returns:
        - risk : torch.Tensor 
    
    """

    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()

def _update_arrays(all_risk_scores, all_censorships, all_event_times, event_time, censor, risk):
    r"""
    Update the arrays with new values 
    
    Args:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - risk : torch.Tensor
        - clinical_data_list : List
    
    Returns:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
    
    """
    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    return all_risk_scores, all_censorships, all_event_times








def train_loop(args, epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    total_loss = 0.

    all_risk_scores = []
    all_censorships = []
    all_event_times = []


    print('\n')
    for batch_idx, (data_s, coord_s, data_l, coords_l, data_f, label, status, time, disc, soft_0, soft_1, soft_2, soft_3) in enumerate(loader):
        optimizer.zero_grad()
        data_s, coord_s, data_l, coords_l, data_f, label,status,time,disc, soft_0, soft_1, soft_2, soft_3 = data_s.to(device), coord_s.to(device), data_l.to(device), coords_l.to(device), data_f.to(device), label.to(device),status.to(device),time.to(device),disc.to(device),  soft_0.to(device), soft_1.to(device), soft_2.to(device), soft_3.to(device)

        
        logits, Y_hat, loss = model(data_s, coord_s, data_l, coords_l, data_f, label,status,time,disc,soft_0, soft_1, soft_2, soft_3)

        loss.backward()
        optimizer.step()


        loss_value = loss.item()
        total_loss += loss_value 

        risk, _ = _calculate_risk(logits)

        all_risk_scores, all_censorships, all_event_times = _update_arrays(all_risk_scores, all_censorships, all_event_times, time, status, risk)

        # print('all_risk_scores',all_risk_scores)

        

    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    # print(all_risk_scores)
    # print(all_event_times)
    # print(all_censorships)

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    train_c_index = c_index

    logging.info('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))
    return  train_c_index

def validate(args, cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    all_risk_scores_val = []
    all_censorships_val = []
    all_event_times_val = []
    all_logits_val = []
    with torch.no_grad():
        for batch_idx, (data_s, coord_s, data_l, coords_l,data_f, label,status,time,disc,soft_0,soft_1,soft_2,soft_3) in enumerate(loader):
            data_s, coord_s, data_l, coords_l, data_f, label,status,time,disc,soft_0,soft_1,soft_2,soft_3 = data_s.to(device, non_blocking=True), coord_s.to(device, non_blocking=True), \
                                                                  data_l.to(device, non_blocking=True), coords_l.to(device, non_blocking=True),data_f.to(device, non_blocking=True), \
                                                                  label.to(device, non_blocking=True),status.to(device, non_blocking=True), time.to(device, non_blocking=True),disc.to(device, non_blocking=True),soft_0.to(device, non_blocking=True),soft_1.to(device, non_blocking=True),soft_2.to(device, non_blocking=True),soft_3.to(device, non_blocking=True)
            logits, Y_hat, loss = model(data_s, coord_s, data_l, coords_l,data_f, label,status,time,disc,soft_0,soft_1,soft_2,soft_3)
            loss_value = loss.item()
            val_loss += loss_value
            risk, _ = _calculate_risk(logits)
            all_risk_scores_val, all_censorships_val, all_event_times_val = _update_arrays(all_risk_scores_val, all_censorships_val, all_event_times_val, time, status, risk)
            all_logits_val.append(logits.detach().cpu().numpy())
    # print(all_logits_val)
    val_loss /= len(loader.dataset)
    all_risk_scores_val = np.concatenate(all_risk_scores_val, axis=0)
    all_censorships_val = np.concatenate(all_censorships_val, axis=0)
    all_event_times_val = np.concatenate(all_event_times_val, axis=0)
    c_index = concordance_index_censored((1-all_censorships_val).astype(bool), all_event_times_val, all_risk_scores_val, tied_tol=1e-08)[0]
    # c_index = harrell_c(all_event_times_val, all_risk_scores_val, all_censorships_val)
    # print(all_risk_scores_val)
    logging.info('Epoch: {}, validation_loss: {:.4f}, validation_c_index: {:.4f}'.format(epoch, val_loss, c_index))
    
    validation_c_index = c_index

    if early_stopping:
        assert results_dir
        results_dir_cancer = results_dir+'/'+args.cancer


        a =  os.path.join(results_dir_cancer, "cancer={}_lr={}_max-epoch={}_drop-out={}_hard_or_soft={}-s_{}_checkpoint.pt".format(args.cancer,args.lr,args.max_epochs, args.drop_out,args.hard_or_soft,cur))
        print(os.path.exists(a))
        print(a)
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir_cancer, "cancer={}_lr={}_max-epoch={}_drop-out={}_hard_or_soft={}-s_{}_checkpoint.pt".format(args.cancer,args.lr,args.max_epochs, args.drop_out,args.hard_or_soft,cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True, validation_c_index

    return False, validation_c_index

def summary(mode, model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    all_risk_scores_test = []
    all_censorships_test = []
    all_event_times_test = []

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    if(mode == 'transformer'):
        for batch_idx, (data_s, coord_s, data_l, coords_l,data_f, label, status, time, disc,soft_0,soft_1,soft_2,soft_3) in enumerate(loader):
            data_s, coord_s, data_l, coords_l, data_f, label, status, time, disc,soft_0,soft_1,soft_2,soft_3 = data_s.to(device), coord_s.to(device), data_l.to(device), coords_l.to(device), data_f.to(device), label.to(device),status.to(device),time.to(device),disc.to(device),soft_0.to(device),soft_1.to(device),soft_2.to(device),soft_3.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            with torch.no_grad():
                logits, Y_hat, loss = model(data_s, coord_s, data_l, coords_l, data_f, label,status,time,disc,soft_0,soft_1,soft_2,soft_3)
                loss_value = loss.item()
                test_loss += loss_value
                risk, _ = _calculate_risk(logits)
                
                all_risk_scores_test, all_censorships_test, all_event_times_test = _update_arrays(all_risk_scores_test, all_censorships_test, all_event_times_test, time, status, risk)
        

        test_loss /= len(loader.dataset)
        all_risk_scores_test = np.concatenate(all_risk_scores_test, axis=0)
        all_censorships_test = np.concatenate(all_censorships_test, axis=0)
        all_event_times_test = np.concatenate(all_event_times_test, axis=0)
        c_index = concordance_index_censored((1-all_censorships_test).astype(bool), all_event_times_test, all_risk_scores_test, tied_tol=1e-08)[0]
        # c_index = harrell_c(all_event_times_test, all_risk_scores_test, all_censorships_test)
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': logits, 'label': label.item()}})
            

        return patient_results, test_loss, c_index
    






def nll_loss(h, y, c, alpha=0.4, eps=1e-7, reduction='mean'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
        The weight on uncensored loss 
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    # print("h shape", h.shape)

    # make sure these are ints
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h)
    # print("hazards shape", hazards.shape)

    S = torch.cumprod(1 - hazards, dim=1)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # hazards[y] = hazards(1)
    # S[1] = S(1)
    # TODO: document and check
    # print("S_padded.shape", S_padded.shape, S_padded)


    # TODO: document/better naming
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)


    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)


    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)
    # print('s_prev.s_prev', s_prev.shape, s_prev)
    # print('h_this.shape', h_this.shape, h_this)
    # print('s_this.shape', s_this.shape, s_this)

    # c = 1 means censored. Weight 0 in this case 
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)
    

    # print('uncensored_loss.shape', uncensored_loss.shape)
    # print('censored_loss.shape', censored_loss.shape)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss



def nll_loss_soft(h, y, c, soft_0, soft_1, soft_2, soft_3, alpha=0.4, eps=1e-7, reduction='mean'):




    w_soft = torch.stack([soft_0, soft_1, soft_2, soft_3], dim=1).to(h.dtype)

    # make sure these are ints
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h)
    # print("hazards shape", hazards.shape)

    S = torch.cumprod(1 - hazards, dim=1)
    # print("S.shape", S.shape, S)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # hazards[y] = hazards(1)
    # S[1] = S(1)
    # TODO: document and check
    # print("S_padded.shape", S_padded.shape, S_padded)


    # TODO: document/better naming
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)


    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)


    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)
    # print('s_prev.s_prev', s_prev.shape, s_prev)
    # print('h_this.shape', h_this.shape, h_this)
    # print('s_this.shape', s_this.shape, s_this)

    # c = 1 means censored. Weight 0 in this case 
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    # print('w_soft',w_soft)
    # print('torch.log(S)',torch.log(S))
    censored_loss = - c * (w_soft*torch.log(S)).sum(dim=1)
    # print('censored_loss',censored_loss)
    

    # print('uncensored_loss.shape', uncensored_loss.shape)
    # print('censored_loss.shape', censored_loss.shape)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss






def _check_estimate(estimate, test_time):
    estimate = check_array(estimate, ensure_2d=False)
    if estimate.ndim != 1:
        raise ValueError(
            'Expected 1D array, got {:d}D array instead:\narray={}.\n'.format(
                estimate.ndim, estimate))
    check_consistent_length(test_time, estimate)
    return estimate


def _check_inputs(event_indicator, event_time, estimate):
    check_consistent_length(event_indicator, event_time, estimate)
    event_indicator = check_array(event_indicator, ensure_2d=False)
    event_time = check_array(event_time, ensure_2d=False)
    estimate = _check_estimate(estimate, event_time)

    if not numpy.issubdtype(event_indicator.dtype, numpy.bool_):
        raise ValueError(
            'only boolean arrays are supported as class labels for survival analysis, got {0}'.format(
                event_indicator.dtype))

    if len(event_time) < 2:
        raise ValueError("Need a minimum of two samples")

    if not event_indicator.any():
        raise ValueError("All samples are censored")

    return event_indicator, event_time, estimate


def _check_times(test_time, times):
    times = check_array(numpy.atleast_1d(times), ensure_2d=False, dtype=test_time.dtype)
    times = numpy.unique(times)

    if times.max() >= test_time.max() or times.min() < test_time.min():
        raise ValueError(
            'all times must be within follow-up time of test data: [{}; {}['.format(
                test_time.min(), test_time.max()))

    return times


def _get_comparable(event_indicator, event_time, order):
    n_samples = len(event_time)
    tied_time = 0
    comparable = {}
    i = 0
    while i < n_samples - 1:
        time_i = event_time[order[i]]
        start = i + 1
        end = start
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        # check for tied event times
        event_at_same_time = event_indicator[order[i:end]]
        censored_at_same_time = ~event_at_same_time
        for j in range(i, end):
            if event_indicator[order[j]]:
                mask = numpy.zeros(n_samples, dtype=bool)
                mask[end:] = True
                # an event is comparable to censored samples at same time point
                mask[i:end] = censored_at_same_time
                comparable[j] = mask
                tied_time += censored_at_same_time.sum()
        i = end

    return comparable, tied_time


def _estimate_concordance_index(event_indicator, event_time, estimate, weights, tied_tol=1e-8):
    order = numpy.argsort(event_time)

    comparable, tied_time = _get_comparable(event_indicator, event_time, order)

    if len(comparable) == 0:
        raise NoComparablePairException(
            "Data has no comparable pairs, cannot estimate concordance index.")

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]
        w_i = weights[order[ind]]

        est = estimate[order[mask]]

        assert event_i, 'got censored sample at index %d, but expected uncensored' % order[ind]

        ties = numpy.absolute(est - est_i) <= tied_tol
        n_ties = ties.sum()
        # an event should have a higher score
        con = est < est_i
        n_con = con[~ties].sum()

        numerator += w_i * n_con + 0.5 * w_i * n_ties
        denominator += w_i * mask.sum()

        tied_risk += n_ties
        concordant += n_con
        discordant += est.size - n_con - n_ties

    cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time


def concordance_index_censored(event_indicator, event_time, estimate, tied_tol=1e-8):
    """Concordance index for right-censored data

    The concordance index is defined as the proportion of all comparable pairs
    in which the predictions and outcomes are concordant.

    Two samples are comparable if (i) both of them experienced an event (at different times),
    or (ii) the one with a shorter observed survival time experienced an event, in which case
    the event-free subject "outlived" the other. A pair is not comparable if they experienced
    events at the same time.

    Concordance intuitively means that two samples were ordered correctly by the model.
    More specifically, two samples are concordant, if the one with a higher estimated
    risk score has a shorter actual survival time.
    When predicted risks are identical for a pair, 0.5 rather than 1 is added to the count
    of concordant pairs.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb>`
    and [1]_ for further description.

    Parameters
    ----------
    event_indicator : array-like, shape = (n_samples,)
        Boolean array denotes whether an event occurred

    event_time : array-like, shape = (n_samples,)
        Array containing the time of an event or time of censoring

    estimate : array-like, shape = (n_samples,)
        Estimated risk of experiencing an event

    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.

    Returns
    -------
    cindex : float
        Concordance index

    concordant : int
        Number of concordant pairs

    discordant : int
        Number of discordant pairs

    tied_risk : int
        Number of pairs having tied estimated risks

    tied_time : int
        Number of comparable pairs sharing the same time

    See also
    --------
    concordance_index_ipcw
        Alternative estimator of the concordance index with less bias.

    References
    ----------
    .. [1] Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L., Rosati, R.A,
           "Multivariable prognostic models: issues in developing models,
           evaluating assumptions and adequacy, and measuring and reducing errors",
           Statistics in Medicine, 15(4), 361-87, 1996.
    """
    event_indicator, event_time, estimate = _check_inputs(
        event_indicator, event_time, estimate)

    w = numpy.ones_like(estimate)

    return _estimate_concordance_index(event_indicator, event_time, estimate, w, tied_tol)