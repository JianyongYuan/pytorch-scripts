#!/usr/bin/env python3
'''conda create -n pytorch-env python=3.9 shap pandas optuna=2.10.1 xgboost scikit-learn sklearn-pandas rdkit pytorch torchvision torchaudio pytorch-cuda=11.6 cairosvg dgllife dgl=0.9.1 dgl-cuda11.6 ipython -c pytorch -c nvidia -c dglteam'''
import numpy as np
import pandas as pd
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score
import dgl
from dgllife.model import MPNNPredictor
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
import datetime, time, sys, os, random


'''set required (hyper)parameters'''
early_stopping_patience = 10
epoches = 3   # number of iterations
batch_size = 32   # for the dataloader module
cv_fold = 3   # if cv_fold in [0, 0.5], then the validation set is FIXED with the size of one-n th of the original training set (n = int(1/cv_fold))
seed_number = 0  # set a seed number
score_opt_direction = "minimize"   # "minimize" or "maximize"
sample_method = "TPESampler"  # "TPESampler" or "RandomSampler"
timeout = None # time limit in seconds for the search of appropriate models. 
n_trials = 3


'''hyperparameters for the MPNN model'''
node_in_feats = 39  # Size for the input node features
edge_in_feats = 10  # Size for the input edge features
n_tasks = 2   # the number of classifications


'''hyperparameters to be optimized'''
def param_trial(trial):
    #learning_rate = trial.suggest_categorical('learning_rate', [1e-3])
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)  # similar to L2 regularization weight
    node_out_feats = trial.suggest_int('node_out_feats', 30, 300, step=1)
    edge_hidden_feats = trial.suggest_int('edge_hidden_feats', 10, 150, step=1)
    num_step_message_passing  = trial.suggest_int('num_step_message_passing', 2, 10, step=1)
    num_step_set2set  = trial.suggest_int('num_step_set2set', 2, 10, step=1)
    num_layer_set2set  = trial.suggest_int('num_layer_set2set', 2, 6, step=1)

    params = {'learning_rate':learning_rate, 'weight_decay':weight_decay, 'node_out_feats':node_out_feats,      
              'edge_hidden_feats':edge_hidden_feats, 'num_step_message_passing':num_step_message_passing,
              'num_step_set2set':num_step_set2set, 'num_layer_set2set':num_layer_set2set}
              
    loss_cv_valid = single_trial(**params)

    return loss_cv_valid


'''input data'''
sdf_filename = ''  # input the filename of *.sdf file; leave empty for generating molecules directly from SMILES strings
df = pd.read_csv('../MolLogP_dataset.csv')
data_X = df['Smiles']  # load SMILES data here
data_y = df['MolLogP<2']

# print(data_X.iloc[[7,2,3]])

# exit()

"""batching a list of datapoints for dataloader"""
def collate_molgraphs(data):
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))
 
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
     
    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
        
    return smiles, bg, labels, masks

def get_molgraphs(sdf_filename, smiles_data):
    atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='atom_feat')
    bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='bond_feat')

    if sdf_filename != '':
        sdf_file = sdf_filename + ".sdf"
        print(f'Generating molecules from the \'{sdf_file}\' file...')
        mols = Chem.SDMolSupplier(sdf_file)
    else:
        print(f'Generating molecules from the SMILES strings...')
        mols = [None]*len(smiles_data)
        for i, smi in enumerate(smiles_data):
            try:
                mol_name = f"{i:0>6d}"
                mols[i] = Chem.MolFromSmiles(smi)
                mols[i].SetProp("Name", f'mol_{mol_name}')
                mols[i].SetProp("SMILES", smi)
            except Exception as e:
                print(e)
        #mols = [Chem.AddHs(m) if m != None else None for m in mols]
    
    mol_graphs = [mol_to_bigraph(mol,
                           node_featurizer=atom_featurizer, 
                           edge_featurizer=bond_featurizer) for mol in mols]
    
    return mols, mol_graphs

'''match mol_graphs from *.sdf with target y labels'''
def match_mol_label(sdf_filename, data_X, data_y):
    mols, mol_graphs = get_molgraphs(sdf_filename, data_X)
    smi_list, y_label_list, mol_graphs_list = [], [], []
    for i, mol in enumerate(mols):
        smi = mol.GetProp('SMILES')
        if smi in data_X.values:
            index = data_X.loc[data_X == smi].index[0]
            y_label = data_y[index]
            smi_list.append(smi)
            y_label_list.append(y_label)
            mol_graphs_list.append(mol_graphs[i])
    
    smi_list = pd.DataFrame(smi_list, columns=[data_X.name])
    y_label_list = pd.DataFrame(y_label_list, columns=[data_y.name])
    tot_mol_valid = len(mol_graphs_list)
    print(f'Total {tot_mol_valid} molecules matched with y labels by using SMILES strings.\n')

    return smi_list, mol_graphs_list, y_label_list

'''early stops the training if validation loss doesn't improve after a given patience'''
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

'''count trials'''
class CallingCounter(object):
    def __init__ (self, func):
        self.func = func
        self.count = 0
        
    def __call__ (self, *args, **kwargs):
        self.count += 1

        return self.func(*args, **kwargs)

'''calculate the class weight of y'''
def class_weight(y):
    recip_freq = len(y) / (len(set(y)) * np.bincount(y).astype(np.float64))
    recip_freq =  torch.FloatTensor(recip_freq)

    return recip_freq

'''initialize weights of model'''
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0.0)

'''define other useful functions'''
def train(train_loader, model, loss_fn, optimizer):
    loss_tot = 0.0
    train_size = len(train_loader.dataset)
    model.train()   # set model to training mode

    for smiles, X, y, masks in train_loader:
        X, y = X.to(device), y.long().squeeze(-1).to(device)
        out = model(X, X.ndata['atom_feat'], X.edata['bond_feat'])
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()   # backward propagation: calculate gradient
        optimizer.step()  # update parameters according to the gradient
        loss_tot += loss.item() * len(y) 
        
    loss_tot /= train_size
    
    return loss_tot

def valid(valid_loader, model, loss_fn):
    loss_tot, correct = 0.0, 0
    valid_size = len(valid_loader.dataset)
    model.eval()   # set model to evaluation mode

    with torch.no_grad():
        for smiles, X, y, masks in valid_loader:
            X, y = X.to(device), y.long().squeeze(-1).to(device)
            out = model(X, X.ndata['atom_feat'], X.edata['bond_feat'])
            loss = loss_fn(out, y)
            loss_tot += loss.item() * len(y)
            pred_probab = nn.Softmax(dim=1)(out)
            correct += (pred_probab.argmax(1) == y).type(torch.int).sum().item()
    
    loss_tot /= valid_size
    correct /= valid_size

    return loss_tot, correct

def evaluation(test_loader, train_loader, model, chechpoint, loss_fn):
    test_size, train_size = len(test_loader.dataset), len(train_loader.dataset)
    model.eval()   # set model to evaluation mode

    with torch.no_grad():
        for smiles, X, y, masks in test_loader:
            X, y = X.to(device), y.long().squeeze(-1).to(device)
            out = model(X, X.ndata['atom_feat'], X.edata['bond_feat'])
            loss = loss_fn(out, y)
            loss_tot_test = loss.item()
            pred_probab = nn.Softmax(dim=1)(out)
            y_test_pred_proba = pred_probab.cpu().numpy()
            y_test_pred = pred_probab.argmax(1)
            y_test = y
            accuracy_test = accuracy_score(y_test.cpu(), y_test_pred.cpu())
        
        for smiles, X, y, masks in train_loader:
            X, y = X.to(device), y.long().squeeze(-1).to(device)
            out = model(X, X.ndata['atom_feat'], X.edata['bond_feat'])
            loss = loss_fn(out, y)
            loss_tot_train = loss.item()
            pred_probab = nn.Softmax(dim=1)(out)
            y_train_pred = pred_probab.argmax(1)
            y_train = y
            accuracy_train = accuracy_score(y_train.cpu(), y_train_pred.cpu())

    print(f"> Dataset size: {train_size} (training) / {test_size} (test)")
    print(f"> Best trial: {chechpoint['trial']}")
    print(f"> Best epoch: {chechpoint['epoch']}")
    print(f"> Learning rate: {study.best_params['learning_rate']:.9f} (initial) / {chechpoint['lr']:.9f} (final)")
    if cv_flag:
        print(f"> Average loss on cross-validation sets: {chechpoint['loss']:.7f}")
    else:
        print(f"> Loss on the validation set: {chechpoint['loss']:.7f}")

    print("\n            >>>>  Metrics based on the best model  <<<<\n")
    print(f"> Loss on the training set: {loss_tot_train:.7f}")
    print(f"> Loss on the test set: {loss_tot_test:.7f}")
    print(f"> Accuracy on the training set: {accuracy_train:.2%}")
    print(f"> Accuracy on the test set: {accuracy_test:.2%}\n")
    print('> Classification report on the training set:')
    print(classification_report(y_train.cpu(), y_train_pred.cpu()))
    print('> Classification report on the test set:')
    print(classification_report(y_test.cpu(), y_test_pred.cpu()))

    roc_auc_test, average_precision_test = [], []
    for i in range(len(set(y_train.cpu().numpy()))):
        roc_auc_test.append(roc_auc_score(y_test.cpu(), y_test_pred_proba[:,i], multi_class='ovr'))
        average_precision_test.append(average_precision_score(y_test.cpu(), y_test_pred_proba[:,i]))
    pd.set_option('display.float_format','{:12.6f}'.format)
    pd.set_option('display.colheader_justify', 'center')
    test_reports = pd.DataFrame(np.vstack((roc_auc_test, average_precision_test)).T, columns=['ROC-AUC','AP(PR-AUC)'])
    print('> Area under the receiver operating characteristic curve (ROC-AUC) and\n  average precision (AP) which summarizes a precision-recall curve as the weighted mean\n  of precisions achieved at each threshold on the test set:\n  {}\n'.format(test_reports))

def save_model(state):
    filename = sys.argv[0].split(os.sep)[-1].split(".")[0]
    file_pkl = filename + "_best.pkl"
    torch.save(state, file_pkl)

def load_best_model():
    best_params = study.best_params
    filename = sys.argv[0].split(os.sep)[-1].split(".")[0]
    file_pkl = filename + "_best.pkl"
    model = MPNNPredictor(node_in_feats=node_in_feats, edge_in_feats=edge_in_feats,
                        node_out_feats=best_params['node_out_feats'], edge_hidden_feats=best_params['edge_hidden_feats'],
                        num_step_message_passing=best_params['num_step_message_passing'], n_tasks=n_tasks,
                        num_step_set2set=best_params['num_step_set2set'], num_layer_set2set=best_params['num_layer_set2set'])
    model = model.to(device)
    checkpoint = torch.load(file_pkl, map_location = device)
    model.load_state_dict(checkpoint['net'])
    optimizer = Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    schedular = ReduceLROnPlateau(optimizer)
    print(f"Loading the best model from the '{file_pkl}' file...\n")

    return model, optimizer, schedular, checkpoint

def total_running_time(end_time, start_time):
    tot_seconds = round(end_time - start_time, 2)
    days = tot_seconds // 86400
    hours = (tot_seconds % 86400) // 3600
    minutes = (tot_seconds % 86400 % 3600) // 60
    seconds = tot_seconds % 60
    print(">> Elapsed Time: {0:2d} day(s) {1:2d} hour(s) {2:2d} minute(s) {3:5.2f} second(s) <<".format(int(days), int(hours), int(minutes), seconds))

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    # dgl.random.seed(seed)
    # dgl.seed(seed)


def train_main(model, optimizer, schedular, X_train, y_train, mol_graphs_train, loss_fn):
    global cv_flag
    if cv_fold <= 0.5:
        cv_flag = False
        kf = StratifiedKFold(n_splits=int(1/cv_fold))
    else:
        cv_flag = True
        kf = StratifiedKFold(n_splits=cv_fold)
    lowest_loss = float("inf")
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    save_state = {}

    for epoch in range(epoches):
        if early_stopping.early_stop:
            break

        train_loss_cv, valid_loss_cv, valid_accuracy_cv = 0, 0, 0
        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]

        for train_index, valid_index in kf.split(X_train, y_train):
            train_smi_cv, valid_smi_cv = X_train.iloc[list(train_index)].values, X_train.iloc[list(valid_index)].values
            train_graph_cv, valid_graph_cv = [mol_graphs_train[i] for i in list(train_index)], [mol_graphs_train[i] for i in list(valid_index)]
            y_train_cv, y_valid_cv = y_train.iloc[list(train_index)], y_train.iloc[list(valid_index)]
            y_train_cv, y_valid_cv = torch.tensor(y_train_cv.values).float().reshape(-1,1), torch.tensor(y_valid_cv.values).float().reshape(-1,1)

            train_dataloader = DataLoader(dataset=list(zip(train_smi_cv, train_graph_cv, y_train_cv)), batch_size=batch_size, collate_fn=collate_molgraphs)
            valid_dataloader = DataLoader(dataset=list(zip(valid_smi_cv, valid_graph_cv, y_valid_cv)), batch_size=batch_size, collate_fn=collate_molgraphs)

            train_loss = train(train_dataloader, model, loss_fn, optimizer)
            train_loss_cv += train_loss

            valid_loss, valid_accuracy = valid(valid_dataloader, model, loss_fn)
            valid_loss_cv += valid_loss
            valid_accuracy_cv += valid_accuracy
        
            if not cv_flag:
                break

        if cv_flag:
            train_loss_cv /= cv_fold
            valid_loss_cv /= cv_fold
            valid_accuracy_cv /= cv_fold

        schedular.step(valid_loss_cv)  # update learning rate according to loss
        early_stopping(valid_loss_cv)

        if valid_loss_cv < lowest_loss:
            best_epoch = epoch
            lowest_loss = valid_loss_cv
            save_state['net'] = model.state_dict()
            save_state['optimizer'] = optimizer.state_dict()
            save_state['epoch'] = best_epoch
            save_state['loss'] = lowest_loss
            save_state['lr'] = current_lr
    
    return save_state, lowest_loss

def test_main(X_train, X_test, y_train, y_test, mol_graphs_train, mol_graphs_test):
    train_smi, test_smi = X_train.values, X_test.values
    y_train, y_test = torch.tensor(y_train.values).float().reshape(-1,1), torch.tensor(y_test.values).float().reshape(-1,1)
    train_dataloader = DataLoader(dataset=list(zip(train_smi, mol_graphs_train, y_train)), batch_size=X_train.shape[0], collate_fn=collate_molgraphs)
    test_dataloader = DataLoader(dataset=list(zip(test_smi, mol_graphs_test, y_test)), batch_size=X_test.shape[0], collate_fn=collate_molgraphs)

    best_model, optimizer, schedular, checkpoint = load_best_model()
    
    loss_fn = CrossEntropyLoss(weight=weight_CE, reduction='mean').to(device)
    evaluation(test_dataloader, train_dataloader, best_model, checkpoint, loss_fn)

@CallingCounter
def single_trial(node_out_feats, edge_hidden_feats, num_step_message_passing, num_step_set2set, num_layer_set2set, weight_decay, learning_rate):
    model = MPNNPredictor(node_in_feats=node_in_feats, edge_in_feats=edge_in_feats,
                            node_out_feats=node_out_feats, edge_hidden_feats=edge_hidden_feats,
                            num_step_message_passing=num_step_message_passing, n_tasks=n_tasks,
                            num_step_set2set=num_step_set2set, num_layer_set2set=num_layer_set2set)
    model = model.to(device)
    model.apply(weights_init)

    loss_fn = CrossEntropyLoss(weight=weight_CE, reduction='mean').to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    schedular = ReduceLROnPlateau(optimizer)

    single_trial_state, lowest_loss = train_main(model, optimizer, schedular, X_train, y_train, mol_graphs_train, loss_fn)

    if single_trial_state['loss'] < best_trial_state['loss']:
        best_trial_state['net'] = single_trial_state['net']
        best_trial_state['optimizer'] = single_trial_state['optimizer']
        best_trial_state['epoch'] = single_trial_state['epoch']
        best_trial_state['loss'] = single_trial_state['loss']
        best_trial_state['lr'] = single_trial_state['lr']
        best_trial_state['trial'] = single_trial.count - 1

    return lowest_loss

def set_sampler(sample_method, seed_number):
    if sample_method == "TPESampler":
        sampler = optuna.samplers.TPESampler(seed=seed_number)
    elif sample_method == "RandomSampler":
        sampler = optuna.samplers.RandomSampler(seed=seed_number)
    
    return sampler
    

###############  The neural network training for classification tasks starts from here  ###############
start_time = time.time()
start_date = datetime.datetime.now()
set_seed(seed=seed_number)
print('      ***  PyTorch for classification tasks (Message Passing Neural Networks, MPNN) started at {0}  ***\n'.format(start_date.strftime("%Y-%m-%d %H:%M:%S")))
if torch.cuda.is_available():
    print("CUDA is detected, and thus the task is performed with the GPU acceleration...\n")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cv_flag = True

'''split training/test sets'''
data_X, mol_graphs, data_y = match_mol_label(sdf_filename, data_X, data_y)
X_train, X_test, mol_graphs_train, mol_graphs_test, y_train, y_test = train_test_split(data_X, mol_graphs, data_y, test_size=0.2, random_state=0)

best_trial_state = {'net':None, 'optimizer':None, 'epoch':0, 'loss':float("inf"), 'lr':None, 'trial':0}
weight_CE = class_weight(y_train.values.reshape(-1)).to(device)

study = optuna.create_study(sampler=set_sampler(sample_method, seed_number), direction=score_opt_direction, study_name=sys.argv[0].split(os.sep)[-1].split(".")[0])
study.optimize(param_trial, n_trials=n_trials, timeout=timeout, n_jobs=1)

best_trial = study.best_trial
lowest_loss = study.best_value
best_params = study.best_params
print(f"\n-> Best trial information: {best_trial}\n")
print(f"-> Sampling algorithm: {study.sampler.__class__.__name__}")
if cv_flag:
    print(f"-> Lowest average loss on cross-validation sets: {lowest_loss:.7f}") 
else:
    print(f"-> Lowest average loss on the validation set: {lowest_loss:.7f}") 
print("-> Best params: ")
for key, value in best_params.items(): 
    print(f"  {key:>25s}: {value}")

'''retrieve and evaluate the model with the best hyperparameters'''
print(f"\nSaving the best model based on the above optimal hyperparameters...\n")
save_model(best_trial_state)
test_main(X_train, X_test, y_train, y_test, mol_graphs_train, mol_graphs_test)


end_time = time.time()
end_date = datetime.datetime.now()
print('\n      ***  PyTorch for classification tasks (Message Passing Neural Networks, MPNN) terminated at {0}  ***\n'.format(end_date.strftime("%Y-%m-%d %H:%M:%S")))
if torch.cuda.is_available():
    print("                  >>>  Thanks to the GPU acceleration with {} <<<\n".format(torch.cuda.get_device_properties(device).name))
total_running_time(end_time, start_time)


