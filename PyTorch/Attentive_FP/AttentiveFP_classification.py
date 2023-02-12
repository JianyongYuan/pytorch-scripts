#!/usr/bin/env python3
'''conda create -n pytorch-env python=3.9 shap pandas optuna=2.10.1 xgboost scikit-learn sklearn-pandas rdkit pytorch torchvision torchaudio pytorch-cuda=11.6 cairosvg dgllife dgl=0.9.1 dgl-cuda11.6 ipython -c pytorch -c nvidia -c dglteam'''
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score
import dgl
from dgllife.model import AttentiveFPPredictor
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
import datetime, time, sys, os, random


'''set required (hyper)parameters'''
print_details = True
restart = False  # resume the training process from the *.pkl file
early_stopping_patience = 10
weight_decay = 0  # similar to L2 regularization weight
learning_rate = 1e-3
epoches = 3   # number of iterations
batch_size = 32   # for the dataloader module
cv_fold = 3   # if cv_fold in [0, 0.5], then the validation set is FIXED with the size of one-n th of the original training set (n = int(1/cv_fold))
seed_number = 0  # set a seed number


'''hyperparameters for the AttentiveFPPredictor model'''
node_feat_size = 39 
edge_feat_size = 10
num_layers = 2
num_timesteps = 2
graph_feat_size = 200
n_tasks = 2   # the number of classifications
dropout = 0.2


'''input data'''
sdf_filename = ''  # input the filename of *.sdf file; leave empty for generating molecules directly from SMILES strings
df = pd.read_csv('../MolLogP_dataset.csv')
data_X = df['Smiles']  # load SMILES data here
data_y = df['MolLogP<2']

# print(data_X.iloc[[7,2,3]])

# exit()

'''Set the attentive fringerprints'''
def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally
        a binary mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : BatchedDGLGraph
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels. If binary masks are not
        provided, return a tensor with ones.
    """
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
    '''consider the following atom descriptors: + type/atomic number + degree (excluding neighboring hydrogen atoms) + total degree (including neighboring hydrogen atoms) + explicit valence + implicit valence + hybridization + total number of neighboring hydrogen atoms + formal charge + number of radical electrons + aromatic atom + ring membership + chirality + mass
    *The atom features include:
    (1) One hot encoding of the atom type. The supported atom types include C, N, O, S, F, Si, P, Cl, Br, Mg, Na, Ca, Fe, As, Al, I, B, V, K, Tl, Yb, Sb, Sn, Ag, Pd, Co, Se, Ti, Zn, H, Li, Ge, Cu, Au, Ni, Cd, In, Mn, Zr, Cr, Pt, Hg, Pb.
    (2) One hot encoding of the atom degree. The supported possibilities include 0 - 10.
    (3) One hot encoding of the number of implicit Hs on the atom. The supported possibilities include 0 - 6.
    (4) Formal charge of the atom.
    (5) Number of radical electrons of the atom.
    (6) One hot encoding of the atom hybridization. The supported possibilities include SP, SP2, SP3, SP3D, SP3D2.
    (7) Whether the atom is aromatic.
    (8) One hot encoding of the number of total Hs on the atom. The supported possibilities include 0 - 4.'''
    #The atom featurizer used in AttentiveFP
    atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='atom_feat')

    '''consider the following bond descriptors: + type + conjugated bond + ring membership + stereo configuration
    *The bond features include: 
    (1) One hot encoding of the bond type. The supported bond types include:
    SINGLE, DOUBLE, TRIPLE, AROMATIC.
    (2) Whether the bond is conjugated..
    (3) Whether the bond is in a ring of any size.
    (4) One hot encoding of the stereo configuration of a bond. The supported bond stereo configurations include STEREONONE, STEREOANY, STEREOZ, STEREOE, STEREOCIS, STEREOTRANS.'''
    #The bond featurizer used in AttentiveFP
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
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        #self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        #self.val_loss_min = np.Inf
        self.delta = delta
        #self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'          !!!  EarlyStopping counter: {self.counter} out of {self.patience}  !!!\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.trace_func(f'  !!!  Reset the EarlyStopping counter: New best epoch found  !!!\n')
            #self.save_checkpoint(val_loss, model)
            
    # def save_checkpoint(self, val_loss, model):
    #     '''Saves model when validation loss decrease.'''
    #     if self.verbose:
    #         self.trace_func(
    #             f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    #     torch.save(model.state_dict(), self.path)
    #     self.val_loss_min = val_loss

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

    for batch, (smiles, X, y, masks) in enumerate(train_loader):
        X, y = X.to(device), y.long().squeeze(-1).to(device)
        out = model(X, X.ndata['atom_feat'], X.edata['bond_feat'])
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()   # backward propagation: calculate gradient
        optimizer.step()  # update parameters according to the gradient
        loss_tot += loss.item() * len(y) 
        if print_details and (batch + 1) % 10 == 0:
            current_num_samples = (batch + 1) * len(y)
            print(f"    loss: {loss_tot/current_num_samples:.7f} [{current_num_samples:7d} / {train_size:7d}]")
            
    loss_tot /= train_size
    print(f"    loss: {loss_tot:.7f} [{train_size:7d} / {train_size:7d}]")

    return loss_tot

def valid(valid_loader, model, loss_fn):
    loss_tot, correct = 0.0, 0
    valid_size = len(valid_loader.dataset)
    model.eval()   # set model to evaluation mode

    with torch.no_grad():
        for (smiles, X, y, masks) in valid_loader:
            X, y = X.to(device), y.long().squeeze(-1).to(device)
            out = model(X, X.ndata['atom_feat'], X.edata['bond_feat'])
            loss = loss_fn(out, y)
            loss_tot += loss.item() * len(y)
            pred_probab = nn.Softmax(dim=1)(out)
            correct += (pred_probab.argmax(1) == y).type(torch.int).sum().item()
    
    loss_tot /= valid_size
    correct /= valid_size
    print(f"    loss: {loss_tot:.7f} [{valid_size:7d} / {valid_size:7d}]      accuracy: {(100*correct):0.1f}%")

    return loss_tot, correct

def evaluation(test_loader, train_loader, model, chechpoint, loss_fn):
    test_size, train_size = len(test_loader.dataset), len(train_loader.dataset)
    model.eval()   # set model to evaluation mode

    with torch.no_grad():
        for (smiles, X, y, masks) in test_loader:
            X, y = X.to(device), y.long().squeeze(-1).to(device)
            out = model(X, X.ndata['atom_feat'], X.edata['bond_feat'])
            loss = loss_fn(out, y)
            loss_tot_test = loss.item()
            pred_probab = nn.Softmax(dim=1)(out)
            y_test_pred_proba = pred_probab.cpu().numpy()
            y_test_pred = pred_probab.argmax(1)
            y_test = y
            accuracy_test = accuracy_score(y_test.cpu(), y_test_pred.cpu())
        
        for (smiles, X, y, masks) in train_loader:
            X, y = X.to(device), y.long().squeeze(-1).to(device)
            out = model(X, X.ndata['atom_feat'], X.edata['bond_feat'])
            loss = loss_fn(out, y)
            loss_tot_train = loss.item()
            pred_probab = nn.Softmax(dim=1)(out)
            y_train_pred = pred_probab.argmax(1)
            y_train = y
            accuracy_train = accuracy_score(y_train.cpu(), y_train_pred.cpu())

    print(f"> Dataset size: {train_size} (training) / {test_size} (test)")
    print(f"> Best epoch: {chechpoint['epoch']}")
    print(f"> Learning rate: {learning_rate:.9f} (initial) / {chechpoint['lr']:.9f} (final)")
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

def save_model(model, optimizer, valid_loss_cv, epoch, learning_rate):
    state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch, 'loss':valid_loss_cv, 'lr':learning_rate}
    filename = sys.argv[0].split(os.sep)[-1].split(".")[0]
    file_pkl = filename + "_best.pkl"
    torch.save(state, file_pkl)

def load_best_model():
    filename = sys.argv[0].split(os.sep)[-1].split(".")[0]
    file_pkl = filename + "_best.pkl"
    model = AttentiveFPPredictor(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size,
                            num_layers=num_layers, num_timesteps=num_timesteps,
                            graph_feat_size=graph_feat_size, n_tasks=n_tasks, dropout=dropout)
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

def train_main(model, optimizer, X_train, y_train, mol_graphs_train, loss_fn):
    global cv_flag
    if cv_fold <= 0.5:
        cv_flag = False
        kf = StratifiedKFold(n_splits=int(1/cv_fold))
    else:
        cv_flag = True
        kf = StratifiedKFold(n_splits=cv_fold)
    lowest_loss = float("inf")
    early_stopping = EarlyStopping(patience=early_stopping_patience)

    for epoch in range(epoches):
        if early_stopping.early_stop:
            print(f"Early stops the training because the validation loss doesn't improve after {early_stopping. patience} epoches...\n")
            break

        print(f"---------------------------  Epoch {epoch:4d}  ---------------------------")
        epoch_start = time.time()
        train_loss_cv, valid_loss_cv, valid_accuracy_cv = 0, 0, 0
        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        print(f"Learning rate: {current_lr:.9f}\n")

        for i, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
            train_smi_cv, valid_smi_cv = X_train.iloc[list(train_index)].values, X_train.iloc[list(valid_index)].values
            train_graph_cv, valid_graph_cv = [mol_graphs_train[i] for i in list(train_index)], [mol_graphs_train[i] for i in list(valid_index)]
            y_train_cv, y_valid_cv = y_train.iloc[list(train_index)], y_train.iloc[list(valid_index)]
            y_train_cv, y_valid_cv = torch.tensor(y_train_cv.values).float().reshape(-1,1), torch.tensor(y_valid_cv.values).float().reshape(-1,1)

            train_dataloader = DataLoader(dataset=list(zip(train_smi_cv, train_graph_cv, y_train_cv)),  batch_size=batch_size, collate_fn=collate_molgraphs)
            valid_dataloader = DataLoader(dataset=list(zip(valid_smi_cv, valid_graph_cv, y_valid_cv)),  batch_size=batch_size, collate_fn=collate_molgraphs)

            if cv_flag:
                print(f"--> Training Stage ({i+1}/{cv_fold}):")
            else:
                print(f"--> Training Stage:")
            train_start = time.time()
            train_loss = train(train_dataloader, model, loss_fn, optimizer)
            train_loss_cv += train_loss
            train_end = time.time()
            print(f"    time: {datetime.timedelta(seconds=round(train_end - train_start, 0))}")

            if cv_flag:
                print(f"==> Validation Stage ({i+1}/{cv_fold}):")
            else:
                print(f"==> Validation Stage:")
            (valid_loss, valid_accuracy) = valid(valid_dataloader, model, loss_fn)
            valid_loss_cv += valid_loss
            valid_accuracy_cv += valid_accuracy
            valid_end = time.time()
            print(f"    time: {datetime.timedelta(seconds=round(valid_end - train_end, 0))}\n")

            if not cv_flag:
                break
            
        if cv_flag:
            train_loss_cv /= cv_fold
            valid_loss_cv /= cv_fold
            valid_accuracy_cv /= cv_fold
            print(f"   +++  Average loss on the training sets: {train_loss_cv:14.7f}  +++")
            print(f"   +++  Average loss on the validation sets: {valid_loss_cv:12.7f}  +++")
            print(f"   +++  Average accuracy on the validation sets: {valid_accuracy_cv:8.2%}  +++\n")
        else:
            print(f"   +++  Loss on the training set: {train_loss_cv:14.7f}  +++")
            print(f"   +++  Loss on the validation set: {valid_loss_cv:12.7f}  +++")
            print(f"   +++  Accuracy on the validation set: {valid_accuracy_cv:8.2%}  +++\n")

        schedular.step(valid_loss_cv)  # update learning rate according to loss
        early_stopping(valid_loss_cv)

        if valid_loss_cv < lowest_loss:
            best_epoch = epoch
            lowest_loss = valid_loss_cv
            save_model(model, optimizer, valid_loss_cv, epoch, current_lr)

        print(f" ###  Best epoch: {best_epoch:4d}  <-->  Loss (validation): {lowest_loss:12.7f}  ###\n")
        epoch_end = time.time()
        print(f"~~~~~~~~~~  Elapsed time for the current epoch: {datetime.timedelta(seconds=round(epoch_end -   epoch_start, 0))}  ~~~~~~~~~~\n\n")

    print("             @@ Classification Model Training Done! @@\n")
    
    return best_epoch, lowest_loss

def test_main(X_train, X_test, y_train, y_test, mol_graphs_train, mol_graphs_test, loss_fn, best_epoch):
    print(f"           @@ Model Evaluation For The Best Epoch ({best_epoch}) @@\n")
    train_smi, test_smi = X_train.values, X_test.values
    y_train, y_test = torch.tensor(y_train.values).float().reshape(-1,1), torch.tensor(y_test.values).float().reshape(-1,1)
    train_dataloader = DataLoader(dataset=list(zip(train_smi, mol_graphs_train, y_train)), batch_size=X_train.shape[0], collate_fn=collate_molgraphs)
    test_dataloader = DataLoader(dataset=list(zip(test_smi, mol_graphs_test, y_test)), batch_size=X_test.shape[0], collate_fn=collate_molgraphs)

    best_model, optimizer, schedular, checkpoint = load_best_model()
    evaluation(test_dataloader, train_dataloader, best_model, checkpoint, loss_fn)


###############  The neural network training for classification tasks starts from here  ###############
start_time = time.time()
start_date = datetime.datetime.now()
set_seed(seed=seed_number)
print('      ***  PyTorch for classification tasks (Attentive FP) started at {0}  ***\n'.format(start_date.strftime("%Y-%m-%d %H:%M:%S")))
if torch.cuda.is_available():
    print("CUDA is detected, and thus the task is performed with the GPU acceleration...\n")
cv_flag = True

'''split training/test sets'''
data_X, mol_graphs, data_y = match_mol_label(sdf_filename, data_X, data_y)
X_train, X_test, mol_graphs_train, mol_graphs_test, y_train, y_test = train_test_split(data_X, mol_graphs, data_y, test_size=0.2, random_state=0)

'''define a neural network model'''
model = AttentiveFPPredictor(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size,
                            num_layers=num_layers, num_timesteps=num_timesteps,
                            graph_feat_size=graph_feat_size, n_tasks=n_tasks, dropout=dropout)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.apply(weights_init)

'''choose a loss function and an optimizer'''
weight_CE = class_weight(y_train.values.reshape(-1)).to(device)
loss_fn = CrossEntropyLoss(weight=weight_CE, reduction='mean').to(device)
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
schedular = ReduceLROnPlateau(optimizer)

if restart:
    model, optimizer, schedular, checkpoint = load_best_model()

'''training process'''
best_epoch, lowest_loss = train_main(model, optimizer, X_train, y_train, mol_graphs_train, loss_fn)

'''test process'''
test_main(X_train, X_test, y_train, y_test, mol_graphs_train, mol_graphs_test, loss_fn, best_epoch)


end_time = time.time()
end_date = datetime.datetime.now()
print('\n      ***  PyTorch for classification tasks (Attentive FP) terminated at {0}  ***\n'.format(end_date.strftime("%Y-%m-%d %H:%M:%S")))
if torch.cuda.is_available():
    print("                  >>>  Thanks to the GPU acceleration with {} <<<\n".format(torch.cuda.get_device_properties(device).name))
total_running_time(end_time, start_time)


