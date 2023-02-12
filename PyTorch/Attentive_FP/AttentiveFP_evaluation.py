#!/usr/bin/env python3
'''conda create -n pytorch-env python=3.9 shap pandas optuna=2.10.1 xgboost scikit-learn sklearn-pandas rdkit pytorch torchvision torchaudio pytorch-cuda=11.6 cairosvg dgllife dgl=0.9.1 dgl-cuda11.6 ipython -c pytorch -c nvidia -c dglteam'''
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import dgl
from dgllife.model import AttentiveFPPredictor
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
import datetime, time


'''set basic parameters'''
filename_pkl = 'AttentiveFP_regression_best'  # load target model from the *.pkl file
split_dataset = False  # whether to split the dataset into training and test sets


'''hyperparameters for the AttentiveFPPredictor model'''
node_feat_size = 39 
edge_feat_size = 10
num_layers = 2
num_timesteps = 2
graph_feat_size = 200
n_tasks = 1   # >=2: the number of y labels for classification tasks; 1: for regression tasks
dropout = 0.2


'''input data'''
sdf_filename = ''  # input the filename of *.sdf file; leave empty for generating molecules directly from SMILES strings
df = pd.read_csv('../MolLogP_dataset.csv')
data_X = df['Smiles']  # load SMILES data here
data_y = df['MolLogP']

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

'''define other useful functions'''
def evaluation_classification(test_loader, model):
    test_size = len(test_loader.dataset)
    model.eval()   # set model to evaluation mode

    with torch.no_grad():
        for (smiles, X, y, masks) in test_loader:
            X, y = X.to(device), y.long().squeeze(-1).to(device)
            out = model(X, X.ndata['atom_feat'], X.edata['bond_feat'])
            pred_probab = nn.Softmax(dim=1)(out)
            y_test_pred_proba = pred_probab.cpu().numpy()
            y_test_pred = pred_probab.argmax(1)
            y_test = y
            accuracy_test = accuracy_score(y_test.cpu(), y_test_pred.cpu())

    print("            >>>>  Metrics based on the target model  <<<<\n")
    print(f"> Test set size: {test_size}")
    print(f"> Accuracy on the test set: {accuracy_test:.2%}\n")
    print('> Classification report on the test set:')
    print(classification_report(y_test.cpu(), y_test_pred.cpu()))

    roc_auc_test, average_precision_test = [], []
    for i in range(len(set(y_test.cpu().numpy()))):
        roc_auc_test.append(roc_auc_score(y_test.cpu(), y_test_pred_proba[:,i], multi_class='ovr'))
        average_precision_test.append(average_precision_score(y_test.cpu(), y_test_pred_proba[:,i]))
    pd.set_option('display.float_format','{:12.6f}'.format)
    pd.set_option('display.colheader_justify', 'center')
    test_reports = pd.DataFrame(np.vstack((roc_auc_test, average_precision_test)).T, columns=['ROC-AUC','AP(PR-AUC)'])
    print('> Area under the receiver operating characteristic curve (ROC-AUC) and\n  average precision (AP) which summarizes a precision-recall curve as the weighted mean\n  of precisions achieved at each threshold on the test set:\n  {}\n'.format(test_reports))

def evaluation_regression(test_loader, model):
    test_size = len(test_loader.dataset)
    model.eval()   # set model to evaluation mode

    with torch.no_grad():
        for (smiles, X, y, masks) in test_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X, X.ndata['atom_feat'], X.edata['bond_feat'])
            mse_test = mean_squared_error(y.cpu(), y_pred.cpu())
            mae_test = mean_absolute_error(y.cpu(), y_pred.cpu())
            r2_test = r2_score(y.cpu(), y_pred.cpu())

    print("            >>>>  Metrics based on the target model  <<<<\n")
    print(f"> Test set size: {test_size}")
    print(f"> Mean squared error (MSE) on the test set: {mse_test:.6f}")
    print(f"> Mean absolute error (MAE) on the test set: {mae_test:.6f}")
    print(f"> R-squared (R^2) value on the test set: {r2_test:.6f}")

def load_model(filename_pkl):
    model = AttentiveFPPredictor(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size,
                            num_layers=num_layers, num_timesteps=num_timesteps,
                            graph_feat_size=graph_feat_size, n_tasks=n_tasks, dropout=dropout)
    model = model.to(device)
    checkpoint = torch.load(filename_pkl + ".pkl", map_location = device)
    model.load_state_dict(checkpoint['net'])
    print(f"Loading the target model from the '{filename_pkl}.pkl' file...\n")

    return model

def total_running_time(end_time, start_time):
    tot_seconds = round(end_time - start_time, 2)
    days = tot_seconds // 86400
    hours = (tot_seconds % 86400) // 3600
    minutes = (tot_seconds % 86400 % 3600) // 60
    seconds = tot_seconds % 60
    print(">> Elapsed Time: {0:2d} day(s) {1:2d} hour(s) {2:2d} minute(s) {3:5.2f} second(s) <<".format(int(days), int(hours), int(minutes), seconds))

def test_main(X_test, y_test, mol_graphs_test):
    test_smi = X_test.values
    y_test = torch.tensor(y_test.values).float().reshape(-1,1)
    test_dataloader = DataLoader(dataset=list(zip(test_smi, mol_graphs_test, y_test)), batch_size=X_test.shape[0], collate_fn=collate_molgraphs)

    target_model = load_model(filename_pkl)

    if n_tasks >= 2: 
        evaluation_classification(test_dataloader, target_model)
    else:
        evaluation_regression(test_dataloader, target_model)


###############  The neural network training for classification tasks starts from here  ###############
start_time = time.time()
start_date = datetime.datetime.now()
print('      ***  PyTorch for evaluating the Attentive FP model started at {0}  ***\n'.format(start_date.strftime("%Y-%m-%d %H:%M:%S")))
if torch.cuda.is_available():
    print("CUDA is detected, and thus the task is performed with the GPU acceleration...\n")


'''split training/test sets'''
data_X, mol_graphs, data_y = match_mol_label(sdf_filename, data_X, data_y)

if split_dataset:
    print('The dataset is splited into training and test sets, and therefore the target model will be evaluated on the test set...\n')
    X_train, X_test, mol_graphs_train, mol_graphs_test, y_train, y_test = train_test_split(data_X, mol_graphs, data_y, test_size=0.2, random_state=0)
else:
    print('The whole dataset will be used to evaluate the target model...\n')
    X_test, mol_graphs_test, y_test = data_X, mol_graphs, data_y

'''define a neural network model'''
model = AttentiveFPPredictor(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size,
                            num_layers=num_layers, num_timesteps=num_timesteps,
                            graph_feat_size=graph_feat_size, n_tasks=n_tasks, dropout=dropout)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

'''test process'''
test_main(X_test, y_test, mol_graphs_test)


end_time = time.time()
end_date = datetime.datetime.now()
print('\n      ***  PyTorch for evaluating the Attentive FP model terminated at {0}  ***\n'.format(end_date.strftime("%Y-%m-%d %H:%M:%S")))
if torch.cuda.is_available():
    print("                  >>>  Thanks to the GPU acceleration with {} <<<\n".format(torch.cuda.get_device_properties(device).name))
total_running_time(end_time, start_time)


