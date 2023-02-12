#!/usr/bin/env python3
'''conda create -n pytorch-env python=3.9 shap pandas optuna=2.10.1 xgboost scikit-learn sklearn-pandas rdkit pytorch torchvision torchaudio pytorch-cuda=11.6 cairosvg dgllife dgl=0.9.1 dgl-cuda11.6 ipython -c pytorch -c nvidia -c dglteam'''
import pandas as pd
import numpy as np
import torch
from torch import nn
import dgl
from dgllife.model import AttentiveFPPredictor
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
import datetime, time, os


###############  Set required parameters here  ###############
input_filename = [  'AttentiveFP_classification_best'
                ]  # the filename.pkl files in the current working directory to be load as target models
output_filename = 'prediction_results' # the results are saved to the *.csv file
sorting_choice = 'descending' # "ascending" or "descending"
num_out_labels = 2  # >=2: the number of y labels for classification tasks; 1: for regression tasks

'''hyperparameters for the AttentiveFPPredictor model'''
node_feat_size = 39 
edge_feat_size = 10
num_layers = 2
num_timesteps = 2
graph_feat_size = 200
dropout = 0.2

'''load datasets'''
sdf_filename = 'mols'  # input the filename of *.sdf file; leave empty for generating molecules directly from SMILES strings
df = pd.read_csv('../MolLogP_dataset.csv')
data_X = df['Smiles']   # load SMILES data here


'''set target Y ranges; 'all' for all ranges'''
# y_ranges = "(data_X_y['Hardness_EN_best'] > 60.5)&(data_X_y['Hardness_EN_best'] < 61)&(data_X_y['MA300_EN_best'] > 1)"
y_ranges = "all"


'''Set the attentive fringerprints'''
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
    
    mol_graphs =[mol_to_bigraph(mol,
                           node_featurizer=atom_featurizer, 
                           edge_featurizer=bond_featurizer) for mol in mols]
    
    return mols, mol_graphs

'''match mol_graphs from *.sdf with SMILES from orginal data_X'''
def match_mol_smi(sdf_filename, data_X):
    mols, mol_graphs = get_molgraphs(sdf_filename, data_X)
    smi_list = []
    for mol in mols:
        smi = mol.GetProp('SMILES')
        smi_list.append(smi)
    
    smi_list = pd.DataFrame(smi_list, columns=[data_X.name])
    tot_mol_valid = len(mols)
    bg = dgl.batch(mol_graphs)
    print(f'Total {tot_mol_valid} molecules have been generated.\n')

    return smi_list, bg

'''define a neural network model'''
model = AttentiveFPPredictor(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size,
                            num_layers=num_layers, num_timesteps=num_timesteps,
                            graph_feat_size=graph_feat_size, n_tasks=num_out_labels, dropout=dropout)


###############  Some user-defined functions and variables  ###############
def total_running_time(end_time, start_time):
    tot_seconds = round(end_time - start_time, 2)
    days = tot_seconds // 86400
    hours = (tot_seconds % 86400) // 3600
    minutes = (tot_seconds % 86400 % 3600)// 60
    seconds = tot_seconds % 60
    print(">> Elapsed time: {0:2d} day(s) {1:2d} hour(s) {2:2d} minute(s) {3:5.2f} second(s) <<".format(int(days),int(hours),int(minutes),seconds))

def model_predictions(model, batch_mol_graphs, data_X, input_filename, sorting_choice):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sorting = True if sorting_choice == 'ascending' else False
    pkl_file = input_filename + ".pkl"
    model = model.to(device)
    checkpoint = torch.load(pkl_file, map_location = device)
    model.load_state_dict(checkpoint['net'])
    model.eval()   # set model to evaluation mode
    X = batch_mol_graphs.to(device)
    out = model(X, X.ndata['atom_feat'], X.edata['bond_feat'])

    if num_out_labels > 1:
        pred_probab = nn.Softmax(dim=1)(out)
        y_pred = pred_probab.argmax(1)
    elif num_out_labels == 1:
        y_pred = out.detach().squeeze(-1)

    data_y = pd.DataFrame(pd.Series(y_pred.cpu()), columns=[input_filename])
    data_X_y = pd.concat([data_X, data_y], axis=1)
    data_X_y_sorted = data_X_y.sort_values(input_filename,ascending=sorting,ignore_index=True)

    print('---------- Information of the {0} model ----------'.format(input_filename))
    #print('> Parameters of the {0} model:\n {1}\n'.format(input_filename, model.state_dict()))
    print('> Results based on the {0} model:\n {1}\n'.format(input_filename, data_X_y_sorted))
    return y_pred

def get_results(y_preds, data_X, input_filename, sorting_choice, y_ranges='all'):
    sorting = True if sorting_choice == 'ascending' else False
    print('\n-------->> FINAL RESULTS BASED ON THE ABOVE {0} MODEL(S) <<--------'.format(len(input_filename)))
    print('> Target Y ranges:\n  {0}\n'.format(y_ranges))
    # pd.set_option('max_colwidth', 50)
    # pd.set_option('display.max_columns', 15)
    # pd.set_option('display.max_rows', 30)
    # pd.set_option('display.width', 1000)
    data_X_y = pd.concat([data_X, y_preds], axis=1)
    #data_X_y = pd.concat([data_X_y, df['MOE_S']], axis=1)  #增加smiles与docking score

    y_ranges = eval(y_ranges)
    final_results = data_X_y[y_ranges] if type(y_ranges).__name__ == "Series" else data_X_y

    final_results = final_results.sort_values(input_filename,ascending=sorting,ignore_index=True)
    final_results = final_results if len(final_results.index) != 0 else ' @@ No record falls in the target Y ranges! @@'
    print('> Final results shown in {0} orders of target Y columns:\n {1}\n'.format(sorting_choice,final_results))

    if type(final_results).__name__ == "DataFrame":
        csv_file = output_filename + ".csv"
        final_results.to_csv(csv_file, encoding ='utf_8')
        print('~~ The results are saved as the \'{0}\' file in the current working directory ~~'.format(csv_file))
        print('   CWD: {0}\n'.format(os.getcwd()))


###############  The ML prediction script starts from here  ###############
start_time = time.time()
start_date = datetime.datetime.now()
print('***  PyTorch predictions (in target Y ranges) started at {0}  ***\n'.format(start_date.strftime("%Y-%m-%d %H:%M:%S")))

data_X, batch_mol_graphs = match_mol_smi(sdf_filename, data_X)
y_preds = []
for i in input_filename:
    y_preds.append(model_predictions(model, batch_mol_graphs, data_X, i, sorting_choice)[:,np.newaxis])

y_preds = pd.DataFrame(np.hstack(y_preds), columns=input_filename)
get_results(y_preds, data_X, input_filename, sorting_choice, y_ranges)

end_time = time.time()
end_date = datetime.datetime.now()
print('***  PyTorch predictions (in target Y ranges) terminated at {0}  ***\n'.format(end_date.strftime("%Y-%m-%d %H:%M:%S")))
total_running_time(end_time, start_time)





