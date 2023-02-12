#!/usr/bin/env python3
'''conda create -n pytorch-env python=3.9 shap pandas optuna=2.10.1 xgboost scikit-learn sklearn-pandas rdkit pytorch torchvision torchaudio pytorch-cuda=11.6 cairosvg dgllife dgl=0.9.1 dgl-cuda11.6 ipython -c pytorch -c nvidia -c dglteam'''
import time, os, copy
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cairosvg
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import torch
import dgl
from dgllife.model import AttentiveFPPredictor
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from dgllife.utils import mol_to_bigraph


###############  Set required parameters here  ###############
save_svg_flag = False
save_each_png = False 
cmap_scheme = 'Blues'  # colorbar scheme (e.g. 'Greens', 'Blues', 'Purples', 'bwr', 'PRGn')
each_pic_size = [300, 150]  # [width, height]
molsPerRow = 4
model_filename = 'AttentiveFP_classification_best'  # load target model from the *.pkl file
model_params = {'node_feat_size':39, 'edge_feat_size':10, 'num_layers':2, 'num_timesteps':2, 'graph_feat_size':200, 'n_tasks':2, 'dropout':0.2}
timestep = 'last'  # 0, 1, ..., num_timesteps-1, or 'last' = num_timesteps-1
index_list = [13,0,20,7,9,67,78,55,672,111,23,888,989]

'''input data'''
sdf_filename = 'mols'  # input the filename of *.sdf file; leave empty for generating molecules directly from SMILES strings
df = pd.read_csv('../MolLogP_dataset.csv')
data_X = df['Smiles']  # load SMILES data here


###############  Some user-defined functions  ###############
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
  
    print(f'Calculating molecule graphs from molecules...')
    mol_graphs =[mol_to_bigraph(mol,
                           node_featurizer=atom_featurizer, 
                           edge_featurizer=bond_featurizer) for mol in mols]
    
    return mols, mol_graphs

def get_smi(sdf_filename, data_X):
    mols, mol_graphs = get_molgraphs(sdf_filename, data_X)
    print(f'Extracting SMILES strings from molecule properties...')
    smi_list = []
    for mol in mols:
        smi = mol.GetProp('SMILES')
        smi_list.append(smi)
    tot_mol_valid = len(mols)
    print(f'\n     *****  SMILES strings of total {tot_mol_valid} molecules have been extracted  *****\n')

    return smi_list, mol_graphs, mols

def load_model(model_filename, model_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentiveFPPredictor(**model_params)
    model = model.to(device)
    checkpoint = torch.load(model_filename + ".pkl", map_location = device)
    model.load_state_dict(checkpoint['net'])
    model.eval()   # set model to evaluation mode

    return model

'''Draw 2D molecules with colored atom_weights'''
def drawmol(model, idx, dataset, timestep, cmap_scheme, each_pic_size):
    smiles, graph, mol = dataset[idx]
    bg = dgl.batch([graph])
    atom_feats, bond_feats = bg.ndata['atom_feat'], bg.edata['bond_feat']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bg, atom_feats, bond_feats = bg.to(device), atom_feats.to(device), bond_feats.to(device)

    _, atom_weights = model(bg, atom_feats, bond_feats, get_node_weight=True)
    assert timestep < len(atom_weights), 'Error: the ID of timestep is out of range!'
    atom_weights = atom_weights[timestep]
    min_value, max_value =  torch.min(atom_weights), torch.max(atom_weights)
    atom_weights = (atom_weights - min_value) / (max_value - min_value)   # MinMaxScaling atom_weights to [0, 1]

    matplotlib.use('TkAgg')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.28)
    cmap = cm.get_cmap(cmap_scheme)
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    atom_colors = {i: plt_colors.to_rgba(atom_weights[i].data.item()) for i in range(bg.number_of_nodes())}
    
    #mol = Chem.MolFromSmiles(smiles)
    mol2d = copy.copy(mol)
    rdDepictor.Compute2DCoords(mol2d)
    drawer = rdMolDraw2D.MolDraw2DSVG(each_pic_size[0], each_pic_size[1])
    drawer.SetFontSize(1)
    #op = drawer.drawOptions()
    #op.prepareMolsBeforeDrawing = False
    
    mol2d = rdMolDraw2D.PrepareMolForDrawing(mol2d)
    drawer.DrawMolecule(mol2d, highlightAtoms=range(bg.number_of_nodes()),
                             highlightBonds=[],
                             highlightAtomColors=atom_colors,
                             legend=f'index {str(idx)}')
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    #svg = svg.replace('svg:', '')

    return (mol, smiles, atom_weights.to('cpu').data.numpy(), svg)

def colorbar(timestep, cmap_scheme):
    a = np.array([[0,1]])
    plt.figure(figsize=(9, 1.5))
    plt.imshow(a, cmap=cmap_scheme)
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.8, 0.2])
    plt.colorbar(orientation='horizontal', cax=cax)
    plt.suptitle(f'Color bar for atom weights of timestep {timestep}', fontsize=16, y=0.7)
    plt.savefig('colorbar', bbox_inches='tight', dpi=300)
    #plt.show()

def save_pic(svg, idx, smi, save_svg_flag=False):  
    filename = f'idx_{str(idx)}.svg'
    with open(filename, 'w') as result:
        result.write(svg)
    if not save_svg_flag:
        svg_filename = filename
        filename = f'idx_{str(idx)}.png'
        cairosvg.svg2png(file_obj=open(svg_filename, "rb"), write_to=filename, scale=3.0)
        os.remove(svg_filename)
    print('Generated the {0:>16s} file -> {1}'.format("'"+filename+"'", smi))

    return filename

def combine_png(png_list, output, molsPerRow=4):
    tot_pngs = len(png_list)
    tot_rows = int(np.ceil(tot_pngs / molsPerRow))
    width, height = Image.open(png_list[0]).width, Image.open(png_list[0]).height
    gap = 2
    h = height * tot_rows + gap * (tot_rows+1)
    w = width * molsPerRow + gap * (molsPerRow+1)
    imgnew = Image.new('RGB', (w, h), (255, 255, 255))

    for i in range(tot_pngs):
        img = Image.open(png_list[i])
        row_idx, col_idx = i // molsPerRow, i % molsPerRow  # graph location: (row_idx, col_idx), e.g. (0,0),(0,1),(1,0)
        imgnew.paste(img, (gap*(col_idx+1)+width*col_idx, gap*(row_idx+1)+height*row_idx))
        img.close()
    imgnew.save(output)
    
def total_running_time(end_time, start_time):
    tot_seconds = round(end_time - start_time,2)
    days = tot_seconds // 86400
    hours = (tot_seconds % 86400) // 3600
    minutes = (tot_seconds % 86400 % 3600)// 60
    seconds = tot_seconds % 60
    print(">> Elapsed time: {0:2d} day(s) {1:2d} hour(s) {2:2d} minute(s) {3:5.2f} second(s) <<".format(int(days),int(hours),int(minutes),seconds))


if __name__ == '__main__':
    start_time = time.time()
    if torch.cuda.is_available():
        print("CUDA is detected, and thus the task is performed with the GPU acceleration...\n")

    print(f'Loading the attentive fingerprint(FP) model from the \'{model_filename}.pkl\' file...')
    model = load_model(model_filename, model_params)
    smi, mol_graph, mols = get_smi(sdf_filename, data_X)
    dataset = list(zip(smi, mol_graph, mols))

    if timestep == 'last':
        timestep = model_params['num_timesteps'] - 1

    print(f'The molecules with following indices are chosen to generate 2D structures colored by the atom weights of timestep {timestep}:\n{index_list}\n')
    pic_name_list = []
    pic_type = 'svg' if save_svg_flag else 'png'
    current_work_dir = os.getcwd() + os.sep
    for idx in index_list:
        mol, smi, aw, svg = drawmol(model, idx, dataset, timestep, cmap_scheme, each_pic_size)
        pic_name = save_pic(svg, idx, smi, save_svg_flag=save_svg_flag)
        pic_name_list.append(pic_name)
    if not save_svg_flag:
        combine_png(pic_name_list, 'molecules.png', molsPerRow)
        print(f'\n     *****  All above {len(pic_name_list)} files have been merged into the \'molecules.png\' file  *****')
    colorbar(timestep, cmap_scheme)
    if not save_each_png and not save_svg_flag:
        for each_png in pic_name_list:
            os.remove(each_png)
        print(f'\nInfo: The merged graph (timestep {timestep}) is saved in the following path:')
        print('      {0}\nInfo: The colorbar is saved in the following path:\n      {1}'.format(current_work_dir + 'molecules.png', current_work_dir + 'colorbar.png\n'))
    else:
        print(f'\nInfo: The target graphs (timestep {timestep}) are saved in the following path:')
        print('      {0}\nInfo: The colorbar is saved in the following path:\n      {1}'.format(current_work_dir + f'*.{pic_type}', current_work_dir + 'colorbar.png\n'))

    end_time = time.time()
    total_running_time(end_time, start_time)

    
    






