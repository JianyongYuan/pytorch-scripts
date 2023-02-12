#!/usr/bin/env python3
'''conda create -n pytorch-env python=3.9 shap pandas optuna=2.10.1 xgboost scikit-learn sklearn-pandas rdkit pytorch torchvision torchaudio pytorch-cuda=11.6 cairosvg dgllife dgl=0.9.1 dgl-cuda11.6 ipython -c pytorch -c nvidia -c dglteam'''
import time, os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from openbabel import pybel

def get_3D_mols(smiles_strs, xyz_flag):
    start_time = time.time()
    counter = {"rdkit":0, "pybel":0,"failed":0}
    nsmiles = len(smiles_strs)
    mols = [None]*nsmiles

    for i, smi in enumerate(smiles_strs):
        try:
            mols[i] = Chem.MolFromSmiles(smi)
        except Exception as e:
            print(e)

    mols = [Chem.AddHs(m) if m != None else None for m in mols]
    print('Converting total {} molecules from SMILES to 3D structures...'.format(nsmiles))
    
    for i, m in enumerate(mols):
        view_bar(i+1, nsmiles)
        if m != None:
            mol_name = f"{i:0>6d}"
            target_smi = smiles_strs[i]
            
            if AllChem.EmbedMolecule(m, randomSeed=0) == 0:
                AllChem.MMFFOptimizeMolecule(m)
                m.SetProp("Name", f'mol_{mol_name}')
                m.SetProp("SMILES", target_smi)
                counter["rdkit"] += 1
                save_file(m, mol_name, xyz_flag)
            else:
                out = pybel_gen_3D(target_smi)

                if out == "":
                    print(f"Error: Failed to use pybel to convert the \'{mol_name}\' molecule to the 3D structure!")
                    mols[i] = None
                    counter["failed"] += 1
                else:
                    try:
                        m = Chem.MolFromMolBlock(out, removeHs=False)
                        m.SetProp("Name", f'mol_{mol_name}')
                        m.SetProp("SMILES", target_smi)
                        counter["pybel"] += 1
                        mols[i] = m
                        save_file(m, mol_name, xyz_flag)    
                    except Exception as e:
                        print(e)
                        print(f"Error: Failed to use rdkit to convert the \'{mol_name}\' molecule to the 3D structure!")
                        mols[i] = None
                        counter["failed"] += 1

    is_valid = np.array([(m != None) for m in mols], dtype=bool)
    mols = np.array(mols)[is_valid]
    print(f"\nInfo: rdkit -> {counter['rdkit']}   pybel -> {counter['pybel']}    failed -> {counter['failed']}")
    print('Info: The target 3D structures are saved in the following path:')
    print('      {0}\n'.format(os.getcwd() + os.sep + 'mols.sdf'))
    end_time = time.time()
    total_running_time(end_time, start_time)

    return mols

def pybel_gen_3D(input_str):
    mol = pybel.readstring("smi", input_str)
    mol.addh()
    if mol.make3D() == None:
        output_str = mol.write("mol")
    else:
        output_str = ""

    return output_str

def save_file(mol, mol_name, xyz_flag, sdf_writer = Chem.SDWriter('mols.sdf')):
    sdf_writer.write(mol)
    #Draw.MolToFile(mol, f'mol_{mol_name}.png', size=(150, 150))

    if xyz_flag:
        Chem.MolToXYZFile(mol, f'mol_{mol_name}.xyz')
        with open('mols.xyz','a+') as f:
            f.write(Chem.MolToXYZBlock(mol))

def view_bar(current, total):
    p = int(100*current/total)
    a = "#"*int(p/2)
    b = "-"*(50-int(p/2))
    print("\r[{:s}{:s}] {:5.2%} ({:6d} / {:6d})".format(a,b, current/total, current, total),end='')

def total_running_time(end_time, start_time):
    tot_seconds = round(end_time - start_time,2)
    days = tot_seconds // 86400
    hours = (tot_seconds % 86400) // 3600
    minutes = (tot_seconds % 86400 % 3600)// 60
    seconds = tot_seconds % 60
    print(">> Elapsed time: {0:2d} day(s) {1:2d} hour(s) {2:2d} minute(s) {3:5.2f} second(s) <<".format(int(days),int(hours),int(minutes),seconds))


if __name__ == '__main__':
    xyz_flag = False   # output *.xyz files

    df = pd.read_csv('../MolLogP_dataset.csv')
    smiles_list = df["Smiles"]

    #smiles_list = ['C1CC2C=CC1OO2','N#Cc1c(c(c(c(c1n1c2ccc(cc2c2c1ccc(c2)c1ccccc1)c1ccccc1)n1c2ccc(cc2c2c1ccc(c2)c1ccccc1)c1ccccc1)C#N)n1c2ccc(cc2c2c1ccc(c2)c1ccccc1)c1ccccc1)n1c2ccc(cc2c2c1ccc(c2)c1ccccc1)c1ccccc1','c1ccccc1']

    
    get_3D_mols(smiles_list, xyz_flag)
    
    






