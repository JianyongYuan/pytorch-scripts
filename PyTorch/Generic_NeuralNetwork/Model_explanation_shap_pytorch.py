#!/usr/bin/env python3
'''conda create -n pytorch-env python=3.9 shap pandas optuna=2.10.1 xgboost scikit-learn sklearn-pandas rdkit pytorch torchvision torchaudio pytorch-cuda=11.6 cairosvg dgllife dgl=0.9.1 dgl-cuda11.6 ipython -c pytorch -c nvidia -c dglteam'''
import os,time
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib
import matplotlib.pyplot as plt


'''input basic settings'''
num_out_labels = 2 # >=2: the number of y labels for classification tasks; 1: for regression tasks
scaler_flag = True
filename = "cla"   # the filename.pkl file in the current working directory to be load as target models
filename_scaler = "clas"   # the filename.pkl file in the current working directory to be load as the scaler
classification_label = 1 # None for regression tasks; 0, 1, 2, etc for classification tasks
plot_type = "dependence" # "importance", "summary", "force", "waterfall", and "dependence"
show_fig = True  # whether to show figures after plotting results
sample_index = 1  # only for the force and waterfall plots with the specific sample, "0", "1", "2", etc.
feature_dependence = ['AromaticProportion','NumRotatableBonds']  # only for the dependence plot with specific two features, e.g.:['DS','R1502']
max_display = 10  # set the number of features to be displayed
model_type = "classification" if num_out_labels > 1 else "regression"


'''Define a neural network model'''
class NeuralNetwork(nn.Module):
    def __init__(self, num_features, num_out_labels):
        super(NeuralNetwork, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(num_features, 50),
            nn.BatchNorm1d(50),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.BatchNorm1d(50),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.BatchNorm1d(50),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(50, num_out_labels)
        )

    def forward(self, x):
        out = self.layer_stack(x)
        return out


'''load dataset'''
selected_features = ['MolWt', 'NumRotatableBonds', 'AromaticProportion']
df = pd.read_csv('../MolLogP_dataset.csv')
data_X = df[selected_features]
data_y = df['MolLogP<2']


'''select data'''
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=0)

# X_background = data_X    # background data samples (e.g. training set)
# X_selected = data_X      # selected data samples to be plotted (e.g. test set)

X_background = X_train.sample(n=5,random_state=0)     # background data samples (e.g. training set)
X_selected = X_test.sample(n=5,random_state=0)        # selected data samples to be plotted (e.g. test set)


# exit()


'''define useful functions'''
def model_predictions(X_scaled):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = data_X.shape[1]
    model = NeuralNetwork(num_features, num_out_labels).to(device)
    checkpoint = torch.load(filename + ".pkl", map_location = device)
    model.load_state_dict(checkpoint['net'])
    model.eval()   # set model to evaluation mode
    
    if type(X_scaled) == np.ndarray:
        X_scaled_tensor = torch.tensor(X_scaled).float()
    elif type(X_scaled) == pd.DataFrame:
        X_scaled_tensor = torch.tensor(X_scaled.values).float()
    X = X_scaled_tensor.to(device)
    out = model(X)

    if num_out_labels > 1:
        pred_probab = nn.Softmax(dim=1)(out)
        y_pred = pred_probab[:,classification_label].detach()
    elif num_out_labels == 1:
        y_pred = out.detach().squeeze(-1)
    data_y = pd.DataFrame(pd.Series(y_pred.cpu()), columns=["y_pred"])

    return data_y

def gen_shap(X_background, X_selected):
    feature_values = X_selected
    if scaler_flag:
        X_background = scale_transform(X_background)
        X_selected = scale_transform(X_selected)
        print("The scaled feature values of the first five samples:\n{0}\n".format(X_selected.iloc[:5,:]))
    else:
        print("The feature values of the first five samples:\n{0}\n".format(X_selected.iloc[:5,:]))

    f = lambda x: model_predictions(x)
    explainer = shap.KernelExplainer(f, X_background)
    expected_values = explainer.expected_value
    shap_values = explainer.shap_values(X_selected)
    shap_values = pd.DataFrame(shap_values, index=X_selected.index, columns=X_selected.columns)

    return feature_values, shap_values, expected_values

def scale_transform(X):
    scaler = joblib.load(filename_scaler + ".pkl") 
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

    return X_scaled

def total_running_time(end_time, start_time):
    tot_seconds = round(end_time - start_time,2)
    days = tot_seconds // 86400
    hours = (tot_seconds % 86400) // 3600
    minutes = (tot_seconds % 86400 % 3600)// 60
    seconds = tot_seconds % 60
    print("Info: Elapsed time: {0:2d} day(s) {1:2d} hour(s) {2:2d} minute(s) {3:5.2f} second(s)".format(int(days),int(hours),int(minutes),seconds))
    
def input_check():
    if not isinstance(num_out_labels, int):
        print("Error: Please input an integer in the 'num_out_labels' field!")
        exit()
    
    if scaler_flag == True:
        if filename_scaler == "":
            print("Error: Please input the filename of the target scaler in the 'filename_scaler' field!")
            exit()
    
    if model_type == "classification":
        if not isinstance(classification_label, int):
            print("Error: Please input an integer in the 'classification_label' field!")
            exit()

    if plot_type not in ['importance', 'summary', 'force', 'waterfall', 'dependence']:
        print("Error: Please input one of the following plot types:")
        print("       'importance', 'summary', 'force', 'waterfall', or 'dependence'")
        exit()
    elif plot_type in ['force', 'waterfall']:
        if not isinstance(sample_index, int): 
            print("Error: Please input an integer in the 'sample_index' field!")
            exit()
    elif plot_type == "dependence":
        if not isinstance(feature_dependence, list) or len(feature_dependence) != 2: 
            print("Error: Please input a list with two elements in the 'feature_dependence' field!")
            exit()


'''creat explainer and get shap_values and expected_values'''
input_check()
start_time = time.time()


print("Use the KernelExplainer to analyse the target model (Neural Network)...\n")
feature_values, shap_values, expected_values = gen_shap(X_background, X_selected)


'''obtain results and plots'''
matplotlib.use('TkAgg')
save_plot_name = filename + "_" + plot_type + ".png"

if plot_type == "importance":
    feature_importance = pd.DataFrame()
    feature_importance['feature'] = shap_values.columns
    feature_importance['importance'] = feature_importance['feature'].map(np.abs(shap_values).mean(0))
    feature_importance = feature_importance.sort_values('importance', ascending=False)[:max_display]
    print("Feature importance:\n{0}\n".format(feature_importance))
    shap.summary_plot(shap_values.values, feature_values, plot_type="bar", show=False, max_display=max_display)
    for x,y in enumerate(feature_importance.sort_values('importance', ascending=True).values[-max_display:]):
        plt.text(y[1], x-0.1, '%.4f' %round(y[1],4), ha='left')

elif plot_type == "summary":
    shap.summary_plot(shap_values.values, feature_values, show=False, max_display=max_display)

elif plot_type == "force" or plot_type == "waterfall":
    if scaler_flag == True:
        X_selected_scaled = scale_transform(X_selected)
        y_pred = model_predictions(X_selected_scaled)
    else:
        y_pred = model_predictions(X_selected)

    print("Target sample:\n{0}\n".format(feature_values.iloc[sample_index,:]))
    print("Expected [base] value:   {:.4f}".format(expected_values))
    print("Predicted [f(x)] value:  {:.4f}\n".format(y_pred.iloc[sample_index,0]))
    if plot_type == "force":
        shap.force_plot(expected_values, shap_values.values[sample_index,:], feature_values.iloc[sample_index,:], matplotlib=True, show=False)
    elif plot_type == "waterfall":
        shap.waterfall_plot(shap.Explanation(values=shap_values.values[sample_index,:], base_values=expected_values, data=feature_values.iloc[sample_index,:]), max_display=max_display, show=False)

elif plot_type == "dependence":
    shap.dependence_plot(feature_dependence[0], shap_values.values, feature_values, interaction_index=feature_dependence[1], show=False)


'''output target figures'''
end_time = time.time()
if show_fig != None:
    plt.savefig(save_plot_name, bbox_inches='tight', dpi=300)
    print('Info: The target graph \'{}\' is saved in the following path:'.format(save_plot_name))
    print('      {0}\n'.format(os.getcwd() + os.sep + save_plot_name))
    total_running_time(end_time, start_time)
    
    if show_fig == True:
        plt.show()



