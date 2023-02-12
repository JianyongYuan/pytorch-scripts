#!/usr/bin/env python3
'''conda create -n pytorch-env python=3.9 shap pandas optuna=2.10.1 xgboost scikit-learn sklearn-pandas rdkit pytorch torchvision torchaudio pytorch-cuda=11.6 cairosvg dgllife dgl=0.9.1 dgl-cuda11.6 ipython -c pytorch -c nvidia -c dglteam'''
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime, time, joblib


'''set basic parameters'''
filename_pkl = 'pytorch_regression_best'  # load the target model from the *.pkl file
filename_scaler = 'pytorch_regression_scaler' # load the scaler from the *.pkl file; empty for None
split_dataset = False  # whether to split the dataset into training and test sets
num_out_labels = 1  # >=2: the number of y labels for classification tasks; 1: for regression tasks


'''input data'''
selected_features = ['MolWt', 'NumRotatableBonds', 'AromaticProportion']
df = pd.read_csv('../MolLogP_dataset.csv')
data_X = df[selected_features]
data_y = df['MolLogP']


'''define a neural network model'''
class NeuralNetwork(nn.Module):
    def __init__(self, num_features, num_out_labels):
        super(NeuralNetwork, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(num_features, 50),
            nn.BatchNorm1d(50),
            nn.Dropout(0.27),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.BatchNorm1d(50),
            nn.Dropout(0.27),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.BatchNorm1d(50),
            nn.Dropout(0.27),
            nn.ReLU(),
            nn.Linear(50, num_out_labels)
        )

    def forward(self, x):
        out = self.layer_stack(x)
        return out

'''transform from DataFrame/Series data to tensors'''
class NN_Dataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series, num_out_labels: int):
        super(NN_Dataset, self).__init__()
        self.num_out_labels = num_out_labels
        self.X = torch.tensor(X.values).float()
        if self.num_out_labels >= 2:
            self.y = torch.tensor(y.values).long()
        else:
            self.y = torch.tensor(y.values).float().reshape(-1,1)
        
    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return self.X.shape[0]

'''scale data'''
def scaler(filename_scaler, X_test):  
    scale_flag = True if filename_scaler else False
    
    if scale_flag:
        scaler = joblib.load(filename_scaler + ".pkl") 
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=data_X.columns)
    else:
        X_test_scaled = X_test

    return X_test_scaled

'''define other useful functions'''
def evaluation_classification(test_loader, model):
    test_size = len(test_loader.dataset)
    model.eval()   # set model to evaluation mode

    with torch.no_grad():
        for (X, y) in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
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
        for (X, y) in test_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            mse_test = mean_squared_error(y.cpu(), y_pred.cpu())
            mae_test = mean_absolute_error(y.cpu(), y_pred.cpu())
            r2_test = r2_score(y.cpu(), y_pred.cpu())

    print("            >>>>  Metrics based on the target model  <<<<\n")
    print(f"> Test set size: {test_size}")
    print(f"> Mean squared error (MSE) on the test set: {mse_test:.6f}")
    print(f"> Mean absolute error (MAE) on the test set: {mae_test:.6f}")
    print(f"> R-squared (R^2) value on the test set: {r2_test:.6f}")

def load_model(filename_pkl):
    model = NeuralNetwork(num_features, num_out_labels).to(device)
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

def test_main(X_test_scaled, y_test):
    test_dataset = NN_Dataset(X=X_test_scaled, y=y_test, num_out_labels=num_out_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=X_test_scaled.shape[0])
    target_model = load_model(filename_pkl)

    if num_out_labels >= 2: 
        evaluation_classification(test_dataloader, target_model)
    else:
        evaluation_regression(test_dataloader, target_model)

###############  The neural network training for classification tasks starts from here  ###############
start_time = time.time()
start_date = datetime.datetime.now()
print('      ***  PyTorch for evaluating the generic neural network model started at {0}  ***\n'.format(start_date.strftime("%Y-%m-%d %H:%M:%S")))
if torch.cuda.is_available():
    print("CUDA is detected, and thus the task is performed with the GPU acceleration...\n")

'''split training/test sets'''
if split_dataset:
    print('The dataset is splited into training and test sets, and therefore the target model will be evaluated on the test set...\n')
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=0)
else:
    print('The whole dataset will be used to evaluate the target model...\n')
    X_test, y_test = data_X, data_y
X_test_scaled = scaler(filename_scaler, X_test)


'''choose "cpu" or "gpu" and instantiated the neural network model'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_features = X_test_scaled.shape[1]
model = NeuralNetwork(num_features, num_out_labels).to(device)

'''test process'''
test_main(X_test_scaled, y_test)


end_time = time.time()
end_date = datetime.datetime.now()
print('\n      ***  PyTorch for evaluating the generic neural network model terminated at {0}  ***\n'.format(end_date.strftime("%Y-%m-%d %H:%M:%S")))
if torch.cuda.is_available():
    print("          >>>  Thanks to the GPU acceleration with {} <<<\n".format(torch.cuda.get_device_properties(device).name))
total_running_time(end_time, start_time)


