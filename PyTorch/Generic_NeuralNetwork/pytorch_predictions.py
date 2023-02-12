#!/usr/bin/env python3
'''conda create -n pytorch-env python=3.9 shap pandas optuna=2.10.1 xgboost scikit-learn sklearn-pandas rdkit pytorch torchvision torchaudio pytorch-cuda=11.6 cairosvg dgllife dgl=0.9.1 dgl-cuda11.6 ipython -c pytorch -c nvidia -c dglteam'''
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os, datetime, time, joblib


###############  Set required parameters here  ###############
input_filename = [  'pytorch_regression_best'
                ]  # the filename.pkl file in the current working directory to be load as target models
output_filename = 'prediction_results_5' # the results are saved to the *.csv file
scaler_filename = 'pytorch_regression_scaler' # the filename.pkl file in the current working directory to be load as the scaler
sorting_choice = 'descending' # "ascending" or "descending"
num_out_labels = 1 # >=2: the number of y labels for classification tasks; 1: for regression tasks
scaler_flag = True


'''load datasets'''
# selected_features = ['MW', 'dipole', 'SASA', 'FOSA', 'FISA', 'PISA', 'WPSA', 'volume', 'HBA', 'IP_HOMO', 'EA_LUMO', 'dip', 'ACxDN', 'glob', 'QPlogP', 'QPlogS', 'PSA', 'h_pKb', 'in56', 'QPpolrz', 'QPlogPC16', 'QPlogPoct', 'QPlogPw', 'CIQPlogS', 'QPlogHERG', 'QPPCaco', 'QPlogBB', 'QPPMDCK', 'QPlogKp', 'QPlogKhsa', 'PercentHumanOralAbsorption']
selected_features = ['MW', 'FOSA', 'PISA', 'IP_HOMO', 'EA_LUMO', 'dip', 'ACxDN', 'glob', 'QPlogS', 'PSA', 'h_pKa', 'h_pKb', 'QPpolrz', 'QPlogPC16', 'QPlogPoct', 'QPlogPw', 'QPlogHERG', 'QPPCaco', 'QPPMDCK', 'QPlogKhsa']
df = pd.read_csv('../dataset.csv')
data_X = df[selected_features]


# mask_na = df[selected_features].isnull().any(axis=1)
# print(mask_na)
# data_X = df[selected_features].loc[mask_na]
# print(data_X)
#data_X = df[selected_features].dropna(inplace=False)
# print(data_X)
# exit()

'''set target Y ranges; 'all' for all ranges'''
# y_ranges = "(data_X_y['Hardness_EN_best'] > 60.5)&(data_X_y['Hardness_EN_best'] < 61)&(data_X_y['MA300_EN_best'] > 1)"
y_ranges = "all"


'''define a neural network model'''
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

        # # 多层网络初始化
        # for m in self.modules():
        #      if isinstance(m, nn.Conv2d):
        #          nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #          nn.init.constant_(m.weight, 1)
        #          nn.init.constant_(m.bias, 0)


    def forward(self, x):
        out = self.layer_stack(x)
        return out


###############  Some user-defined functions and variables  ###############
def total_running_time(end_time, start_time):
    tot_seconds = round(end_time - start_time, 2)
    days = tot_seconds // 86400
    hours = (tot_seconds % 86400) // 3600
    minutes = (tot_seconds % 86400 % 3600)// 60
    seconds = tot_seconds % 60
    print(">> Elapsed time: {0:2d} day(s) {1:2d} hour(s) {2:2d} minute(s) {3:5.2f} second(s) <<".format(int(days),int(hours),int(minutes),seconds))

def model_predictions(X_pred_scaled, input_filename, sorting_choice):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = data_X.shape[1]
    sorting = True if sorting_choice == 'ascending' else False
    pkl_file = input_filename + ".pkl"
    model = NeuralNetwork(num_features, num_out_labels).to(device)
    checkpoint = torch.load(pkl_file, map_location = device)
    model.load_state_dict(checkpoint['net'])
    model.eval()   # set model to evaluation mode
    X = torch.tensor(X_pred_scaled.values).float().to(device)
    out = model(X)

    if num_out_labels > 1:
        pred_probab = nn.Softmax(dim=1)(out)
        y_pred = pred_probab.argmax(1)
    elif num_out_labels == 1:
        y_pred = out.detach().squeeze(-1)

    data_y = pd.DataFrame(pd.Series(y_pred.cpu()), columns=[input_filename])
    data_X_y = pd.concat([data_X, data_y], axis=1)
    data_X_y_sorted = data_X_y.sort_values(input_filename,ascending=sorting,ignore_index=True)

    print('---------- Information of the {0} model ----------'.format(input_filename))
    print('> Parameters of the {0} model:\n {1}\n'.format(input_filename, model.state_dict()))
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
    data_X_y = pd.concat([df['Smiles'], data_X_y], axis=1)  #增加smiles

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

y_preds = []
for i in input_filename:
    if scaler_flag:
        scaler = joblib.load(scaler_filename + ".pkl") 
        data_X_scaled = pd.DataFrame(scaler.transform(data_X), columns=data_X.columns)
        y_preds.append(model_predictions(data_X_scaled, i, sorting_choice)[:,np.newaxis])
    else:
        y_preds.append(model_predictions(data_X, i, sorting_choice)[:,np.newaxis])

y_preds = pd.DataFrame(np.hstack(y_preds), columns=input_filename)
get_results(y_preds, data_X, input_filename, sorting_choice, y_ranges)

end_time = time.time()
end_date = datetime.datetime.now()
print('***  PyTorch predictions (in target Y ranges) terminated at {0}  ***\n'.format(end_date.strftime("%Y-%m-%d %H:%M:%S")))
total_running_time(end_time, start_time)





