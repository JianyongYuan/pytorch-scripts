#!/usr/bin/env python3
'''conda create -n pytorch-env python=3.9 shap pandas optuna=2.10.1 xgboost scikit-learn sklearn-pandas rdkit pytorch torchvision torchaudio pytorch-cuda=11.6 cairosvg dgllife dgl=0.9.1 dgl-cuda11.6 ipython -c pytorch -c nvidia -c dglteam'''
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score
import datetime, time, sys, os, random, joblib


'''set required (hyper)parameters'''
print_details = True
restart = False  # resume the training process from the *.pkl file
early_stopping_patience = 10
weight_decay = 0  # L2 regularization weight
learning_rate = 1e-3
epoches = 5   # number of iterations
batch_size = 32   # for the dataloader module
scale_method = "MinMaxScaler"  # "MinMaxScaler", "StandardScaler", or empty for None
cv_fold = 3   # if cv_fold in [0, 0.5], then the validation set is FIXED with the size of one-n th of the original training set (n = int(1/cv_fold))
seed_number = 0  # set a seed number


'''input data'''
selected_features = ['MolWt', 'NumRotatableBonds', 'AromaticProportion']
df = pd.read_csv('../MolLogP_dataset.csv')
data_X = df[selected_features]
data_y = df['MolLogP<2']


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

'''transform from DataFrame/Series data to tensors'''
class NN_Dataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        super(NN_Dataset, self).__init__()
        self.X = torch.tensor(X.values).float()
        self.y = torch.tensor(y.values).long()
        
    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return self.X.shape[0]

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

'''scale data'''
def scaler(method, X_train, X_test):
    if method == "MinMaxScaler":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scale_flag = True
    elif method == "StandardScaler":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scale_flag = True
    else:
        scale_flag = False
    
    if scale_flag:
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        filename = sys.argv[0].split(os.sep)[-1].split(".")[0]
        file_pkl = filename + "_scaler.pkl"
        joblib.dump(scaler, file_pkl)
    else:
        X_train_scaled, X_test_scaled = X_train, X_test

    return X_train_scaled, X_test_scaled

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

    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        out = model(X)
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
    loss_tot, correct = 0, 0
    valid_size = len(valid_loader.dataset)
    model.eval()   # set model to evaluation mode

    with torch.no_grad():
        for (X, y) in valid_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
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
        for (X, y) in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = loss_fn(out, y)
            loss_tot_test = loss.item()
            pred_probab = nn.Softmax(dim=1)(out)
            y_test_pred_proba = pred_probab.cpu().numpy()
            y_test_pred = pred_probab.argmax(1)
            y_test = y
            accuracy_test = accuracy_score(y_test.cpu(), y_test_pred.cpu())
        
        for (X, y) in train_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
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
    model = NeuralNetwork(num_features, num_out_labels).to(device)
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

def train_main(model, optimizer, X_train_scaled, y_train, loss_fn):
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

        for i, (train_index, valid_index) in enumerate(kf.split(X_train_scaled, y_train)):
            X_train_cv, X_valid_cv = X_train_scaled.iloc[list(train_index)], X_train_scaled.iloc[list(valid_index)]
            y_train_cv, y_valid_cv = y_train.iloc[list(train_index)], y_train.iloc[list(valid_index)]

            train_dataset_cv = NN_Dataset(X=X_train_cv, y=y_train_cv)
            vaild_dataset_cv = NN_Dataset(X=X_valid_cv, y=y_valid_cv)
            train_dataloader = DataLoader(train_dataset_cv, batch_size=batch_size)
            valid_dataloader = DataLoader(vaild_dataset_cv, batch_size=batch_size)

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

def test_main(X_train_scaled, X_test_scaled, y_train, y_test, loss_fn, best_epoch):
    print(f"           @@ Model Evaluation For The Best Epoch ({best_epoch}) @@\n")
    train_dataset = NN_Dataset(X=X_train_scaled, y=y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=X_train_scaled.shape[0])
    test_dataset = NN_Dataset(X=X_test_scaled, y=y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=X_test_scaled.shape[0])

    best_model, optimizer, schedular, checkpoint = load_best_model()

    evaluation(test_dataloader, train_dataloader, best_model, checkpoint, loss_fn)

###############  The neural network training for classification tasks starts from here  ###############
start_time = time.time()
start_date = datetime.datetime.now()
set_seed(seed=seed_number)
print('      ***  PyTorch for classification tasks started at {0}  ***\n'.format(start_date.strftime("%Y-%m-%d %H:%M:%S")))
if torch.cuda.is_available():
    print("CUDA is detected, and thus the task is performed with the GPU acceleration...\n")
cv_flag = True

'''split training/test sets'''
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=0)
X_train_scaled, X_test_scaled = scaler(scale_method, X_train, X_test)

'''choose "cpu" or "gpu" and instantiated the neural network model'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_features = X_train.shape[1]
num_out_labels = len(set(data_y))
model = NeuralNetwork(num_features, num_out_labels).to(device)
model.apply(weights_init)

'''choose a loss function and an optimizer'''
weight_CE = class_weight(y_train).to(device)
loss_fn = CrossEntropyLoss(weight=weight_CE, reduction='mean').to(device)
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
schedular = ReduceLROnPlateau(optimizer)

if restart:
    model, optimizer, schedular, checkpoint = load_best_model()

'''training process'''
best_epoch, lowest_loss = train_main(model, optimizer, X_train_scaled, y_train, loss_fn)

'''test process'''
test_main(X_train_scaled, X_test_scaled, y_train, y_test, loss_fn, best_epoch)


end_time = time.time()
end_date = datetime.datetime.now()
print('\n      ***  PyTorch for classification tasks terminated at {0}  ***\n'.format(end_date.strftime("%Y-%m-%d %H:%M:%S")))
if torch.cuda.is_available():
    print("          >>>  Thanks to the GPU acceleration with {} <<<\n".format(torch.cuda.get_device_properties(device).name))
total_running_time(end_time, start_time)


