from codes.utils import accuracy_regression, accuracy_nnrank, ordinal_loss

from codes.utils import pred2class, accuracy_classification, f1score_cnn_fn, data_loader_reg, data_loader_test_reg, transform, transform_test, data_loader_mlp_reg, data_loader_nnrak, data_loader_mlp


import random
import numpy as np
import os
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, models, transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.optim as optim
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler


from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import RadiusNeighborsRegressor as RNR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import GradientBoostingRegressor as GBR





def svr_gridsearch(train_X, train_Y, val_X, val_Y, Kernel, Degree, Gamma, Coef0, Ce,Epsilon,Shrinking,fold, model_path):
    """support vector regressor grid hyperparameters gridsearch"""
    
    variables_best = {}
    performance_best = {}
    accuracy_max = 0
    for kernel in Kernel:
        for degree in Degree:
            for gamma in Gamma:
                for coef0 in Coef0:
                    for C in Ce:
                        for epsilon in Epsilon:
                            for shrinking in Shrinking:

                                S_V_R = SVR(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, 
                                            C=C, epsilon=epsilon, shrinking=shrinking)
       
                                S_V_R.fit(train_X, train_Y)

                                pred_val_Y = S_V_R.predict(val_X)
                                pred_val_Y = pred2class(pred_val_Y)

                                accuracy_val = accuracy_score(val_Y, pred_val_Y)
                                cm_val = confusion_matrix(val_Y, pred_val_Y)

                                pred_train_Y = S_V_R.predict(train_X)
                                pred_train_Y = pred2class(pred_train_Y)

                                accuracy_train = accuracy_score(train_Y, pred_train_Y)
                                cm_train = confusion_matrix(train_Y, pred_train_Y)

                                if accuracy_val > accuracy_max:
                                    print(f'best_train_acc: {accuracy_train}, best_val_acc: {accuracy_val}')
                                    accuracy_max = accuracy_val
                                    variables_best['C'] = C
                                    variables_best['kernel'] = kernel
                                    variables_best['degree'] = degree
                                    variables_best['gamma'] = gamma
                                    variables_best['coef0'] = coef0
                                    variables_best['epsilon'] = epsilon
                                    variables_best['shrinking'] = shrinking

                                    performance_best['accuracy_train'] = accuracy_train
                                    performance_best['accuracy_val'] = accuracy_val
                                    performance_best['cm_train'] = cm_train
                                    performance_best['cm_val'] = cm_val

                                    #filename = f'Model/Reg/TrainedImageNet/Up/SVR_{fold}_model.sav'
                                    
                                    PATH = os.path.join('Models', model_path)
                                    pickle.dump(S_V_R, open(PATH, 'wb'))
                            
    return variables_best, performance_best





def krr_gridsearch(train_X, train_Y, val_X, val_Y, Alpha, Gamma, Degree, fold, model_path):
    """kernel ridge regression grid hyperparameters gridsearch"""
    
    variables_best = {}
    performance_best = {}
    accuracy_max = 0
    for alpha in Alpha:
        for gamma in Gamma:
            
            for degree in Degree:
            
                K_R_R = KernelRidge(alpha=alpha, gamma=gamma, kernel='polynomial',degree=degree)



                K_R_R.fit(train_X, train_Y)

                pred_val_Y = K_R_R.predict(val_X)
                pred_val_Y = pred2class(pred_val_Y)

                accuracy_val = accuracy_score(val_Y, pred_val_Y)
                cm_val = confusion_matrix(val_Y, pred_val_Y)

                pred_train_Y = K_R_R.predict(train_X)
                pred_train_Y = pred2class(pred_train_Y)

                accuracy_train = accuracy_score(train_Y, pred_train_Y)
                cm_train = confusion_matrix(train_Y, pred_train_Y)

                if accuracy_val > accuracy_max:
                    print(f'best_train_acc: {accuracy_train}, best_val_acc: {accuracy_val}')
                    accuracy_max = accuracy_val
                    variables_best['alpha'] = alpha
                    variables_best['gamma'] = gamma
                    variables_best['degree'] = degree

                    performance_best['accuracy_train'] = accuracy_train
                    performance_best['accuracy_val'] = accuracy_val
                    performance_best['cm_train'] = cm_train
                    performance_best['cm_val'] = cm_val

                    #filename = f'Model/Reg/TrainedImageNet/Up/KRR_{fold}_model.sav'
                    PATH = os.path.join('Models', model_path)
                    pickle.dump(K_R_R, open(PATH, 'wb'))
                            
    return variables_best, performance_best




def rnr_gridsearch(train_X, train_Y, val_X, val_Y, Radius, Weights, Algorithm, P, fold, model_path):
    """radius neighbors regression grid hyperparameters gridsearch"""
    
    variables_best = {}
    performance_best = {}
    accuracy_max = 0
    for radius in Radius:
        for weights in Weights:
            for algorithm in Algorithm:
                for p in P:
                    R_N_N = RNR(radius=radius, weights=weights, algorithm=algorithm, p=p)



                    R_N_N.fit(train_X, train_Y)

                    pred_val_Y = R_N_N.predict(val_X)
                    #print(pred_val_Y)
                    pred_val_Y = pred2class(pred_val_Y)
                    #print(len(pred_val_Y))
                    accuracy_val = accuracy_score(val_Y, pred_val_Y)
                    cm_val = confusion_matrix(val_Y, pred_val_Y)

                    pred_train_Y = R_N_N.predict(train_X)
                    pred_train_Y = pred2class(pred_train_Y)
                    accuracy_train = accuracy_score(train_Y, pred_train_Y)
                    cm_train = confusion_matrix(train_Y, pred_train_Y)

                    if accuracy_val > accuracy_max:
                        print(f'best_train_acc: {accuracy_train}, best_val_acc: {accuracy_val}')
                        accuracy_max = accuracy_val
                        variables_best['radius'] = radius
                        variables_best['weights'] = weights
                        variables_best['algorithm'] = algorithm
                        variables_best['p'] = p

                        performance_best['accuracy_train'] = accuracy_train
                        performance_best['accuracy_val'] = accuracy_val
                        performance_best['cm_train'] = cm_train
                        performance_best['cm_val'] = cm_val

                        #filename = f'Model/Reg/TrainedImageNet/Up/RNN_{fold}_model.sav'
                        
                        PATH = os.path.join('Models', model_path)
                        pickle.dump(R_N_N, open(PATH, 'wb'))
                            
    return variables_best, performance_best






def gpr_gridsearch(train_X, train_Y, val_X, val_Y,Alpha, N_restarts_optimizer, Normalize_y, fold, model_path):
    """Gaussian process regression grid hyperparameters gridsearch"""
    
    variables_best = {}
    performance_best = {}
    accuracy_max = 0
    for alpha in Alpha:
        for n_restarts_optimizer in N_restarts_optimizer:
            for normalize_y in Normalize_y:
                G_P_C = GaussianProcessRegressor(alpha=alpha,n_restarts_optimizer=n_restarts_optimizer,normalize_y=normalize_y, random_state=1)

                G_P_C.fit(train_X, train_Y)

                pred_val_Y = G_P_C.predict(val_X)
                pred_val_Y = pred2class(pred_val_Y)
                accuracy_val = accuracy_score(val_Y, pred_val_Y)
                cm_val = confusion_matrix(val_Y, pred_val_Y)

                pred_train_Y = G_P_C.predict(train_X)
                pred_train_Y = pred2class(pred_train_Y)
                accuracy_train = accuracy_score(train_Y, pred_train_Y)
                cm_train = confusion_matrix(train_Y, pred_train_Y)

                if accuracy_val > accuracy_max:
                    print(f'best_train_acc: {accuracy_train}, best_val_acc: {accuracy_val}')
                    accuracy_max = accuracy_val
                    variables_best['n_restarts_optimizer'] = n_restarts_optimizer
                    variables_best['normalize_y'] = normalize_y
    
                    performance_best['accuracy_train'] = accuracy_train
                    performance_best['accuracy_val'] = accuracy_val
                    performance_best['cm_train'] = cm_train
                    performance_best['cm_val'] = cm_val

                    #filename = f'Model/Reg/TrainedImageNet/Up/GPR_{fold}_model.sav'
                    PATH = os.path.join('Models', model_path)
                    pickle.dump(G_P_C, open(PATH, 'wb'))
                            
    return variables_best, performance_best





def knr_gridsearch(train_X, train_Y, val_X, val_Y, N_neighbors, Weights, Algorithm, P, fold, model_path):
    """k-nearest-neighbors regression grid hyperparameters gridsearch"""
    
    variables_best = {}
    performance_best = {}
    accuracy_max = 0
    for n_neighbors in N_neighbors:
        for weights in Weights:
            for algorithm in Algorithm:
                for p in P:
                    K_N_N = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, p=p)



                    K_N_N.fit(train_X, train_Y)

                    pred_val_Y = K_N_N.predict(val_X)
                    rmse_val = np.sqrt(mean_squared_error(val_Y, pred_val_Y))
                    pred_val_Y = pred2class(pred_val_Y)
                    accuracy_val = accuracy_score(val_Y, pred_val_Y)
                    cm_val = confusion_matrix(val_Y, pred_val_Y)

                    pred_train_Y = K_N_N.predict(train_X)
                    pred_train_Y = pred2class(pred_train_Y)
                    accuracy_train = accuracy_score(train_Y, pred_train_Y)
                    cm_train = confusion_matrix(train_Y, pred_train_Y)

                    if accuracy_val > accuracy_max:
                        print(f'best_train_acc: {accuracy_train :.3f}, best_val_acc: {accuracy_val :.3f}')
                        accuracy_max = accuracy_val
                        variables_best['n_neighbors'] = n_neighbors
                        variables_best['weights'] = weights
                        variables_best['algorithm'] = algorithm
                        variables_best['p'] = p

                        performance_best['accuracy_train'] = accuracy_train
                        performance_best['accuracy_val'] = accuracy_val
                        performance_best['rmse_val'] = rmse_val
                        performance_best['cm_train'] = cm_train
                        performance_best['cm_val'] = cm_val

                        #filename = f'Model/Reg/TrainedImageNet/Up/KNN_{fold}_model.sav'
                        PATH = os.path.join('Models', model_path)
                        pickle.dump(K_N_N, open(PATH, 'wb'))
                            
    return variables_best, performance_best



def dtr_gridsearch(train_X, train_Y, val_X, val_Y, Criterion,Max_depth, Min_samples_split, Max_features, fold, model_path):
    """decision tree regression grid hyperparameters gridsearch"""
    
    variables_best = {}
    performance_best = {}
    accuracy_max = 0
    accuracy_train_max = 0
    for criterion in Criterion:
        for max_depth in Max_depth:
            for min_samples_split in Min_samples_split:
                for max_features in Max_features:
                    D_T_C = DTR(random_state =1, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                                              max_features=max_features)



                    D_T_C.fit(train_X, train_Y)

                    pred_val_Y = D_T_C.predict(val_X)
                    pred_val_Y = pred2class(pred_val_Y)
                    accuracy_val = accuracy_score(val_Y, pred_val_Y)
                    cm_val = confusion_matrix(val_Y, pred_val_Y)

                    pred_train_Y = D_T_C.predict(train_X)
                    pred_train_Y = pred2class(pred_train_Y)
                    accuracy_train = accuracy_score(train_Y, pred_train_Y)
                    cm_train = confusion_matrix(train_Y, pred_train_Y)

                    if accuracy_val > accuracy_max and accuracy_train >= accuracy_train_max:
                        print(f'best_train_acc: {accuracy_train}, best_val_acc: {accuracy_val}')
                        accuracy_max = accuracy_val
                        accuracy_train_max = accuracy_train
                        variables_best['criterion'] = criterion
                        variables_best['max_depth'] = max_depth
                        variables_best['min_samples_split'] = min_samples_split
                        variables_best['max_features'] = max_features

                        performance_best['accuracy_train'] = accuracy_train
                        performance_best['accuracy_val'] = accuracy_val
                        performance_best['cm_train'] = cm_train
                        performance_best['cm_val'] = cm_val

                        #filename = f'Model/Reg/TrainedImageNet/Up/DTR_{fold}_model.sav'
                        PATH = os.path.join('Models', model_path)
                        pickle.dump(D_T_C, open(PATH, 'wb'))
                            
    return variables_best, performance_best






def gbr_gridsearch(train_X, train_Y, val_X, val_Y, N_estimators, Learning_rate, Min_samples_split,Max_depth,fold, model_path):
    """gradient boost regression grid hyperparameters gridsearch"""
    
    variables_best = {}
    performance_best = {}
    accuracy_max = 0
    for n_estimators  in N_estimators:
        for learning_rate in Learning_rate:
            for min_samples_split in Min_samples_split:
                for max_depth in Max_depth:
                    GBC = GBR(n_estimators=n_estimators,learning_rate=learning_rate,
                                  min_samples_split=min_samples_split,max_depth=max_depth,random_state=1)


                    GBC.fit(train_X, train_Y)

                    pred_val_Y = GBC.predict(val_X)
                    pred_val_Y = pred2class(pred_val_Y)
                    accuracy_val = accuracy_score(val_Y, pred_val_Y)
                    cm_val = confusion_matrix(val_Y, pred_val_Y)

                    pred_train_Y = GBC.predict(train_X)
                    pred_train_Y = pred2class(pred_train_Y)
                    accuracy_train = accuracy_score(train_Y, pred_train_Y)
                    cm_train = confusion_matrix(train_Y, pred_train_Y)

                    if accuracy_val > accuracy_max:
                        print(f'best_train_acc: {accuracy_train}, best_val_acc: {accuracy_val}')
                        accuracy_max = accuracy_val
                        variables_best['n_estimators'] = n_estimators
                        variables_best['learning_rate'] = learning_rate
                        variables_best['min_samples_split'] =min_samples_split
                        variables_best['max_depth'] = max_depth

                        performance_best['accuracy_train'] = accuracy_train
                        performance_best['accuracy_val'] = accuracy_val
                        performance_best['cm_train'] = cm_train
                        performance_best['cm_val'] = cm_val

                        #filename = f'Model/Reg/TrainedImageNet/Up/GBC_{fold}_model.sav'
                        PATH = os.path.join('Models', model_path)
                        pickle.dump(GBC, open(PATH, 'wb'))
                            
    return variables_best, performance_best


class MLP(nn.Module):
    def __init__(self, l1=120, l2=84, p = 0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(100, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 1)
        #self.activ = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def mlp_regression_gridsearch(train_X, train_Y, val_X, val_Y,Learning_rate, L1, L2, Drop_out,Batch_size,fold,model_path, epochs=350):
    """Multilayer perceptron regression grid hyperparameters gridsearch"""
    

    best_val_acc = 0
    best_performance_record = 0
    hyper = {}
    performance_record = {'loss':[], 'val_loss': []}
    for learning_rate in Learning_rate:
        for p in Drop_out:
       
            for batch_size in Batch_size:
                for l1 in L1:
                    for l2 in L2:



                        train_loader = data_loader_mlp_reg(train_X, train_Y, batch_size=batch_size, shuffle=True, drop_last=False)
                        val_loader = data_loader_mlp_reg(val_X, val_Y, batch_size=len(val_Y), shuffle=True, drop_last=False)


                        model = MLP(l1, l2, p).to(device)

                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                        criterion = nn.MSELoss()
                        running_loss_list = []
                        val_running_loss_list = []


                        performance_record = {'loss':[], 'val_loss': []}


                        early_stop_thresh = 20
                        best_loss = 100000000
                        best_epoch = 0

                        for epoch in range(1, epochs):  # loop over the dataset multiple times  3001th
                            running_loss = []
                            val_running_loss = []

                            #model_micro.train()
                            for i, data in enumerate(train_loader, 0):
                                inputs, labels = data
                                inputs, labels = inputs.to(device), labels.to(device)
                                #print(inputs)
                                optimizer.zero_grad()

                                outputs = model(inputs)
                                loss = criterion(outputs, labels)
                                loss.backward()
                                optimizer.step()

                                running_loss.append(loss.item())

                            #model_micro.eval()    
                            with torch.no_grad():
                                for i, data in enumerate(val_loader, 0):

                                    inputs, labels = data
                                    inputs, labels = inputs.to(device), labels.to(device)
                                    outputs = model(inputs)
                                    val_loss = criterion(outputs, labels)
                                    val_running_loss.append(val_loss.item())


                            running_loss_list.append(float(f'{np.mean(running_loss) :.4f}'))
                            val_running_loss_list.append(float(f'{np.mean(val_running_loss):.4f}'))

                            print(f'Epoch{epoch}: loss: {np.mean(running_loss):.4f} val_loss: {np.mean(val_running_loss):.4f}')
                            this_loss = np.mean(val_running_loss)

                            if this_loss < best_loss:
                                best_loss = this_loss
                                best_epoch = epoch
                                #checkpoint(model, "best_model.pth")
                            elif epoch - best_epoch > early_stop_thresh:
                                print("Early stopped training at epoch %d" % epoch)
                                break  # terminate the training loop

                        performance_record['loss'] += running_loss_list

                        performance_record['val_loss'] += val_running_loss_list


                        train_accuracy = accuracy_regression(model, train_loader, 'train')
                        val_accuracy = accuracy_regression(model, val_loader, 'val')

                        if val_accuracy > best_val_acc:
                            best_val_acc = val_accuracy
                            best_train_acc = train_accuracy
                            best_performance_record = performance_record
                            hyper['learning_rate'] = learning_rate
                            hyper['batch_size'] = batch_size
                            hyper['l1'] = l1
                            hyper['l2'] = l2
                            hyper['p'] = p
                            hyper['epoch'] = epoch

                            PATH = os.path.join('Models', model_path)
                            torch.save(model.state_dict(), PATH)




    return best_train_acc, best_val_acc, best_performance_record, hyper


def pretrained_model(drop_out):
    
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    model.fc = nn.Sequential(nn.ReLU(),
                            nn.Dropout(p=drop_out),
                                    nn.Linear(512, 100), #150, 50176
                                    nn.ReLU(),
                                    nn.Dropout(p=drop_out),
                                    nn.Linear(100,1)
                                     )
    model.to(device)
    return model
    

def cnn_regression_gridsearch(train_X, train_Y, val_X, val_Y,Learning_rate, Drop_out, Batch_size,fold,model_path, epochs=350):
    """CNN regression grid hyperparameters gridsearch"""
    
    criterion = nn.MSELoss()
    best_val_acc = 0
    best_train_acc = 0
    best_performance_record = 0
    hyper = {}
    performance_record = {'loss':[], 'val_loss': []}
    for learning_rate in Learning_rate:
        for drop_out in Drop_out:
            for batch_size in Batch_size:
                
                
                train_loader= data_loader_reg(train_X, train_Y, transform, batch_size)
                val_loader = data_loader_test_reg(val_X, val_Y, transform, batch_size=len(val_Y))
                val_loader2 = data_loader_test_reg(val_X, val_Y, transform_test,batch_size=len(val_Y))

                model = pretrained_model(drop_out)
                
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                

                running_loss_list = []
                val_running_loss_list = []


                performance_record = {'loss':[], 'val_loss': []}
                
                early_stop_thresh = 15
                best_loss = 10000000000
                best_epoch = 0
                
                for epoch in range(1, epochs):  # loop over the dataset multiple times  3001th
                    running_loss = []
                    val_running_loss = []

                    #model_micro.train()
                    for i, data in enumerate(train_loader, 0):
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)

                        optimizer.zero_grad()

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        running_loss.append(loss.item())

                    #model_micro.eval()    
                    with torch.no_grad():
                        for i, data in enumerate(val_loader, 0):

                            inputs, labels = data
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = model(inputs)
                            val_loss = criterion(outputs, labels)
                            val_running_loss.append(val_loss.item())


                    running_loss_list.append(float(f'{np.mean(running_loss) :.4f}'))
                    val_running_loss_list.append(float(f'{np.mean(val_running_loss):.4f}'))

                    print(f'Epoch{epoch}: loss: {np.mean(running_loss):.4f} val_loss: {np.mean(val_running_loss):.4f}')

                    this_loss = np.mean(val_running_loss)

                    #if this_loss < best_loss:
                    #    best_loss = this_loss
                    #    best_epoch = epoch
                        #checkpoint(model, "best_model.pth")
                    #elif epoch - best_epoch > early_stop_thresh:
                    #    print("Early stopped training at epoch %d" % epoch)
                    #    break  # terminate the training loop


                performance_record['loss'] += running_loss_list

                performance_record['val_loss'] += val_running_loss_list


                train_accuracy = accuracy_regression(model, train_loader, 'train')
                val_accuracy = accuracy_regression(model, val_loader2, 'val')

                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    best_train_acc = train_accuracy
                    best_performance_record = performance_record
                    hyper['learning_rate'] = learning_rate
                    hyper['batch_size'] = batch_size
                    hyper['drop_out'] = drop_out
                    hyper['epoch'] = epoch
                    
                    PATH = os.path.join('Models', model_path)
                    torch.save(model.state_dict(), PATH)


            
    
    return best_train_acc, best_val_acc, best_performance_record, hyper
    
    
def pretrained_rank(drop_out):
    
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    model.fc = nn.Sequential(nn.ReLU(),
                                 nn.Dropout(p=drop_out),
                                    nn.Linear(512, 100), #150, 50176
                                     nn.ReLU(),
                                     nn.Dropout(p=drop_out),
                                     nn.Linear(100, 3),
                                     nn.Sigmoid()
                                     )
    model.to(device)
    return model    
    

class MLPNNRank(nn.Module):
    def __init__(self, l1=120, l2=84, p = 0.2):
        super(MLPNNRank, self).__init__()
        self.fc1 = nn.Linear(100, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 3)
        self.activ = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.activ(x)    

    
def mlp_nnrank_gridsearch(train_X, train_Y, val_X, val_Y,Learning_rate, L1, L2, Drop_out,Batch_size,fold, model_path, epochs=3500):
    best_val_acc = 0
    best_performance_record = 0
    hyper = {}
    performance_record = {'loss':[], 'val_loss': []}
    for learning_rate in Learning_rate:
        for p in Drop_out:
       
            for batch_size in Batch_size:
                for l1 in L1:
                    for l2 in L2:



                        train_loader = data_loader_mlp_reg(train_X, train_Y, batch_size=batch_size, shuffle=True, drop_last=False)
                        val_loader = data_loader_mlp_reg(val_X, val_Y, batch_size=len(val_Y), shuffle=True, drop_last=False)


                        model = MLPNNRank(l1, l2, p).to(device)

                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                        running_loss_list = []
                        val_running_loss_list = []


                        performance_record = {'loss':[], 'val_loss': []}


                        early_stop_thresh = 20
                        best_loss = 10000
                        best_epoch = 0

                        for epoch in range(1, epochs):  # loop over the dataset multiple times  3001th
                            running_loss = []
                            val_running_loss = []

                            #model_micro.train()
                            for i, data in enumerate(train_loader, 0):
                                inputs, labels = data
                                inputs, labels = inputs.to(device), labels.to(device)
                                #print(inputs)
                                optimizer.zero_grad()

                                outputs = model(inputs)
                                loss = ordinal_loss(outputs, labels)
                                loss.backward()
                                optimizer.step()

                                running_loss.append(loss.item())

                            #model_micro.eval()    
                            with torch.no_grad():
                                for i, data in enumerate(val_loader, 0):

                                    inputs, labels = data
                                    inputs, labels = inputs.to(device), labels.to(device)
                                    outputs = model(inputs)
                                    val_loss = ordinal_loss(outputs, labels)
                                    val_running_loss.append(val_loss.item())


                            running_loss_list.append(float(f'{np.mean(running_loss) :.4f}'))
                            val_running_loss_list.append(float(f'{np.mean(val_running_loss):.4f}'))

                            print(f'Epoch{epoch}: loss: {np.mean(running_loss):.4f} val_loss: {np.mean(val_running_loss):.4f}')
                            this_loss = np.mean(val_running_loss)

                            if this_loss < best_loss:
                                best_loss = this_loss
                                best_epoch = epoch
                                #checkpoint(model, "best_model.pth")
                            elif epoch - best_epoch > early_stop_thresh:
                                print("Early stopped training at epoch %d" % epoch)
                                break  # terminate the training loop

                        performance_record['loss'] += running_loss_list

                        performance_record['val_loss'] += val_running_loss_list


                        train_accuracy = accuracy_nnrank(model, train_loader, 'train')
                        val_accuracy = accuracy_nnrank(model, val_loader, 'val')

                        if val_accuracy > best_val_acc:
                            best_val_acc = val_accuracy
                            best_train_acc = train_accuracy
                            best_performance_record = performance_record
                            hyper['learning_rate'] = learning_rate
                            hyper['batch_size'] = batch_size
                            hyper['l1'] = l1
                            hyper['l2'] = l2
                            hyper['p'] = p
                            hyper['epochs'] = epoch

                            PATH = os.path.join('Models', model_path)
                            torch.save(model.state_dict(), PATH)




    return best_train_acc, best_val_acc, best_performance_record, hyper


    
def cnn_nnrank_gridsearch(train_X, train_Y, val_X, val_Y,Learning_rate, Drop_out, Batch_size,fold,model_path, epochs=350):

    best_val_acc = 0
    best_performance_record = 0
    hyper = {}
    performance_record = {'loss':[], 'val_loss': []}
    for learning_rate in Learning_rate:
        for drop_out in Drop_out:
            for batch_size in Batch_size:
                
                
                train_loader= data_loader_nnrak(train_X, train_Y, transform, batch_size)
                val_loader = data_loader_test_reg(val_X, val_Y, transform, batch_size=len(val_Y))
                val_loader2 = data_loader_test_reg(val_X, val_Y, transform_test,batch_size=len(val_Y))

                model = pretrained_rank(drop_out)
                
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                running_loss_list = []
                val_running_loss_list = []


                performance_record = {'loss':[], 'val_loss': []}

                early_stop_thresh = 15
                best_loss = 10000
                best_epoch = 0
                
                
                for epoch in range(1, epochs):  # loop over the dataset multiple times  3001th
                    running_loss = []
                    val_running_loss = []

                    #model_micro.train()
                    for i, data in enumerate(train_loader, 0):
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)

                        optimizer.zero_grad()

                        outputs = model(inputs)
                        loss = ordinal_loss(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        running_loss.append(loss.item())

                    #model_micro.eval()    
                    with torch.no_grad():
                        for i, data in enumerate(val_loader, 0):

                            inputs, labels = data
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = model(inputs)
                            val_loss = ordinal_loss(outputs, labels)
                            val_running_loss.append(val_loss.item())


                    running_loss_list.append(float(f'{np.mean(running_loss) :.4f}'))
                    val_running_loss_list.append(float(f'{np.mean(val_running_loss):.4f}'))

                    print(f'Epoch{epoch}: loss: {np.mean(running_loss):.4f} val_loss: {np.mean(val_running_loss):.4f}')

                    this_loss = np.mean(val_running_loss)

                    if this_loss < best_loss:
                        best_loss = this_loss
                        best_epoch = epoch
                        #checkpoint(model, "best_model.pth")
                    elif epoch - best_epoch > early_stop_thresh:
                        print("Early stopped training at epoch %d" % epoch)
                        break  # terminate the training loop



                performance_record['loss'] += running_loss_list

                performance_record['val_loss'] += val_running_loss_list


                train_accuracy = accuracy_nnrank(model, train_loader, 'train')
                val_accuracy = accuracy_nnrank(model, val_loader2, 'val')

                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    best_train_acc = train_accuracy
                    best_performance_record = performance_record
                    hyper['learning_rate'] = learning_rate
                    hyper['batch_size'] = batch_size
                    hyper['drop_out'] = drop_out
                    hyper['epochs'] = epoch
                    
                    PATH = os.path.join('Models', model_path)
                    torch.save(model.state_dict(), PATH)


            
    
    return best_train_acc, best_val_acc, best_performance_record, hyper





