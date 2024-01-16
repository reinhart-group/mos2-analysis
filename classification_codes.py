from codes.utils import pred2class, accuracy_classification, f1score_cnn_fn, data_loader_fn, data_loader_test_fn, transform, transform_test, data_loader_mlp


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



from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier



def svc_gridsearch(train_X, train_Y, val_X, val_Y, Ce, Kernel, Degree, Gamma, Coef0, Max_iter, fold, model_path, random_state=1):
    """support vector classifier grid hyperparameters gridsearch"""
    variables_best = {}
    performance_best = {}
    accuracy_max = 0
    for C in Ce:
        for kernel in Kernel:
            for degree in Degree:
                for gamma in Gamma:
                    for coef0 in Coef0:
                        for max_iter in Max_iter:
                            S_V_C = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,    
          coef0=coef0, shrinking=True, probability=False, 
              tol=0.001, cache_size=200, class_weight='balanced', 
              verbose=False,max_iter=max_iter, decision_function_shape='ovr', 
              break_ties=False, random_state=0)



                            S_V_C.fit(train_X, train_Y)

                            pred_val_Y = S_V_C.predict(val_X)
                            accuracy_val = accuracy_score(val_Y, pred_val_Y)
                            cm_val = confusion_matrix(val_Y, pred_val_Y)

                            pred_train_Y = S_V_C.predict(train_X)
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
                                variables_best['max_iter'] = max_iter

                                performance_best['accuracy_train'] = accuracy_train
                                performance_best['accuracy_val'] = accuracy_val
                                performance_best['cm_train'] = cm_train
                                performance_best['cm_val'] = cm_val
                                
                                PATH = os.path.join('Models', model_path)
                                
                                pickle.dump(S_V_C, open(PATH, 'wb'))
                            
    return variables_best, performance_best
       


def krc_gridsearch(train_X, train_Y, val_X, val_Y, Alpha, Max_iter, fold, model_path):
    """kernel ridge classifier grid hyperparameters gridsearch"""
    
    variables_best = {}
    performance_best = {}
    accuracy_max = 0
    for alpha in Alpha:
        for max_iter in Max_iter:
            
            K_R_C = RidgeClassifier(alpha=alpha, max_iter=max_iter, random_state=0)



            K_R_C.fit(train_X, train_Y)

            pred_val_Y = K_R_C.predict(val_X)
            accuracy_val = accuracy_score(val_Y, pred_val_Y)
            cm_val = confusion_matrix(val_Y, pred_val_Y)

            pred_train_Y = K_R_C.predict(train_X)
            accuracy_train = accuracy_score(train_Y, pred_train_Y)
            cm_train = confusion_matrix(train_Y, pred_train_Y)

            if accuracy_val > accuracy_max:
                print(f'best_train_acc: {accuracy_train}, best_val_acc: {accuracy_val}')
                accuracy_max = accuracy_val
                variables_best['alpha'] = alpha
                variables_best['max_iter'] = max_iter

                performance_best['accuracy_train'] = accuracy_train
                performance_best['accuracy_val'] = accuracy_val
                performance_best['cm_train'] = cm_train
                performance_best['cm_val'] = cm_val
                
                PATH = os.path.join('Models', model_path)
                #filename = f'Model/Class/TrainedMicroNet/Up/KRC_{fold}_model.sav'
                pickle.dump(K_R_C, open(PATH, 'wb'))
                            
    return variables_best, performance_best



def rnc_gridsearch(train_X, train_Y, val_X, val_Y, Radius, Weights, Algorithm, P, fold, model_path):
    """radius neighbors classifier grid hyperparameters gridsearch"""
    
    variables_best = {}
    performance_best = {}
    accuracy_max = 0
    for radius in Radius:
        for weights in Weights:
            for algorithm in Algorithm:
                for p in P:
                    R_N_N = RadiusNeighborsClassifier(radius=radius, weights=weights, algorithm=algorithm, p=p, outlier_label=0, random_state=0)



                    R_N_N.fit(train_X, train_Y)

                    pred_val_Y = R_N_N.predict(val_X)
                    accuracy_val = accuracy_score(val_Y, pred_val_Y)
                    cm_val = confusion_matrix(val_Y, pred_val_Y)

                    pred_train_Y = R_N_N.predict(train_X)
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
                        
                        PATH = os.path.join('Models', model_path)
                        #filename = f'Model/Class/TrainedImageNet/Up/RNN_{fold}_model.sav'
                        pickle.dump(R_N_N, open(PATH, 'wb'))
                            
    return variables_best, performance_best





def gpc_gridsearch(train_X, train_Y, val_X, val_Y, N_restarts_optimizer, Max_iter_predict, Multi_class, fold, model_path):
    """Gaussian process classifier grid hyperparameters gridsearch"""
    
    variables_best = {}
    performance_best = {}
    accuracy_max = 0
    for n_restarts_optimizer in N_restarts_optimizer:
        for max_iter_predict in Max_iter_predict:
            for multi_class in Multi_class:
                G_P_C = GaussianProcessClassifier(n_restarts_optimizer=n_restarts_optimizer,max_iter_predict=max_iter_predict,
                                            multi_class=multi_class, random_state=0 )



                G_P_C.fit(train_X, train_Y)

                pred_val_Y = G_P_C.predict(val_X)
                accuracy_val = accuracy_score(val_Y, pred_val_Y)
                cm_val = confusion_matrix(val_Y, pred_val_Y)

                pred_train_Y = G_P_C.predict(train_X)
                accuracy_train = accuracy_score(train_Y, pred_train_Y)
                cm_train = confusion_matrix(train_Y, pred_train_Y)

                if accuracy_val > accuracy_max:
                    print(f'best_train_acc: {accuracy_train}, best_val_acc: {accuracy_val}')
                    accuracy_max = accuracy_val
                    variables_best['n_restarts_optimizer'] = n_restarts_optimizer
                    variables_best['max_iter_predict'] = max_iter_predict
                    variables_best['multi_class'] = multi_class

                    performance_best['accuracy_train'] = accuracy_train
                    performance_best['accuracy_val'] = accuracy_val
                    performance_best['cm_train'] = cm_train
                    performance_best['cm_val'] = cm_val

                    #filename = f'Model/Class/TrainedImageNet/Up/GPC_{fold}_model.sav'
                    PATH = os.path.join('Models', model_path)
                    pickle.dump(G_P_C, open(PATH, 'wb'))
                            
    return variables_best, performance_best



    
    
def knc_gridsearch(train_X, train_Y, val_X, val_Y, N_neighbors, Weights, Algorithm, P, fold, model_path):
    """k-nearest neighbors classifier grid hyperparameters gridsearch"""
    variables_best = {}
    performance_best = {}
    accuracy_max = 0
    for n_neighbors in N_neighbors:
        for weights in Weights:
            for algorithm in Algorithm:
                for p in P:
                    K_N_N = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, p=p, random_state=0)
                    K_N_N.fit(train_X, train_Y)

                    pred_val_Y = K_N_N.predict(val_X)
                    accuracy_val = accuracy_score(val_Y, pred_val_Y)
                    cm_val = confusion_matrix(val_Y, pred_val_Y)
                    pred_train_Y = K_N_N.predict(train_X)
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
                        performance_best['cm_train'] = cm_train
                        performance_best['cm_val'] = cm_val
                       
                        PATH = os.path.join('Models', model_path)

                       #filename = f'Model/Class/TrainedImageNet/Up/KNN_{fold}_model.sav'
                        pickle.dump(K_N_N, open(PATH, 'wb'))
                           
    return variables_best, performance_best






def dtc_gridsearch(train_X, train_Y, val_X, val_Y, Criterion, Min_samples_split, Max_features, fold, model_path):
    """decision tree classifier grid hyperparameters gridsearch"""
    
    variables_best = {}
    performance_best = {}
    accuracy_max = 0
    for criterion in Criterion:
        for min_samples_split in Min_samples_split:
            for max_features in Max_features:
                D_T_C = DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split,
                                              max_features=max_features, random_state=1)



                D_T_C.fit(train_X, train_Y)

                pred_val_Y = D_T_C.predict(val_X)
                accuracy_val = accuracy_score(val_Y, pred_val_Y)
                cm_val = confusion_matrix(val_Y, pred_val_Y)

                pred_train_Y = D_T_C.predict(train_X)
                accuracy_train = accuracy_score(train_Y, pred_train_Y)
                cm_train = confusion_matrix(train_Y, pred_train_Y)

                if accuracy_val > accuracy_max:
                    print(f'best_train_acc: {accuracy_train}, best_val_acc: {accuracy_val}')
                    accuracy_max = accuracy_val
                    variables_best['criterion'] = criterion
                    variables_best['min_samples_split'] = min_samples_split
                    variables_best['max_features'] = max_features

                    performance_best['accuracy_train'] = accuracy_train
                    performance_best['accuracy_val'] = accuracy_val
                    performance_best['cm_train'] = cm_train
                    performance_best['cm_val'] = cm_val

                    #filename = f'Model/Class/TrainedImageNet/Up/DTC_{fold}_model.sav'
                    PATH = os.path.join('Models', model_path)
                    
                    pickle.dump(D_T_C, open(PATH, 'wb'))
                            
    return variables_best, performance_best





def gbc_gridsearch(train_X, train_Y, val_X, val_Y, N_estimators, Learning_rate, Min_samples_split,Max_depth,fold, model_path):
    """gradient boost classifier grid hyperparameters gridsearch"""
    
    variables_best = {}
    performance_best = {}
    accuracy_max = 0
    for n_estimators  in N_estimators:
        for learning_rate in Learning_rate:
            for min_samples_split in Min_samples_split:
                for max_depth in Max_depth:
                    GBC = GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=learning_rate,
                                  min_samples_split=min_samples_split,max_depth=max_depth,random_state=1)


                    GBC.fit(train_X, train_Y)

                    pred_val_Y = GBC.predict(val_X)
                    accuracy_val = accuracy_score(val_Y, pred_val_Y)
                    cm_val = confusion_matrix(val_Y, pred_val_Y)

                    pred_train_Y = GBC.predict(train_X)
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

                        #filename = f'Model/Class/TrainedImageNet/Up/GBC_{fold}_model.sav'
                        PATH = os.path.join('Models', model_path)
                        pickle.dump(GBC, open(PATH, 'wb'))
                            
    return variables_best, performance_best


class MLP(nn.Module):
    def __init__(self, l1=120, l2=84, p = 0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(100, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 3)
        #self.activ = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def mlp_class_gridsearch(train_X, train_Y, val_X, val_Y,Learning_rate, L1, L2, Drop_out,Batch_size, fold, model_path, epochs=350):
    """multilayer perceptron classifier grid hyperparameters gridsearch"""
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    best_performance_record = 0
    hyper = {}
    performance_record = {'loss':[], 'val_loss': []}
    for learning_rate in Learning_rate:
        for p in Drop_out:
            for batch_size in Batch_size:
                for l1 in L1:
                    for l2 in L2:



                        train_loader = data_loader_mlp(train_X, train_Y, batch_size=batch_size, shuffle=True, drop_last=False)
                        val_loader = data_loader_mlp(val_X, val_Y, batch_size=len(val_Y), shuffle=True, drop_last=False)


                        model = MLP(l1, l2, p).to(device)

                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                        
                        running_loss_list = []
                        val_running_loss_list = []


                        performance_record = {'loss':[], 'val_loss': []}


                        early_stop_thresh = 10
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

                            #print(f'Epoch{epoch}: loss: {np.mean(running_loss):.4f} val_loss: {np.mean(val_running_loss):.4f}')
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


                        train_accuracy = accuracy_classification(model, train_loader, 'train')
                        val_accuracy = accuracy_classification(model, val_loader, 'val')
                        

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

                            #PATH = f'MLP_{fold}_class_T.pth'
                            torch.save(model.state_dict(), PATH)




    return best_train_acc, best_val_acc, best_performance_record, hyper


def pretrained_model():
    
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    model.fc = nn.Sequential(nn.ReLU(),
                          
                                    nn.Linear(512, 3) #150, 50176
                                   
                                     )
    model.to(device)
    return model
    	
	
def cnn_class_gridsearch(train_X, train_Y, val_X, val_Y,Learning_rate, Batch_size, fold,model_path, epochs=350):
	"""CNN classifier grid hyperparameters gridsearch"""
	criterion = nn.CrossEntropyLoss()
	early_stop_thresh = 15
	best_val_acc = 0
	best_performance_record = 0
	hyper = {}
	performance_record = {'loss':[], 'val_loss': []}
	for learning_rate in Learning_rate:
		for batch_size in Batch_size:

			train_loader= data_loader_fn(train_X, train_Y, transform, batch_size)
			val_loader = data_loader_test_fn(val_X, val_Y, transform, batch_size=len(val_Y))
			val_loader2 = data_loader_test_fn(val_X, val_Y, transform_test,batch_size=len(val_Y))

			model = pretrained_model()
			optimizer = optim.Adam(model.parameters(), lr=learning_rate)

			running_loss_list = []
			val_running_loss_list = []
			performance_record = {'loss':[], 'val_loss': []}
			
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


			train_accuracy = accuracy_classification(model, train_loader, 'train')
			val_accuracy = accuracy_classification(model, val_loader2, 'val')
			train_f1score = f1score_cnn_fn(model, train_loader, 'train')
			val_f1score = f1score_cnn_fn(model, val_loader2, 'val')

			if val_accuracy > best_val_acc:
				best_val_acc = val_accuracy
				best_train_acc = train_accuracy
				best_trainf1 = train_f1score
				best_valf1 = val_f1score

				best_performance_record = performance_record
				hyper['learning_rate'] = learning_rate
				hyper['batch_size'] = batch_size

				hyper['epochs'] = epoch
				
				PATH = os.path.join('Models', model_path)
				#PATH = f'model_micro_{fold}_class_T.pth'
				torch.save(model.state_dict(), PATH)




	return best_train_acc, best_val_acc, best_trainf1, best_valf1, best_performance_record, hyper
	

