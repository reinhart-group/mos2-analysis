import random
import numpy as np
import os
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler

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

# commit check

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='not majority')


def stratified_train_test_group_kfold(X, Y, groups, n_splits, test_fold):
    """this fuction takes X, Y, groups, n_splits, test_fold, val_fold
    and returns the data sets for train, val and test
    X: the features, a numpy arrays
    Y: the targets, a numpy arrays
    groups: group identify of data points, a 1d numpy arrays
    n_splits: the part to which to split the data, 
    test is 1 part, Train is n_splits-1, val is 1/n_splits of Train
    test_fold: which fold is used for test from data, an integer
    """
#Splitting the data to Train and test    
    group_kfold1 = StratifiedGroupKFold(n_splits=n_splits)

    print(type(group_kfold1.split(X, Y, groups)))
    Train_indices = []
    test_indices = []
    for (i, j) in group_kfold1.split(X, Y, groups):
        Train_indices.append(i)
        test_indices.append(j)

    Train_X = X[Train_indices[test_fold]]
    Train_Y = Y[Train_indices[test_fold]]
    Train_groups = groups[Train_indices[test_fold]]

    test_X = X[test_indices[test_fold]]
    test_Y = Y[test_indices[test_fold]]
    
    return Train_groups, Train_X, Train_Y, test_X, test_Y


def pred2class(predicted):
    """the function bins the predicted value into the different classes"""
    #predicted = predicted.tolist()
    pred_class = []
    for index, item in enumerate(predicted):
        if item <= 925:# 0.5, 925
            pred_class.append(900)
        elif item <=975:# 1.5, 975
            pred_class.append(950)
        elif item >975:#1.5, 975
            pred_class.append(1000)    
    
    return pred_class
    
def data_loader_fn(x, y, transform, batch_size):
    target = np.array(y)
    data = np.array(x)
    labels_unique, class_sample_count = np.unique(target, return_counts=True)
    weight = [sum(class_sample_count) / c for c in class_sample_count]


    samples_weight = np.array([weight[t] for t in target])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    target = torch.from_numpy(target)
    data = torch.from_numpy(data)
    #train_dataset = torch.utils.data.TensorDataset(data, target)

    dataset = Dataset(data, target, transform)
    
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, sampler=sampler, drop_last=False)
    
    return data_loader


def data_loader_reg(x, y, transform, batch_size):
    target = np.array(y)
    data = np.array(x)
    labels_unique, class_sample_count = np.unique(target, return_counts=True)
    class_dict = {900.0:0, 950.0:1, 1000.0:2}

    weight = [sum(class_sample_count) / c for c in class_sample_count]


    samples_weight = np.array([weight[class_dict[t]] for t in target])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    target = np.array(target).reshape(len(target),1)
    target = torch.tensor(target, dtype=torch.float32)
    data = torch.tensor(data, dtype=torch.float32)
    #train_dataset = torch.utils.data.TensorDataset(data, target)

    dataset = Dataset(data, target, transform)
    
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, sampler=sampler, drop_last=False)
    
    return data_loader


def data_loader_test_fn(x, y, transform, batch_size):
    data = torch.tensor(x)
    target = torch.tensor(y)
    dataset = Dataset(data, target, transform)

    data_loader = DataLoader(dataset,  batch_size = batch_size, shuffle = False, drop_last=False)#, num_workers= 2)
   
    return data_loader
    

def data_loader_test_reg(x, y, transform, batch_size):
	y = np.array(y).reshape(len(y),1)
	data = torch.tensor(x, dtype=torch.float32)
	target = torch.tensor(y, dtype=torch.float32)
	dataset = Dataset(data, target, transform)

	data_loader = DataLoader(dataset,  batch_size = batch_size, shuffle = False, drop_last=False)#, num_workers= 2)
	return data_loader
	
	
def data_loader_nnrak(x, y, transform, batch_size):
    target = np.array(y)
    data = np.array(x)
    labels_unique, class_sample_count = np.unique(target, return_counts=True)
    weight = [sum(class_sample_count) / c for c in class_sample_count]


    samples_weight = np.array([weight[t] for t in target])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    target = np.array(target).reshape(len(target),1)
    target = torch.tensor(target, dtype=torch.float32)
    data = torch.tensor(data, dtype=torch.float32)
    #train_dataset = torch.utils.data.TensorDataset(data, target)

    dataset = Dataset(data, target, transform)
    
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, sampler=sampler, drop_last=False)
    
    return data_loader
    
    
def ordinal_loss(predictions, targets):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf
    
    predictions: List[List[float]], targets: List[float]"""

    # Create out modified target with [batch_size, num_labels] shape
    modified_target = torch.zeros_like(predictions)

    # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
    for i, target in enumerate(targets):
        target = int(target)
        modified_target[i, 0:target+1] = 1
    loss = nn.MSELoss(reduction='none')(predictions, modified_target).sum(axis=1)
    loss = loss.sum()
    return loss  
    
def nnrank2label(pred):
    """Convert ordinal predictions to class labels, e.g.
    
    [0.9, 0.1, 0.1] -> 0
    [0.60, 0.51, 0.1] -> 1
    [0.7, 0.7, 0.9] -> 2
    etc.
    pred: np.ndarray
    """
    class_pred = (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1
    return class_pred   
     
    
def accuracy_nnrank(trained_model, data_loader, data_type):
	correct = 0
	total = 0
	trained_model.eval()
	#with torch.no_grad():
	for data in data_loader:
		images, labels = data
		images, labels = images.cuda(), labels.cuda()

		outputs = trained_model(images)
		outputs = nnrank2label(outputs)
		labels =[np.rint(item)[0] for item in labels.cpu().numpy().tolist()]
		outputs = [np.rint(item) for item in outputs.cpu().detach().numpy().tolist()]
		for index, item in enumerate(labels):
			if labels[index]==outputs[index]:
	    			correct += 1
			total += 1
	accuracy = 100 * correct / total

	print(f'Accuracy of the network on the {total} {data_type} images: {accuracy :.1f} %')

	return accuracy
	

	      
    	

transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.RandomRotation(degrees= (0, 180)),
      transforms.RandomHorizontalFlip(0.5),
      transforms.RandomVerticalFlip(0.5),
      transforms.ToTensor(),
      #transforms.Normalize(mean=mean, std=std),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

transform_test = transforms.Compose([
      transforms.ToPILImage(),
      transforms.ToTensor(),
      #transforms.Normalize(mean=mean, std=std),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

class Dataset():
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, transform):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples' 
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.transform(ID)
        y = self.labels[index]
        return X, y      



def data_loader_mlp(X, Y, batch_size, shuffle, drop_last):
    data_tensor = torch.tensor(X, dtype=torch.float32)    #all_img test_x, test_y
    target_tensor = torch.tensor(Y)#, dtype = torch.float32)   #target
    
    
    dataset = Dataset_mlp(data_tensor, target_tensor)
    data_loader = DataLoader(dataset,  batch_size = batch_size, shuffle = shuffle, drop_last=drop_last)#model_micro.eval()
    return data_loader
    
def data_loader_mlp_reg(X, Y, batch_size, shuffle, drop_last):
    Y = Y.reshape(len(Y), 1)
    data_tensor = torch.tensor(X, dtype=torch.float32)    #all_img test_x, test_y
    target_tensor = torch.tensor(Y, dtype = torch.float32)   #target
    
    
    dataset = Dataset_mlp(data_tensor, target_tensor)
    data_loader = DataLoader(dataset,  batch_size = batch_size, shuffle = shuffle, drop_last=drop_last)#model_micro.eval()
    return data_loader
        
    
class Dataset_mlp():
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs


  def __len__(self):
        'Denotes the total number of samples' 
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = self.list_IDs[index]
        y = self.labels[index]
        return X, y      



def accuracy_classification(trained_model, data_loader, data_type):
    correct = 0
    total = 0
    trained_model.eval()
    #with torch.no_grad():
    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        outputs = trained_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the {total} {data_type} images: {accuracy :.1f} %')
    
    return accuracy
    

def accuracy_regression(trained_model, data_loader, data_type):
	correct = 0
	total = 0
	trained_model.eval()
	#with torch.no_grad():
	for data in data_loader:
		images, labels = data
		images, labels = images.cuda(), labels.cuda()

		outputs = trained_model(images)
		labels =[item for item in labels.cpu().numpy().tolist()]
		outputs = [pred2class(item) for item in outputs.cpu().detach().numpy().tolist()]
		for index, item in enumerate(labels):
		    if labels[index]==outputs[index]:
		    	correct += 1
		    total += 1
	accuracy = 100 * correct / total

	print(f'Accuracy of the network on the {total} {data_type} images: {accuracy :.1f} %')

	return accuracy

    
        
def f1score_cnn_fn(trained_model, data_loader, data_type):
    trained_model.eval()
    #with torch.no_grad():
    Labels = []
    Predicted = []
    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = trained_model(images)
        _, predicted = torch.max(outputs.data, 1)

        Labels += labels.cpu().tolist()
        Predicted += predicted.cpu().tolist()

    f1score = f1_score(Labels, Predicted, average='macro')
    print(f'f1score of the network on the {data_type}: {f1score :.2f} ')
    return f1score


def confusion_cnn_fn(trained_model, data_loader, data_type):
    correct = 0
    total = 0
    trained_model.eval()
    #with torch.no_grad():
    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = trained_model(images)
        _, predicted = torch.max(outputs.data, 1)
        cm_test = confusion_matrix(labels.cpu(), predicted.cpu())
        #print(f'{data_type} confusion matrix: {cm_test}')

    return cm_test
    

def reg_confusion_cnn_fn(trained_model, data_loader, data_type):
    correct = 0
    total = 0
    trained_model.eval()
    #with torch.no_grad():
    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = trained_model(images)

        labels =[item for item in labels.cpu().numpy().tolist()]
        outputs = [pred2class(item) for item in outputs.cpu().detach().numpy().tolist()]
        cm_test = confusion_matrix(labels, outputs)
        #print(f'{data_type} confusion matrix: {cm_test}')

    return cm_test
    

def nnrank_confusion_cnn_fn(trained_model, data_loader, data_type):
    trained_model.eval()
    Labels = []
    Outputs = []
    for data in data_loader:
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        outputs = trained_model(images)
        outputs = nnrank2label(outputs)
        labels =[np.rint(item) for item in labels.cpu().numpy().tolist()]
        outputs = [np.rint(item) for item in outputs.cpu().detach().numpy().tolist()]
        Labels += labels
        Outputs += outputs

    cm_test = confusion_matrix(Labels, Outputs)
    #print(f'confusion matrix: {cm_test}')

    return cm_test
        

def rmse_cnn_fn(trained_model, data_loader, data_type):
    trained_model.eval()
    #with torch.no_grad():
    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = trained_model(images)

        labels =[item for item in labels.cpu().numpy().tolist()]
        outputs = [item for item in outputs.cpu().detach().numpy().tolist()]

        rmse_test = np.sqrt(mean_squared_error(labels, outputs))
        #print(f'{data_type} rmse: {rmse_test}')

    return rmse_test



def cnn_class_cross_val_final_test(trained_model, X, Y, data_type, root_path):
    best_test = []
    confusion_matrix_test = []
    for fold in range(10):
        
        data_loader = data_loader_test_fn(X, Y, transform_test,batch_size=len(Y))
        
        
        PATH = os.path.join('Models', root_path)
        PATH = os.path.join(PATH, f'{fold}_model.pth')
        trained_model.load_state_dict(torch.load(PATH))
        trained_model.eval()
        
        acc_test = accuracy_classification(trained_model, data_loader, data_type)
        cm_test = confusion_cnn_fn(trained_model, data_loader, data_type)
        best_test.append(acc_test)
        confusion_matrix_test.append(cm_test)
    return best_test, confusion_matrix_test    
    
    
def cnn_reg_cross_val_final_test(trained_model, X, Y, data_type, root_path):
	best_test = []
	confusion_matrix_test = []
	rmse_test_folds = []
	for fold in range(10):
        
		data_loader = data_loader_test_reg(X, Y, transform_test,batch_size=len(Y))


		PATH = os.path.join('Models', root_path)
		PATH = os.path.join(PATH, f'{fold}_model.pth')
		trained_model.load_state_dict(torch.load(PATH))
		trained_model.eval()

		acc_test = accuracy_regression(trained_model, data_loader, data_type)
		cm_test = reg_confusion_cnn_fn(trained_model, data_loader, data_type)
		rmse_test = rmse_cnn_fn(trained_model, data_loader, data_type)

		best_test.append(acc_test)
		confusion_matrix_test.append(cm_test)
		rmse_test_folds.append(rmse_test)
	
	return best_test, confusion_matrix_test, rmse_test_folds
	
	
def cnn_nnrank_cross_val_final_test(trained_model, X, Y, data_type, root_path):
    best_test = []
    confusion_matrix_test = []
    for fold in range(10):
        
        data_loader = data_loader_test_reg(X, Y, transform_test,batch_size=len(Y))
        
        
        PATH = os.path.join('Models', root_path)
        PATH = os.path.join(PATH, f'{fold}_model.pth')
        trained_model.load_state_dict(torch.load(PATH))
        trained_model.eval()
        
        acc_test = accuracy_nnrank(trained_model, data_loader, data_type)
        cm_test = nnrank_confusion_cnn_fn(trained_model, data_loader, data_type)
        best_test.append(acc_test)
        confusion_matrix_test.append(cm_test)
    return best_test, confusion_matrix_test    	
    
    
def mlp_class_cross_val_final_test(trained_model, X, Y, data_type, root_path):
    best_test = []
    confusion_matrix_test = []
    for fold in range(10):
        
        data_loader = data_loader_mlp(X, Y, batch_size=len(Y), shuffle=False, drop_last=False)
        
        
        PATH = os.path.join('Models', root_path)
        PATH = os.path.join(PATH, f'{fold}_model.pth')
        trained_model.load_state_dict(torch.load(PATH))
        trained_model.eval()
        
        acc_test = accuracy_classification(trained_model, data_loader, data_type)
        cm_test = confusion_cnn_fn(trained_model, data_loader, data_type)
        best_test.append(acc_test)
        confusion_matrix_test.append(cm_test)
    return best_test, confusion_matrix_test
    
    
def mlp_nnrank_cross_val_final_test(trained_model, X, Y, data_type, root_path):
    best_test = []
    confusion_matrix_test = []
    for fold in range(10):
        
        data_loader = data_loader_mlp_reg(X, Y, batch_size=len(Y), shuffle=False, drop_last=False)
        
        
        PATH = os.path.join('Models', root_path)
        PATH = os.path.join(PATH, f'{fold}_model.pth')
        trained_model.load_state_dict(torch.load(PATH))
        trained_model.eval()
        
        acc_test = accuracy_nnrank(trained_model, data_loader, data_type)
        cm_test = nnrank_confusion_cnn_fn(trained_model, data_loader, data_type)
        best_test.append(acc_test)
        confusion_matrix_test.append(cm_test)
    return best_test, confusion_matrix_test    	    
    

def mlp_reg_cross_val_final_test(trained_model, X, Y, data_type, root_path):
	best_test = []
	confusion_matrix_test = []
	rmse_test_folds = []
	for fold in range(10):
        
		data_loader = data_loader_mlp_reg(X, Y, batch_size=len(Y), shuffle=False, drop_last=False)


		PATH = os.path.join('Models', root_path)
		PATH = os.path.join(PATH, f'{fold}_model.pth')
		trained_model.load_state_dict(torch.load(PATH))
		trained_model.eval()

		acc_test = accuracy_regression(trained_model, data_loader, data_type)
		cm_test = reg_confusion_cnn_fn(trained_model, data_loader, data_type)
		rmse_test = rmse_cnn_fn(trained_model, data_loader, data_type)

		best_test.append(acc_test)
		confusion_matrix_test.append(cm_test)
		rmse_test_folds.append(rmse_test)
	
	return best_test, confusion_matrix_test, rmse_test_folds             
            

        
def model_test_regression(train_val_X, train_val_Y, train_val_groups,test_X, test_Y, best_fold, model_path):
    best_test = []
    root_mean_squared_error = []
    confusion_matrix_test = []
    
    for fold in range(10):
        
        group, train_X, train_Y, val_X, val_Y = stratified_train_test_group_kfold(train_val_X, train_val_Y, train_val_groups, n_splits=10, test_fold=fold)
        train_X, train_Y = oversample.fit_resample(train_X, train_Y)
        
        PATH = os.path.join('Models', model_path)
        loaded_model = pickle.load(open(PATH, 'rb'))
        loaded_model.fit(train_X, train_Y)
        pred_test_Y =loaded_model.predict(test_X)        
        rmse = np.sqrt(mean_squared_error(test_Y, pred_test_Y))
        pred_test_Y =pred2class(pred_test_Y)
        cm_test = confusion_matrix(test_Y, pred_test_Y)
        acc_test = accuracy_score(test_Y, pred_test_Y)
        
        best_test.append(acc_test)
        root_mean_squared_error.append(rmse)
        confusion_matrix_test.append(cm_test)
    
    return best_test, root_mean_squared_error, confusion_matrix_test
    

    
def model_test_classification(train_val_X, train_val_Y, train_val_groups,test_X, test_Y, best_fold, model_path):
    best_test = []
    confusion_matrix_test = []

    for fold in range(10):
        
        group, train_X, train_Y, val_X, val_Y = stratified_train_test_group_kfold(train_val_X, train_val_Y, train_val_groups, n_splits=10, test_fold=fold)
        train_X, train_Y = oversample.fit_resample(train_X, train_Y)
        
        PATH = os.path.join('Models', model_path)
        loaded_model = pickle.load(open(PATH, 'rb'))
        loaded_model.fit(train_X, train_Y)
        pred_test_Y = loaded_model.predict(test_X)
        pred_val_Y = loaded_model.predict(val_X)
        pred_train_Y = loaded_model.predict(train_X)


        cm_test = confusion_matrix(test_Y, pred_test_Y)
        acc_test = accuracy_score(test_Y, pred_test_Y)

        best_test.append(acc_test)
        confusion_matrix_test.append(cm_test)
    
    return best_test, confusion_matrix_test
        
        
