import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import models,utils,datasets,transforms
from main_model import lenet
import numpy as np
import pickle
from collections import OrderedDict
import time
import cv2
import os
import matplotlib.pyplot as plt
import argparse

save_path = "EWC_LeNet_models"

def get_labels_indices(target, label):
    label_indices = []

    for i in range(len(target)):
        if (target[i] in label):
            label_indices.append(i)

    return label_indices


def permute_input(input_tensor, permutation):
    if permutation is None:
        return input_tensor
    
    c,h,w = input_tensor.size()
    input_tensor = input_tensor.view(c,-1)
    input_tensor = input_tensor[:, permutation]
    input_tensor = input_tensor.view(c,h,w)
    return input_tensor
    

def get_dataset(permutation=None):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: permute_input(x, permutation))
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: permute_input(x, permutation))
        ]),
    }


    og_train_data = datasets.MNIST("MNIST_data", download=True, train=True, transform = data_transforms['train'])
    og_val_data = datasets.MNIST("MNIST_data", download=True, train=False, transform = data_transforms['val'])
    return og_train_data, og_val_data





def progress_bar(i, epochs, train_error, val_error, ewc_error_log, accuracy, shenanigans): 
    print("\r[{}{}] Epoch Num {} Loss's Training:{:.6f} Validation:{:.6f} EWC:{:.6f} Accuracy:{:2.2f}, shenanigans: {}".format("="*i, " "*(epochs - i), i, train_error, val_error, ewc_error_log, accuracy, shenanigans), end='')

class Lenet_train(object):
    def __init__(self, load_file = None, debug=0):
        np.random.seed(0)
        torch.manual_seed(0)
        self.use_gpu = False
        self.debug = debug
        #set some hyperparams
        self.train_batch = 10
        self.val_batch = 10
        self.lamda = 1
        #IO size
        self.rows = 28
        self.cols = 28
        self.input_size = self.rows*self.cols
        self.output_size = 10
        self.random_inputs = 1
        self.num_tasks = 1 + self.random_inputs
        #epochs number
        self.epochs = 60
        #some logging facilities
        self.start_epoch = 0
        self.best_accuracy = 0
        self.accuracy = 0
        self.accuracy_dictionary = {}
        self.train_error_dictionary = {}
        self.validation_error_dictionary = {}
        #track currset accuracy dictionary
        self.accuracy_dictionary_list = []
        for i in range(self.num_tasks):
            self.accuracy_dictionary_list.append({})
        #create model
        self.layer_sizes = [self.input_size, 512,128,32,self.output_size] 
        self.model = lenet()
        if (self.debug):
            print(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.opt = optim.SGD(self.model.parameters(), lr=0.05)
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[10, 40], gamma=0.1)

        #create dataloader
        self.permutations = [np.random.permutation(28*28) for x in range(self.random_inputs)]
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
            ]),
        }
        self.og_train_data = datasets.MNIST("MNIST_data", download=True, train=True, transform = data_transforms['train'])
        self.og_val_data = datasets.MNIST("MNIST_data", download=True, train=False, transform = data_transforms['train'])
        self.val_data_size = len(self.og_val_data)
        self.train_data_size = len(self.og_train_data)
        self.train_loader = DataLoader(self.og_train_data, batch_size=self.train_batch, shuffle=True)
        self.val_loader = DataLoader(self.og_val_data, batch_size=self.val_batch)
        self.val_loader_list = [self.val_loader]
        self.val_data_list = [self.og_val_data]
        if(load_file != None):
            if (self.debug > 0):
                print("loading state")
            checkpoint = torch.load(load_file)
            self.model.load_state_dict(checkpoint['state_dict'])

    
    def train(self, save_model_file=None, save_all=False, progress=True):
        for i in range(self.start_epoch + 1, self.epoch + 1):
            shenanigans = 0
            currset_validation_error = 0
            train_error = 0
            ewc_t = 0
            validation_error = 0
            num_correct = 0
            broke_ewc = False
            currset_correct = [0]*len(self.val_loader_list)
            self.model.train()
            for batch, (data, label) in enumerate(self.train_loader):
                if self.use_gpu:
                    input_data, g_label = Variable(data.cuda()), Variable(label.cuda())
                else:
                    input_data, g_label = Variable(data), Variable(label)
                self.opt.zero_grad()
                output_vector = self.model(input_data)
                batch_error = self.criterion(output_vector, g_label)
                batch_error.backward()
                for name, param in self.model.named_parameters():
                    param_grad = param.grad.clone()
                ewc_error = self.model.get_ewc_loss(100)
                if (ewc_error != 0.0):
                    ewc_error.backward()
                final_error = batch_error + ewc_error
                #print(batch_error.item(), ewc_error.item())
                train_error += final_error.item()
                ewc_t += ewc_error.item()
                self.opt.step()
            
            self.model.eval()
            for val in range(len(self.val_loader_list)):
                for batch,(v_data, v_label) in enumerate(self.val_loader_list[val]):
                    input_data, g_label = Variable(v_data), Variable(v_label)
                    
                    output_vector = self.model(input_data)
                    val_error = self.criterion(output_vector, g_label)
                    validation_error += val_error.item()
                    dummy, predicted_output = torch.max(output_vector, 1)
                    try:
                        for j in range(self.val_batch):
                            if (predicted_output[j].item() == v_label[j]):
                                num_correct += 1
                                currset_correct[val] += 1

                    except:
                        for j in range(predicted_output.size()[0]):
                            if (predicted_output[j].item() == v_label[j]):
                                num_correct += 1
                                currset_correct[val] += 1
                currset_accuracy = (100.0 * currset_correct[val]) / self.val_data_size
                self.accuracy_dictionary_list[val][i] = currset_accuracy
            accuracy = (100.0 * num_correct) / (len(self.val_loader_list) * self.val_data_size)
            train_error = (1.0 * train_error) / (self.train_data_size)
            ewc_t = (1.0 * ewc_t) / (self.train_data_size)
            validation_error = (1.0 * validation_error) / (len(self.val_loader_list) * self.val_data_size)
            if progress:
                progress_bar(i, self.epoch, train_error, validation_error, ewc_t, accuracy, shenanigans)
            if (self.debug >5):
                print(i, train_error, validation_error, ewc_t, accuracy)
            #record stat
            self.accuracy_dictionary[i] = accuracy
            self.accuracy = accuracy
            self.train_error_dictionary[i] = train_error
            #self.ewc_train_error[i] = ewc_t
            self.validation_error_dictionary[i] = validation_error
            if accuracy > self.best_accuracy : 
                self.best_accuracy = accuracy
                self.better = 1
            if (save_model_file != None):
                model_dict = {'epoch' : i,
                        'optimizer_dict':self.opt.state_dict(),
                        'state_dict':self.model.state_dict()}
                best_model_file = save_model_file
                if (save_all):
                    torch.save(model_dict, best_model_file)
                    self.better = 0
                elif (self.better == 1):
                    torch.save(model_dict, best_model_file)
                    self.better = 0
    def make_train_dataloaders(self):
        train_idxs = get_labels_indices(self.og_train_data.train_labels, range(10)) 
        self.train_data_size = len(train_idxs)
        self.train_loader =DataLoader(self.train_data, batch_size=self.train_batch)
        if (self.debug > 10):
            print("\nTrain idxs ", self.train_data_size) 
       
    def make_val_dataloaders(self):
        for k in range(1,len(self.val_data_list)):
            val_idxs = get_labels_indices(self.val_data_list[k].test_labels, range(10))
            self.val_data_size = len(val_idxs)
            val_loader = DataLoader(self.val_data_list[k], batch_size=self.val_batch)
            self.val_loader_list.append(val_loader)
            if (self.debug > 10):
                print(" Val _idxs ", self.val_data_size, "num loaders ", len(self.val_loader_list)) 

    def make_val_datasets(self, task_num):
        for perm in self.permutations:
            dummy, val_dset = get_dataset(perm)
            if (len(self.val_data_list) > task_num):
                break
            else:
                self.val_data_list.append(val_dset)

    def clear_lists(self):
        self.val_loader_list = [self.val_loader]
        self.val_data_list = [self.og_val_data]

    def debug_val_loader_list(self):
        for val in range(len(self.val_loader_list)):
            print("Debug")
            val_iter = iter(self.val_loader_list[val])
            for test in range(10):
                data, label = val_iter.next()
                print(data)

    def run_experiments(self, exp_num):
        if (exp_num == 0):
            self.epochs = 30
            self.start_epoch = 0
            self.epoch = 15
            self.train(save_model_file = os.path.join(save_path, "OG_LeNet.pt"))
            self.accuracy_dictionary_list.append(self.accuracy_dictionary)
            my_dicts = [self.train_error_dictionary, self.validation_error_dictionary, self.accuracy_dictionary_list]
            pickle.dump(my_dicts, open("LeNet_0.pkl", "wb"))
        elif (exp_num == 1):
            for task in range(self.num_tasks):
                print("Task ", task)
                self.start_epoch = 5*(task)

                self.epoch = 5*(task + 1)
                if (task == 0):
                    self.make_val_datasets(task)
                    self.make_val_dataloaders()
                    self.train(save_model_file= os.path.join(save_path, "OG_LeNet_exp1.pt"))
                    self.clear_lists()
                else:
                    self.train_data, dummy = get_dataset(self.permutations[task - 1])
                    self.make_train_dataloaders()
                    self.make_val_datasets(task)
                    self.make_val_dataloaders()
                    self.train(save_model_file= os.path.join(save_path, "OG_LeNet_exp1.pt"))
                    self.clear_lists()
            self.accuracy_dictionary_list.append(self.accuracy_dictionary)
            my_dicts = [self.train_error_dictionary, self.validation_error_dictionary, self.accuracy_dictionary_list]
            pickle.dump(my_dicts, open("LeNet_control.pkl", "wb"))
        elif (exp_num == 2):
            for task in range(self.num_tasks):
                print("Task ", task)
                self.start_epoch = 2*(task)
                self.epoch = 2*(task + 1)
                if (task == 0):
                    self.make_val_datasets(task)
                    self.make_val_dataloaders()
                    self.train(save_model_file= os.path.join(save_path, "OG_LeNet_exp1.pt"))
                    self.clear_lists()
                else:
                    self.train_data, dummy = get_dataset(self.permutations[task - 1])
                    self.make_train_dataloaders()
                    self.make_val_datasets(task)
                    self.make_val_dataloaders()
                    self.train(save_model_file= os.path.join(save_path, "OG_LeNet_exp1.pt"))
                    self.clear_lists()
            self.accuracy_dictionary_list.append(self.accuracy_dictionary)
            my_dicts = [self.train_error_dictionary, self.validation_error_dictionary, self.accuracy_dictionary_list]
            pickle.dump(my_dicts, open("LeNet_hope.pkl", "wb"))
        elif (exp_num == 3):
            for task in range(self.num_tasks):
                print("Task ", task)
                self.start_epoch = 2*(task)
                self.epoch = 2*(task + 1)
                if (task == 0):
                    self.make_val_datasets(task)
                    self.make_val_dataloaders()
                    self.train(save_model_file= os.path.join(save_path, "OG_LeNet_exp1.pt"))
                    self.clear_lists()
                    self.model.make_fisher_matrix(self.og_train_data, self.train_batch, range(10))
                else:
                    self.train_data, dummy = get_dataset(self.permutations[task - 1])
                    self.make_train_dataloaders()
                    self.make_val_datasets(task)
                    self.make_val_dataloaders()
                    self.train(save_model_file= os.path.join(save_path, "OG_LeNet_exp1.pt"))
                    self.clear_lists()
            self.accuracy_dictionary_list.append(self.accuracy_dictionary)
            my_dicts = [self.train_error_dictionary, self.validation_error_dictionary, self.accuracy_dictionary_list]
            pickle.dump(my_dicts, open("LeNet_1.pkl", "wb"))
         

        

        

if __name__ == '__main__':
    a = Lenet_train(debug=1, load_file = "EWC_LeNet_models/OG_LeNet.pt")
    a.run_experiments(2)
    print("\n Finished")



