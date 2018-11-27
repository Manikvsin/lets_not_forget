import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import models,utils,datasets,transforms
from main_model import ewc_MLP
import numpy as np
import pickle
from collections import OrderedDict
import time
import cv2
import os
import matplotlib.pyplot as plt
import argparse

save_path = "EWC_MLP_models"

def get_labels_indices(target, label):
    label_indices = []

    for i in range(len(target)):
        if (target[i] in label):
            label_indices.append(i)

    return label_indices

def progress_bar(i, epochs, train_error, val_error, ewc_error_log, accuracy): 
    print("\r[{}{}] Epoch Num {} Loss's Training:{:.6f} Validation:{:.6f} EWC:{:.6f} Accuracy:{:2.2f}".format("="*i, " "*(epochs - i), i, train_error, val_error, ewc_error_log, accuracy), end='')

class EWC_train(object):
    def __init__(self):
        #self.use_gpu = torch.cuda.is_available()
        #if self.use_gpu:
        #    print("HEYYY CUDAAAA")
        self.use_gpu = False
        #set some hyperparams
        self.train_batch = 10
        self.val_batch = 10
        self.lamda = 5
        #IO size
        self.rows = 28
        self.cols = 28
        self.input_size = self.rows*self.cols
        self.output_size = 10
        #epochs number
        self.epochs = 60
        #some logging facilities
        self.start_epoch = 0
        self.best_accuracy = 0
        self.accuracy = 0
        self.accuracy_dictionary = {}
        self.train_error_dictionary = {}
        #self.ewc_train_error = {}
        self.validation_error_dictionary = {}
        #create model
        self.layer_sizes = [self.input_size, 512,128,32,self.output_size] 
        self.model = ewc_MLP(self.layer_sizes, dropout=False)
        print(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.opt = optim.SGD(self.model.parameters(), lr=0.05)
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[10, 40], gamma=0.1)

        #create dataloader
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
            ]),
        }
        self.og_train_data = datasets.MNIST("MNIST_data", download=True, train=True, transform = data_transforms['train'])
        self.og_val_data = datasets.MNIST("MNIST_data", download=True, train=False, transform = data_transforms['val'])
        self.train_loader = DataLoader(self.og_train_data, batch_size=self.train_batch, shuffle=True)
        self.val_loader = DataLoader(self.og_val_data, batch_size=self.val_batch, shuffle=True)
    
    def train(self, save_model_file=os.path.join(save_path, "finetuned_alexnet.pt"), save_all=False):
        for i in range(self.start_epoch + 1, self.epoch + 1):
            train_error = 0
            ewc_t = 0
            validation_error = 0
            num_correct = 0
            self.model.train()
            for batch, (data, label) in enumerate(self.train_loader):
                if self.use_gpu:
                    input_data, g_label = Variable(data.cuda()), Variable(label.cuda())
                else:
                    input_data, g_label = Variable(data), Variable(label)
                self.opt.zero_grad()
                try:
                    output_vector = self.model(input_data.view(self.train_batch, self.input_size))
                except:
                    output_vector = self.model(input_data.view(input_data.size()[0], self.input_size))
                batch_error = self.criterion(output_vector, g_label)
                ewc_error = self.model.get_ewc_loss(self.lamda)
                final_error = batch_error + ewc_error
                train_error += final_error.item()
                ewc_t += ewc_error.item()
                final_error.backward()
                self.opt.step()

            self.model.eval()
            for batch,(v_data,v_label) in enumerate(self.val_loader):
                input_data, g_label = Variable(v_data), Variable(v_label)
                if self.use_gpu:
                    input_data, g_label = Variable(v_data.cuda()), Variable(v_label.cuda())
                else:
                    input_data, g_label = Variable(v_data), Variable(v_label)
                try:
                    output_vector = self.model(input_data.view(self.train_batch, self.input_size))
                except:
                    output_vector = self.model(input_data.view(input_data.size()[0], self.input_size))
                val_error = self.criterion(output_vector, g_label)
                validation_error += val_error.item()
                dummy, predicted_output = torch.max(output_vector, 1)
                try:
                    for j in range(self.val_batch):
                        if (predicted_output[j].item() == v_label[j]):
                            num_correct += 1
                except:
                    for j in range(predicted_output.size()[0]):
                        if (predicted_output[j].item() == v_label[j]):
                            num_correct += 1

            accuracy = (100.0 * num_correct) / (self.val_data_size)
            train_error = (1.0 * train_error) / (self.train_data_size)
            ewc_t = (1.0 * ewc_t) / (self.train_data_size)
            validation_error = (1.0 * validation_error) / (self.val_data_size)
            progress_bar(i, self.epochs, train_error, val_error, ewc_t, accuracy)
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
    def make_dataloaders(self, train_nums, val_nums):
        train_idxs = get_labels_indices(self.og_train_data.train_labels, train_nums)
        self.train_data_size = len(train_idxs)
        self.train_loader =DataLoader(self.og_train_data, batch_size=self.train_batch, sampler=torch.utils.data.sampler.SubsetRandomSampler(train_idxs))
        
        val_idxs = get_labels_indices(self.og_val_data.test_labels, val_nums)
        self.val_data_size = len(val_idxs)
        self.val_loader = DataLoader(self.og_val_data, batch_size=self.val_batch, sampler=torch.utils.data.sampler.SubsetRandomSampler(val_idxs))


    def run_experiments(self, exp_num):
        if (exp_num == 0):
            self.start_epoch = 0
            self.epoch = 60
            self.make_dataloaders(range(10), range(10))
            self.train(save_model_file = os.path.join(save_path, "OG_MLP.pt"))
        elif (exp_num == 1):
            #do 0 - 3, 4 - 6, 7 - 9
            self.start_epoch = 0
            self.epoch = 20
            self.make_dataloaders(train_nums = [0,1,2,3], val_nums = [0,1,2,3])
            self.train(save_model_file = os.path.join(save_path, "EWC_MLP_03.pt"), save_all = True)
            #4-6
            self.start_epoch = 20
            self.epoch = 40
            self.make_dataloaders(train_nums = [4,5,6], val_nums = [0,1,2,3,4,5,6])
            self.train(save_model_file = os.path.join(save_path, "EWC_MLP_06.pt"), save_all =True)
            #7-9
            self.start_epoch = 40
            self.epoch = 60
            self.make_dataloaders(train_nums = [7,8,9], val_nums = [0,1,2,3,4,5,6,7,8,9])
            self.train(save_model_file = os.path.join(save_path, "EWC_MLP_09.pt"), save_all=True)
        elif (exp_num == 2):
            #do 0 - 3, 4 - 6, 7 - 9
            self.start_epoch = 0
            self.epoch = 20
            self.make_dataloaders(train_nums = [0,1,2,3], val_nums = [0,1,2,3])
            self.train(save_model_file = os.path.join(save_path, "EWC_MLP_03.pt"))
            self.model.make_fisher_matrix(self.og_train_data, self.train_batch, prev_nums=[0,1,2,3])
            #4-6
            self.start_epoch = 20
            self.epoch = 40
            self.make_dataloaders(train_nums = [4,5,6], val_nums = [0,1,2,3,4,5,6])
            self.train(save_model_file = os.path.join(save_path, "EWC_MLP_06.pt"))
            self.model.make_fisher_matrix(self.og_train_data, self.train_batch, prev_nums=[0,1,2,3,4,5,6])
            #7-9
            self.start_epoch = 40
            self.epoch = 60
            self.make_dataloaders(train_nums = [7,8,9], val_nums = [0,1,2,3,4,5,6,7,8,9])
            self.train(save_model_file = os.path.join(save_path, "EWC_MLP_09.pt"))
        

        

if __name__ == '__main__':
    a = EWC_train()
    a.run_experiments(2)
    print("\n Finished")
    #train_error_dict = a.train_error_dictionary
    #val_error_dict = a.validation_error_dictionary
    #accuracy_dict = a.accuracy_dictionary
    #comp_time_dict = a.computation_time_dictionary
    #inf_tim_dict = a.inference_time_dictionary
    #top5_dict = a.top5_dictionary
    my_ft_dicts = [a.train_error_dictionary, a.validation_error_dictionary, a.accuracy_dictionary]
    pickle.dump(my_ft_dicts, open("ewc_MLP_2.pkl", "wb"))
#lenet_lr_dicts = [lenet_lr_training_error_dictionary, lenet_lr_validation_error_dictionary, lenet_lr_computation_time_dictionary, lenet_lr_inference_time_dictionary, lenet_lr_accuracy_dictionary]
#pickle.dump(lenet_lr_dicts, open("Lenet_Dicts_50_epochs_small_batchnorm_adam.pkl", "wb"))



