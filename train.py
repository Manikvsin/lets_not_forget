import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import models,utils,datasets,transforms
from tiny_image_net_torch import TinyImageNet
import numpy as np
import pickle
from collections import OrderedDict
import time
import cv2
import os
import matplotlib.pyplot as plt
import argparse

# Create an ArgumentParser object which will obtain arguments from command line
parser = argparse.ArgumentParser(description="Fine-tunes a pre-trained AlexNet model, given a path to the tiny image dataset and a path where to save the model after training")
parser.add_argument('--data', type=str, help='path to directory where tiny imagenet dataset is present')
parser.add_argument('--save', type=str, help='path to directory to save trained model after completion of training')
args = parser.parse_args()

data_path = args.data
save_path = args.save

def progress_bar(i, epochs, train_error, val_error, accuracy,top5): 
    print("\r[{}{}] Loss's Training:{:.6f} Validation:{:.6f} Accuracy:{:2.2f} top5: {:2.2f}".format("="*i, " "*(epochs - i), train_error, val_error, accuracy, top5), end='')

class ft_Alexnet(object):
    def __init__(self):
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            print("HEYYY CUDAAAA")
        #get batch sizes
        self.train_batch = 100
        self.val_batch = 100
        self.output_size = 200
        self.epochs = 50
        self.start_epoch = 0
        self.best_accuracy = 0
        self.accuracy = 0
        self.computation_time_dictionary = {}
        self.inference_time_dictionary = {}
        self.accuracy_dictionary = {}
        self.top5_dictionary = {}
        self.train_error_dictionary = {}
        self.validation_error_dictionary = {}
        #get pretrained model
        self.model = models.alexnet(pretrained=True)
        for i in self.model.parameters():
            i.requires_grad=False
        self.model.classifier[6] = nn.Linear(in_features=4096,out_features=self.output_size,bias=True)
        if self.use_gpu:
            self.model.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.opt = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[10, 40], gamma=0.1)

        #create dataloader
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        self.train_data = TinyImageNet(root=data_path, train=True, transform=data_transforms['train'])
        self.val_data = TinyImageNet(root=data_path, train=False, transform = data_transforms['val'])

        self.train_loader = DataLoader(self.train_data, batch_size=self.train_batch, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=self.val_batch, shuffle=True)
    
    def train(self, save_model_file=os.path.join(save_path, "finetuned_alexnet.pt")):
        def onehot_encoding(labels, batch_size):
            onehot_labels = torch.zeros(batch_size, self.output_size)
            for i in range(batch_size):
                onehot_labels[i][labels[i]] = 1
            return onehot_labels

        for i in range(self.start_epoch + 1, self.epochs + 1):
            train_error = 0
            validation_error = 0
            num_correct = 0
            top5_correct = 0
            start_time = time.time()
            self.model.train()
            for batch, (data, label) in enumerate(self.train_loader):
                if self.use_gpu:
                    input_data, g_label = Variable(data.cuda()), Variable(label.cuda())
                else:
                    input_data, g_label = Variable(data), Variable(label)

                self.opt.zero_grad()
                output_vector = self.model(input_data)
                batch_error = self.criterion(output_vector, g_label)
                #if (batch == 1):
                #    print(batch_error)
                train_error += batch_error.item()
                batch_error.backward()
                self.opt.step()
            self.computation_time_dictionary[i] = time.time() - start_time

            self.model.eval()
            infer_time = time.time()
            for batch,(v_data,v_label) in enumerate(self.val_loader):
                input_data, g_label = Variable(v_data), Variable(v_label)
                if self.use_gpu:
                    input_data, g_label = Variable(v_data.cuda()), Variable(v_label.cuda())
                else:
                    input_data, g_label = Variable(v_data), Variable(v_label)
                output_vector = self.model(input_data)
                val_error = self.criterion(output_vector, g_label)
                #if (batch == 1):
                #    print(val_error)
                validation_error += val_error.item()
                dummy, predicted_output = torch.max(output_vector, 1)
                dummy, predicted_top5 = torch.topk(output_vector, 5, dim=1)
                for j in range(self.val_batch):
                    if (predicted_output[j].item() == v_label[j]):
                        num_correct += 1
                    top5_list = [tens.item() for tens in predicted_top5[j]]
                    if (v_label[j] in top5_list):
                        top5_correct += 1
            self.inference_time_dictionary[i] = time.time() - infer_time

            accuracy = (100.0 * num_correct) / len(self.val_loader.dataset)
            top5_accuracy = (100.0 * top5_correct) / len(self.val_loader.dataset)
            train_error = (1.0 * train_error) / len(self.train_loader.dataset)
            validation_error = (1.0 * validation_error) / len(self.val_loader.dataset)
            progress_bar(i, self.epochs, train_error, val_error, accuracy, top5_accuracy)
            #record stat
            self.accuracy_dictionary[i] = accuracy
            self.top5_dictionary[i] = top5_accuracy
            self.accuracy = accuracy
            self.train_error_dictionary[i] = train_error
            self.validation_error_dictionary[i] = validation_error
            if accuracy > self.best_accuracy : 
                self.best_accuracy = accuracy
                self.better = 1
            if (save_model_file != None):
                if (self.better == 1):
                    model_dict = {'epoch' : i,
                                'optimizer_dict':self.opt.state_dict(),
                                'state_dict':self.model.state_dict(),
                                'class_to_label': self.train_data.class_to_label,
                                'idx_to_class': self.train_data.tgt_idx_to_class}
                    best_model_file = save_model_file
                    torch.save(model_dict, best_model_file)
                    self.better = 0
 

if __name__ == '__main__':
    a = ft_Alexnet()
    a.train()
    print("\n Finished")
    #train_error_dict = a.train_error_dictionary
    #val_error_dict = a.validation_error_dictionary
    #accuracy_dict = a.accuracy_dictionary
    #comp_time_dict = a.computation_time_dictionary
    #inf_tim_dict = a.inference_time_dictionary
    #top5_dict = a.top5_dictionary
    my_ft_dicts = [train_error_dict, val_error_dict, accuracy_dict, comp_time_dict, inf_tim_dict, top5_dict]
    pickle.dump(my_ft_dicts, open("FT_Alexnet_dicts__top5_cuda.pkl", "wb"))
#lenet_lr_dicts = [lenet_lr_training_error_dictionary, lenet_lr_validation_error_dictionary, lenet_lr_computation_time_dictionary, lenet_lr_inference_time_dictionary, lenet_lr_accuracy_dictionary]
#pickle.dump(lenet_lr_dicts, open("Lenet_Dicts_50_epochs_small_batchnorm_adam.pkl", "wb"))



