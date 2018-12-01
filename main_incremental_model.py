import torch as torch
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import autograd
from torchvision import models,utils,datasets,transforms

def get_labels_indices(target, label):
    label_indices = []

    for i in range(len(target)):
        if (target[i] in label):
            label_indices.append(i)

    return label_indices

class MLP_incremental(nn.Module):
    def __init__(self, layer_sizes,dropout = False, use_gpu=False, debug=False):
        super(MLP_incremental, self).__init__()
        self.layers = nn.ModuleList([]) #the ones i train using the regularized
        self.output_train = nn.ModuleList([]) #the ones i train after initializing to 0
        self.output_fixed = nn.ModuleList([]) #the ones that are fixed for inference
        #for regularization
        self.fisher_matrix = None
        self.prev_parameters = None
        #define layer sizes:
        self.layer_sizes = layer_sizes
        #create proxies for output layer
        self.max_classes = self.layer_sizes[-1]
        self.prev_classes = 0
        self.curr_classes = 0

        self.size = len(layer_sizes)
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.debug =debug
        for i in range(self.size-2):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            layer = nn.ReLU(inplace=True)
            self.layers.append(layer)
            if (self.dropout and i > 0):
                layer = nn.Dropout()
                self.layers.append(layer)

        penultimate_layer_size = layer_sizes[-2]
        #Create the fixed and train output neurons
        for i in range(self.max_classes):
            layer = nn.Linear(penultimate_layer_size, 1)
            for p in layer.parameters():
                p.data.fill_(0)
                p.requires_grad=False
            self.output_train.append(layer)
            layer = nn.Linear(penultimate_layer_size, 1)
            for p in layer.parameters():
                p.requires_grad = False
            self.output_fixed.append(layer)

    def set_numclasses_train(self, num_classes, re_init=False):
        self.prev_classes = self.curr_classes
        self.curr_classes = num_classes
        for i in range(self.max_classes):
            for p in self.output_train[i].parameters():
                if (re_init):
                    p.data.fill_(0)
                if (i < self.curr_classes):
                    p.requires_grad = True

    def forward(self,x):
        mylambda = lambda x,layer:layer(x)
        y = torch.zeros(self.curr_classes)
        if self.training:
            if len(self.layers) > 0:
                penultimate_output = reduce(mylambda, self.layers, x)
            else:
                penultimate_output = x
            for i in range(self.curr_classes):
                if (i == 0):
                    y = self.output_train[i](penultimate_output)
                else:
                    y = torch.cat((y, self.output_train[i](penultimate_output)),1)
        else:
            if len(self.layers) > 0:
                penultimate_output = reduce(mylambda, self.layers, x)
            else:
                penultimate_output = x
            for i in range(self.curr_classes):
                if (i == 0):
                    y = self.output_fixed[i](penultimate_output)
                else:
                    y = torch.cat((y, self.output_fixed[i](penultimate_output)),1)
        output = y 
        return output
 


    def make_fisher_matrix(self, previous_dataset, previous_batch_size, prev_nums):
        if (self.debug > 0):
            print("making_fisher")
        prev_idxs = get_labels_indices(previous_dataset.train_labels, prev_nums)
        loader = DataLoader(previous_dataset, batch_size=previous_batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(prev_idxs))
        likelihoods = []
        for k, (i_data,label) in enumerate(loader):
            try:
                data = i_data.view(previous_batch_size,self.layer_sizes[0])
            except:
                data = i_data.view(i_data.size()[0], self.layer_sizes[0])
            data = Variable(data)
            label = Variable(label)
            previous_prediction = self.forward(data)
            log_pp = F.log_softmax(previous_prediction,dim=0)#take a log and softmax
            try:
                likelihoods.append(log_pp[range(previous_batch_size), label.data]) #choose values that correspond with the correct_prediction
            except:
                likelihoods.append(log_pp[range(i_data.size()[0]), label.data])

        final_likelihoods = torch.cat(likelihoods)
        final_likelihood_avg = final_likelihoods.mean(0)
        likelihood_grads = autograd.grad(final_likelihood_avg, self.layers.parameters())
        parameter_names = [n for n,p in self.layers.named_parameters()]
        self.fisher_matrix = {n:grads**2 for n,grads in zip(parameter_names, likelihood_grads)}
        self.prev_parameters = {n:p.data.clone() for n,p in self.layers.named_parameters()}

    def get_fisher_matrix(self):
        return self.fisher_matrix

    def transfer_weights(self):
        for i in range(self.prev_classes, self.curr_classes):
            for (n,p) in self.output_fixed[i].named_parameters():
                for (nt,pt) in self.output_train[i].named_parameters():
                    if(nt == n):
                        p.data.copy_(pt.data - pt.data.mean())

    #for enabling purely CWR
    def fix_features(self):
        for p in self.layers.parameters():
            p.requires_grad=False

    def get_ewc_loss(self,lamda, debug=False):
        try: 
            losses = torch.zeros(1)
            for n,p in self.layers.named_parameters():
                pp_fisher = self.fisher_matrix[n]
                pp = self.prev_parameters[n]
                loss = (pp_fisher*((p - pp)**2)).sum()
                losses = losses + loss
            return (Variable((lamda/2)*(losses)))
        except: 
            return (Variable(torch.zeros(1)))


class lenet_incremental(nn.Module):
    def __init__(self,use_gpu=False, debug=False):
        super(MLP_incremental, self).__init__()
        self.output_train = nn.ModuleList([]) #the ones i train after initializing to 0
        self.output_fixed = nn.ModuleList([]) #the ones that are fixed for inference
        #for regularization
        self.fisher_matrix = None
        self.prev_parameters = None
        #define layer sizes:
        self.layer_sizes = layer_sizes
        #create proxies for output layer
        self.max_classes = self.layer_sizes[-1]
        self.prev_classes = 0
        self.curr_classes = 0

        self.size = len(layer_sizes)
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.debug =debug
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 6, 5, padding=0)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('conv2', nn.Conv2d(6, 16, kernel_size=(5, 5), padding=0)),
            ('batchnorm1', nn.BatchNorm2d(16)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
            ]))
        self.dense_layers = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(16*5*5, 120)),
            ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(120,84)),
            ('relu4', nn.ReLU()),
            ]))

        #Create the fixed and train output neurons
        penultimate_layer_size = 84
        for i in range(self.max_classes):
            layer = nn.Linear(penultimate_layer_size, 1)
            for p in layer.parameters():
                p.data.fill_(0)
                p.requires_grad=False
            self.output_train.append(layer)
            layer = nn.Linear(penultimate_layer_size, 1)
            for p in layer.parameters():
                p.requires_grad = False
    def set_numclasses_train(self, num_classes, re_init=False):
        self.prev_classes = self.curr_classes
        self.curr_classes = num_classes
        for i in range(self.max_classes):
            for p in self.output_train[i].parameters():
                if (re_init):
                    p.data.fill_(0)
                if (i < self.curr_classes):
                    p.requires_grad = True
            self.output_fixed.append(layer)
    
    def set_numclasses_train(self, num_classes, re_init=False):
        self.prev_classes = self.curr_classes
        self.curr_classes = num_classes
        for i in range(self.max_classes):
            for p in self.output_train[i].parameters():
                if (re_init):
                    p.data.fill_(0)
                if (i < self.curr_classes):
                    p.requires_grad = True
    
    def forward(self,x):
        y = torch.zeros(self.curr_classes)
        if self.training:
            penultimate_output = self.features(x)
            penultimate_output = self.dense_layers(x)
            for i in range(self.curr_classes):
                if (i == 0):
                    y = self.output_train[i](penultimate_output)
                else:
                    y = torch.cat((y, self.output_train[i](penultimate_output)),1)
        else:
            penultimate_output = self.features(x)
            penultimate_output = self.dense_layers(x)
            for i in range(self.curr_classes):
                if (i == 0):
                    y = self.output_fixed[i](penultimate_output)
                else:
                    y = torch.cat((y, self.output_fixed[i](penultimate_output)),1)
        return y
    
    def make_fisher_matrix(self, previous_dataset, previous_batch_size, prev_nums):
        if (self.debug > 0):
            print("making_fisher")
        prev_idxs = get_labels_indices(previous_dataset.train_labels, prev_nums)
        loader = DataLoader(previous_dataset, batch_size=previous_batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(prev_idxs))
        likelihoods = []
        for k, (i_data,label) in enumerate(loader):
            try:
                data = i_data.view(previous_batch_size,self.layer_sizes[0])
            except:
                data = i_data.view(i_data.size()[0], self.layer_sizes[0])
            data = Variable(data)
            label = Variable(label)
            previous_prediction = self.forward(data)
            log_pp = F.log_softmax(previous_prediction,dim=0)#take a log and softmax
            try:
                likelihoods.append(log_pp[range(previous_batch_size), label.data]) #choose values that correspond with the correct_prediction
            except:
                likelihoods.append(log_pp[range(i_data.size()[0]), label.data])

        final_likelihoods = torch.cat(likelihoods)
        final_likelihood_avg = final_likelihoods.mean(0)
        likelihood_grads = autograd.grad(final_likelihood_avg, (self.dense_layers.parameters(), self.features.parameters()))
        parameter_names = [n for n,p in self.layers.named_parameters()]
        self.fisher_matrix = {n:grads**2 for n,grads in zip(parameter_names, likelihood_grads)}
        self.prev_parameters = {n:p.data.clone() for n,p in self.layers.named_parameters()}

    def get_fisher_matrix(self):
        return self.fisher_matrix

    def transfer_weights(self):
        for i in range(self.prev_classes, self.curr_classes):
            for (n,p) in self.output_fixed[i].named_parameters():
                for (nt,pt) in self.output_train[i].named_parameters():
                    if(nt == n):
                        p.data.copy_(pt.data - pt.data.mean())

    #for enabling purely CWR
    def fix_features(self):
        for p in self.layers.parameters():
            p.requires_grad=False

    def get_ewc_loss(self,lamda, debug=False):
        try: 
            losses = torch.zeros(1)
            for n,p in self.layers.named_parameters():
                pp_fisher = self.fisher_matrix[n]
                pp = self.prev_parameters[n]
                loss = (pp_fisher*((p - pp)**2)).sum()
                losses = losses + loss
            return (Variable((lamda/2)*(losses)))
        except: 
            return (Variable(torch.zeros(1)))






