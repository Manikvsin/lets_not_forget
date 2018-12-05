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
from collections import OrderedDict
from copy import deepcopy

def get_labels_indices(target, label):
    label_indices = []

    for i in range(len(target)):
        if (target[i] in label):
            label_indices.append(i)

    return label_indices

class ewc_MLP(nn.Module):
    def __init__(self, layer_sizes,dropout = False, use_gpu=False, debug=False):
        super(ewc_MLP, self).__init__()
        self.layers = nn.ModuleList([])
        self.size = len(layer_sizes)
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.fisher_matrix = None
        self.prev_parameters = None
        self.layer_sizes = layer_sizes
        for i in range(self.size-1):
            if i == (self.size - 2):
                layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
                self.layers.append(layer)
                continue
            else:
                layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
                self.layers.append(layer)
                layer = nn.ReLU(inplace=True)
                self.layers.append(layer)
                if (self.dropout and i > 0):
                    layer = nn.Dropout()
                    self.layers.append(layer)
    def forward(self,x):
        mylambda = lambda x,layer:layer(x)
        return reduce(mylambda, self.layers, x)

    def make_fisher_matrix(self, previous_dataset, previous_batch_size, prev_nums):
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
        likelihood_grads = autograd.grad(final_likelihood_avg, self.parameters())
        parameter_names = [n for n,p in self.named_parameters()]
        self.fisher_matrix = {n:grads**2 for n,grads in zip(parameter_names, likelihood_grads)}
        self.prev_parameters = {n:p.data.clone() for n,p in self.named_parameters()}

    def get_fisher_matrix(self):
        return self.fisher_matrix

    def get_ewc_loss(self,lamda, debug=False):
        try:
            losses = torch.zeros(1)
            for n,p in self.named_parameters():
                pp_fisher = self.fisher_matrix[n]
                pp = self.prev_parameters[n]
                loss = (pp_fisher*((p - pp)**2)).sum()
                losses = losses + loss
            return (Variable((lamda/2)*(losses)))
        except: 
            return (Variable(torch.zeros(1)))


class lenet(nn.Module):
    def __init__(self,use_gpu=False, debug=False):
        super(lenet, self).__init__()
        #for regularization

        self.debug =debug
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 6, kernel_size=(5,5), padding=2)),
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

        self.output_layers = nn.Sequential(OrderedDict([
            ('fc3', nn.Linear(84,10)),
            ]))
        #Create the fixed and train output neurons
    def forward(self,x):
        penultimate_output = self.features(x)
        penultimate_output = penultimate_output.view(penultimate_output.size(0), -1)
        penultimate_output = self.dense_layers(penultimate_output)
        y = self.output_layers(penultimate_output) 
        return y
    
    def make_fisher_matrix(self, previous_dataset, previous_batch_size, prev_nums):
        print("making_fisher")
        prev_idxs = get_labels_indices(previous_dataset.train_labels, prev_nums)
        loader = DataLoader(previous_dataset, batch_size=previous_batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(prev_idxs))
        self.prev_parameters = {n:p.clone() for n,p in self.named_parameters()}
        likelihoods = []
        self.eval()
        #init matrices
        self.fisher_matrix = {n:p.clone().zero_() for n,p in self.named_parameters()}
        for k, (i_data,label) in enumerate(loader):
            self.zero_grad()
            data = Variable(i_data)
            label = Variable(label)
            previous_prediction = self(data)
            log_pp = F.log_softmax(previous_prediction,dim=1)#take a log and softmax
            likelihood = F.nll_loss(log_pp, label)
            likelihood.backward(retain_graph=True)
            for n,p in self.named_parameters():
                self.fisher_matrix[n] = self.fisher_matrix[n] + ((p.grad.data.clone() ** 2)/len(previous_dataset))
            #print(likelihood_grad)


        #for n,p in self.named_parameters():
        #    self.fisher_matrix[n] = self.fisher_matrix[n]
        self.prev_parameters = {n:p.data.clone() for n,p in self.named_parameters()}

#    def make_fisher_matrix_cheat(self, previous_dataset, previous_batch_size, prev_nums):
#        print("making_fisher")
#        prev_idxs = get_labels_indices(previous_dataset.train_labels, prev_nums)
#        loader = DataLoader(previous_dataset, batch_size=previous_batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(prev_idxs))
#        likelihoods = []
#        params = {n:p for n,p in self.named_parameters() if p.requires_grad}
#        self.prev_parameters = {}
#        self.fisher_matrix = {}
#        self.train()
#        likelihoods = []
#        #init matrices
#        for k, (i_data,label) in enumerate(loader):
#            data = Variable(i_data)
#            label = Variable(label)
#            likelihoods.append(
#                    F.log_softmax(self(data))[range(previous_batch_size), label.data]
#                )
#
#        loglikelihood = torch.cat(likelihoods).mean(0)
#        loglikelihood_grads = autograd.grad(loglikelihood, self.parameters())
#        for n,p in self.named_parameters():
#            self.prev_parameters[n] = (p.data.clone())
#
#        parameter_names = [n for n,p in self.named_parameters()]
#        self.fisher_matrix = {n:(g**2).data.clone() for n,g in zip(parameter_names, loglikelihood_grads)}
#
    def get_ewc_loss(self,lamda, debug=False):
        try: 
            losses = Variable(torch.zeros(1), requires_grad=True) 
            for n,p in self.named_parameters():
                pp_fisher = Variable(self.fisher_matrix[n])
                pp = Variable(self.prev_parameters[n])
                loss = (pp_fisher*((p - pp)**2)).sum()
                losses = losses + loss
            losses = lamda*losses
            #print(losses)
            return (losses)
        except:
            return (Variable(torch.zeros(1)))

    def set_num_params(self):
        num_params = 0
        for n,p in self.named_parameters():
            num_params += 1
        return(num_params)


    def dummy_loss(self,lamda, debug=False):
            losses = Variable(torch.zeros(1), requires_grad=True)
            for n,p in self.named_parameters():
                pp_fisher = Variable(torch.ones(1), requires_grad=True)
                pp = Variable(torch.zeros(1))
                myp = Variable(p, requires_grad=True)
                loss = ((myp - p))
                loss = loss**2
                loss = pp_fisher * loss
                loss = loss.sum()
                losses = losses + loss
            losses = 1000 * losses
            print(losses)
            return (losses)
    
    def dummy_2loss(self,lamda, debug=False):
        try: 
            losses = Variable(torch.zeros(1), requires_grad = True)
            for n,p in self.named_parameters():
                pp_fisher = Variable(torch.ones(1))
                pp = Variable(torch.zeros(1))
                myp = Variable(p, requires_grad=True)
                loss = (pp_fisher*((p - pp)**2)).sum()
                losses = loss + losses
            losses = (lamda/2) * losses
            print(losses)
            return ((losses))
        except: 
            return (Variable(torch.zeros(1)))
    def get_ewc_loss_param(self,name, param, lamda, debug=False):
        try: 
            pp_fisher = Variable(self.fisher_matrix[name])
            pp = Variable(self.prev_parameters[name])
            loss = (pp_fisher*((param - pp)**2)).sum()
            return (Variable((lamda)*(loss), requires_grad=True))
        except: 
            return (Variable(torch.zeros(1)))

    def get_penalty(self):
        try:
            loss = 0.0
            for n,p in self.named_parameters():
                _loss = self.fisher_matrix[n]*((p - self.prev_parameters[n])**2)
                loss += _loss.sum()
            return 1000*loss
        except:
            return 0




