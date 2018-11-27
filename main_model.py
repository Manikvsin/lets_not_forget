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

class ewc_MLP(nn.Module):
    def __init__(self, layer_sizes,dropout = False, use_gpu=False):
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
        self.prev_parameters = {n:p for n,p in self.named_parameters()}

    def get_fisher_matrix(self):
        return self.fisher_matrix

    def get_ewc_loss(self,lamda):
        try:
            losses = torch.zeros(1)
            for n,p in self.named_parameters:
                pp_fisher = self.fisher_matrix[n]
                pp = self.prev_parameters[n]
                loss = (pp_fisher*((p - pp)**2)).sum()
                losses = losses + loss
            return (Variable((lamda/2)*(losses)))
        except: 
            return (Variable(torch.zeros(1)))








