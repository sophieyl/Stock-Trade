import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class PolicyGradient(nn.Module):
    def __init__(self,
                 obs_dim = 6,  # observation dimension
                 num_actions = 3,
                 neurons_per_dim=32,  # hidden layer will have obs_dim * neurons_per_dim neurons
                 ):
        super(PolicyGradient,self).__init__()
        self.num_actions = num_actions
        self._hidden_neurons = obs_dim * neurons_per_dim
        self.layer1 = nn.Linear(obs_dim,self._hidden_neurons)
        nn.init.normal(self.layer1.weight,mean=0, std=1./self._hidden_neurons)
        self.layer2 = nn.Linear(self._hidden_neurons,num_actions)
        nn.init.normal(self.layer2.weight,mean=0, std=1./num_actions)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self,x):
        out = self.layer1(x)
        out = F.relu(out)
        logp = self.layer2(out)
        p = F.softmax(logp)
        return p

