#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class MLP_classifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(MLP_classifier, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


class LinearM(nn.Module):
    def __init__(self) -> None:
        super(LinearM, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(10000, 500)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return  logits

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.npw1 = np.zeros((784,1000))
        self.npw2 = np.zeros((1000,500))
        self.npw3 = np.zeros((500,250))
        self.npw4 = np.zeros((250,30))
        self.npw5 = np.zeros((30,250))
        self.npw6 = np.zeros((250,500))
        self.npw7 = np.zeros((500,1000))
        self.npw8 = np.zeros((1000,784))
        self.initialize_weights()
        self.autoencoder = nn.Sequential(
            nn.Linear(784, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 250),
            nn.LeakyReLU(),
            nn.Linear(250, 30),
            nn.Linear(30, 250),
            nn.LeakyReLU(),
            nn.Linear(250, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 784),
            nn.Sigmoid()
        )
        with torch.no_grad():
            # Weights
            self.autoencoder[0].weights  = self.npw1
            self.autoencoder[2].weights  = self.npw2
            self.autoencoder[4].weights  = self.npw3
            self.autoencoder[6].weights  = self.npw4
            self.autoencoder[7].weights  = self.npw5
            self.autoencoder[9].weights  = self.npw6
            self.autoencoder[11].weights = self.npw7
            self.autoencoder[13].weights = self.npw8
            # Biases
            self.autoencoder[0].bias.zero_()
            self.autoencoder[2].bias.zero_()
            self.autoencoder[4].bias.zero_()
            self.autoencoder[6].bias.zero_()
            self.autoencoder[7].bias.zero_()
            self.autoencoder[9].bias.zero_()
            self.autoencoder[11].bias.zero_()
            self.autoencoder[13].bias.zero_()
    def initialize_weights(self):
        """_summary_
        This function is to initialize the weights and biases of the autoencoder.
        """
        # Initialize Weights
        for i in range(1000):
            idx = np.random.choice(784, 15)
            self.npw1[idx,i] = np.random.randn(15)
        for i in range(500):
            idx = np.random.choice(1000, 15)
            self.npw2[idx,i] = np.random.randn(15)
        for i in range(250):
            idx = np.random.choice(500, 15)
            self.npw3[idx,i] = np.random.randn(15)
        for i in range(30):
            idx = np.random.choice(250, 15)
            self.npw4[idx,i] = np.random.randn(15)
        for i in range(250):
            idx = np.random.choice(30, 15)
            self.npw5[idx,i] = np.random.randn(15)
        for i in range(500):
            idx = np.random.choice(250, 15)
            self.npw6[idx,i] = np.random.randn(15)
        for i in range(1000):
            idx = np.random.choice(500, 15)
            self.npw7[idx,i] = np.random.randn(15)
        for i in range(784):
            idx = np.random.choice(1000, 15)
            self.npw8[idx,i] = np.random.randn(15)

    def forward(self, x):
        x = x.view(-1, 784)  # (batch size, 784)
        output = self.autoencoder(x)
        return output  # (batch size, 784)



class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
