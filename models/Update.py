#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import resource
from collections import defaultdict
import gc
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None):
        self.args = args
        if self.args.model == 'LinearM' or self.args.model == 'vit':
            self.loss_func = nn.CrossEntropyLoss()
        elif self.args.model == 'AE':
            self.loss_func = nn.MSELoss()
        self.selected_clients = []
        self.ldr_train = dataset 

    def lr_schedule_local(self, loc_backprop_step):
        warm_end = self.args.local_warm_backprops
        if self.args.localwarmup:
            if loc_backprop_step < warm_end:
                return self.args.cold_lr_client + (self.args.client_lr - self.args.cold_lr_client) * (loc_backprop_step / warm_end)
        return self.args.client_lr

    def _train_one_epoch_linear(self, net, rounds, ada_mode=0, preconditioner=[]):
        net.train()
        # train and update
        if ada_mode <= -1:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.client_lr, momentum=self.args.client_momentum)
        else:
            # Make sgd specifiable as argument. 
            if self.args.client_opt == 'Adagrad':
                optimizer = AdagradOptimizer(net, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner, device=self.args.device)
            elif self.args.client_opt == 'SM3_adagrad':
                optimizer = SM3modOptimizer_adagrad(net, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner, device=self.args.device)
            elif self.args.client_opt == 'Adam':
                optimizer = AdamOptimizer(net, rounds=rounds, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner, device=self.args.device, bias_correct = True)
            elif self.args.client_opt == 'SM3_adam':
                optimizer = SM3Optimizer_adam(net, rounds=rounds, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner, device=self.args.device, bias_correct = True)
            elif self.args.client_opt == 'Adam_noBC':
                optimizer = AdamOptimizer(net, rounds=rounds, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner, device=self.args.device)
            elif self.args.client_opt == 'SM3_adam_noBC':
                optimizer = SM3Optimizer_adam(net, rounds=rounds, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner, device=self.args.device)
            elif self.args.client_opt == 'sgd':
                optimizer = torch.optim.SGD(net.parameters(), lr=self.args.client_lr, momentum=self.args.client_momentum)

        if ada_mode == 1:
            k = self.args.update_delay
        else:
            k = 1
        
        loc_backprop_step = 0 
        for epoch in range(self.args.local_ep):
            for inputs, labels in self.ldr_train:
                if self.args.client_opt == 'sgd':
                    new_lr = self.lr_schedule_local(loc_backprop_step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                else: 
                    optimizer.lr = self.lr_schedule_local(loc_backprop_step)
                inputs = inputs.to(self.args.device)
                labels = labels.to(self.args.device)
                outputs = net(inputs)
                loss = self.loss_func(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                loc_backprop_step += 1
                if ada_mode <= 0:
                    optimizer.step()
                else:
                    optimizer.step(k, epoch)
        return net.state_dict()

    def _train_one_epoch_ae(self, net, rounds, ada_mode=0, preconditioner=[]):
        net.train()
        # train and update
        if ada_mode <= -1:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.client_lr, momentum=self.args.client_momentum)
        else:
            if self.args.client_opt == 'Adagrad':
                optimizer = AdagradOptimizer(net, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner, device=self.args.device)
            elif self.args.client_opt == 'SM3_adagrad':
                optimizer = SM3modOptimizer_adagrad(net, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner, device=self.args.device)
            elif self.args.client_opt == 'Adam':
                optimizer = AdamOptimizer(net, rounds=rounds, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner, device=self.args.device, bias_correct = True)
            elif self.args.client_opt == 'SM3_adam':
                optimizer = SM3Optimizer_adam(net, rounds=rounds, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner, device=self.args.device, bias_correct = True)
            elif self.args.client_opt == 'Adam_noBC':
                optimizer = AdamOptimizer(net, rounds=rounds, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner, device=self.args.device)
            elif self.args.client_opt == 'SM3_adam_noBC':
                optimizer = SM3Optimizer_adam(net, rounds=rounds, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner, device=self.args.device)
            elif self.args.client_opt == 'sgd':
                optimizer = torch.optim.SGD(net.parameters(), lr=self.args.client_lr, momentum=self.args.client_momentum)

        if ada_mode == 1:
            k = self.args.update_delay
        else:
            k = 1

        loc_backprop_step = 0
        for epoch in range(self.args.local_ep):
            for inputs, _ in self.ldr_train:
                if self.args.client_opt == 'sgd':
                    new_lr = self.lr_schedule_local(loc_backprop_step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                else: 
                    optimizer.lr = self.lr_schedule_local(loc_backprop_step)                
                inputs = inputs.to(self.args.device)
                outputs = net(inputs)
                inputs = inputs.reshape(-1, 784)
                # loss = criterion(outputs, labels)
                loss = self.loss_func(outputs, inputs)
                optimizer.zero_grad()
                loss.backward()
                loc_backprop_step += 1
                if ada_mode <= 0:
                    optimizer.step()
                else:
                    #gc.collect()
                    optimizer.step(k, epoch)
        return net.state_dict()
    
    
    def _train_one_epoch_vit(self, net, rounds, ada_mode=0, preconditioner=[]):
        net.train()
        # train and update
        epoch_loss = 0
        total = 0
        correct = 0
        if ada_mode <= -1:
            #print("client side: local sgd")
            optimizer = torch.optim.SGD(net.head.parameters(), lr=self.args.client_lr, momentum=self.args.client_momentum)
        else:
            #print("client side: updates")
            if self.args.client_opt == 'Adagrad':
                optimizer = AdagradOptimizer(net, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner,device=self.args.device)
            elif self.args.client_opt == 'SM3_adagrad':
                if self.args.fine_tune_mode == 1:
                    optimizer = SM3modOptimizer_adagrad(net, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner,device=self.args.device)
                else:
                    optimizer = SM3Optimizer_adagrad_tensor(net, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner,device=self.args.device)
            elif self.args.client_opt == 'Adam':
                optimizer = AdamOptimizer(net, rounds=rounds, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner, device=self.args.device, bias_correct = True)
            elif self.args.client_opt == 'SM3_adam':
                if self.args.fine_tune_mode == 1:
                    optimizer = SM3Optimizer_adam(net, rounds=rounds, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner, device=self.args.device, bias_correct = True)
                else:
                    optimizer = SM3Optimizer_adam_tensor(net, rounds=rounds, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner,device=self.args.device, bias_correct = True)
            elif self.args.client_opt == 'Adam_noBC':
                optimizer = AdamOptimizer(net, rounds=rounds, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner, device=self.args.device)
            elif self.args.client_opt == 'SM3_adam_noBC':
                if self.args.fine_tune_mode == 1:
                    optimizer = SM3Optimizer_adam(net, rounds=rounds, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner, device=self.args.device)
                else:
                    optimizer = SM3Optimizer_adam_tensor(net, rounds=rounds, lr=self.args.client_lr, epsilon=self.args.client_eps, preconditioner=preconditioner,device=self.args.device)
            elif self.args.client_opt == 'sgd':
                optimizer = torch.optim.SGD(net.parameters(), lr=self.args.client_lr, momentum=self.args.client_momentum)

        if ada_mode == 1:
            k = self.args.update_delay
        else:
            k = 1

        loc_backprop_step = 0
        for epoch in range(self.args.local_ep):
            for inputs, labels in self.ldr_train:
                if self.args.client_opt == 'sgd':
                    new_lr = self.lr_schedule_local(loc_backprop_step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                else: 
                    optimizer.lr = self.lr_schedule_local(loc_backprop_step)  
                inputs = inputs.to(self.args.device)
                labels = labels.to(self.args.device)
                # print(f"Input shape is {inputs.shape}")
                outputs = net(inputs)
                loss = self.loss_func(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                loc_backprop_step += 1
                if ada_mode <= 0:
                    optimizer.step()
                else:
                    optimizer.step(k, epoch)

        if self.args.fine_tune_mode == 1:
            return net.head.state_dict()
        return net.state_dict()
    

class AdagradOptimizer:
    def __init__(self, model, lr=0.01, epsilon=0.01, preconditioner=[], device='cpu'):
        self.model = model
        self.lr = lr
        self.epsilon = epsilon
        self.sum_of_squared_gradients = {}
        self.device = device
            
        def preconditioner_zero_initialization():
            for param in self.model.parameters():
                self.sum_of_squared_gradients[param] = torch.zeros_like(param).to(self.device)
                

        def preconditioner_server_initialization(preconditioner):
            count = 0

            for param in self.model.parameters():
                if param.requires_grad: 
                    self.sum_of_squared_gradients[param] = torch.from_numpy(preconditioner[count]).clone().to(self.device)
                    count +=1
        
        if preconditioner != []:
            #print('preconditioner server init', preconditioner[0][0])
            preconditioner_server_initialization(preconditioner)
        else:
            #print('preconditioner zero init')
            preconditioner_zero_initialization()

    #@profile
    def step(self, k=1, i=0): # i is the current local epoch
        for param in self.model.parameters():
            if param.grad is None:
                continue
            if i % k == 0:
                self.sum_of_squared_gradients[param].addcmul_(param.grad, param.grad, value=1)
            denom =  self.sum_of_squared_gradients[param].sqrt().add_(self.epsilon)
            param.data.addcdiv_(param.grad, denom, value=-self.lr)

    def zero_grad(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
                

class SM3modOptimizer_adagrad:
    def __init__(self, model, lr=0.01, epsilon=0.01,preconditioner=[],device='cpu'):
        self.model = model
        self.lr = lr
        self.epsilon = epsilon
        self.accumulator_l = {}
        self.accumulator_r = {}
        self.sum_of_squared_bias_gradients = {}
        self.device = device
        if preconditioner != []:
            self.preconditioner_server_initialization(preconditioner)
        else:
            self.preconditioner_zero_initialization()
        self.bias_preconditioner_zero_initialization()

    
    def check(self):
        for param in self.model.parameters():
            print("Parameter Object:", param)
            print("Parameter Data:", param.data)
            print("Parameter Gradient:", param.grad)
            print("Requires Grad:", param.requires_grad)
            break
    
    def bias_preconditioner_zero_initialization(self):
        count = 0
        for param in self.model.parameters() :
            if param.requires_grad:
                if count % 2 != 0:
                    self.sum_of_squared_bias_gradients[param] = torch.zeros_like(param).to(self.device)
                count+=1
    
    def preconditioner_zero_initialization(self):
        count = 0
        # count variable is created to skip the biases for now
        for param in self.model.parameters() :
            if param.requires_grad:
                if count % 2 == 0:
                    #self.sum_of_squared_gradients[param] = torch.zeros_like(param).to(DEVICE)
                    self.accumulator_l[param] = torch.zeros_like(param.data[:,0]).to(self.device)
                    self.accumulator_r[param] = torch.zeros_like(param.data[0,:]).to(self.device)
                count+=1
    
    def preconditioner_server_initialization(self,preconditioner):
        count = 0
        for param in self.model.parameters() :
            if param.requires_grad:
                if count % 2 == 0:
                    _preconditioner = torch.from_numpy(preconditioner[count]).clone().to(self.device)
                    self.accumulator_l[param] = torch.max(_preconditioner, dim=1).values
                    self.accumulator_r[param] = torch.max(_preconditioner, dim=0).values
                count+=1

    #@profile
    def step(self, k=1, i=0):
        count=0 # this is done to skip the biases update
        for param in self.model.parameters() :
            if param.grad is None:
                continue
            if count % 2 == 0:
                if i % k == 0:
                    self.accumulator_l[param] = self.accumulator_l[param] + torch.max(torch.square(param.grad), dim=1).values  # L: all rows
                    self.accumulator_r[param] = self.accumulator_r[param] + torch.max(torch.square(param.grad), dim=0).values  # R: all columns
                # broadcast one of the accumulators into the shape of the other accumulator as well
                # for v_i do it here:
                broadcasted_accumulator_l = self.accumulator_l[param].sqrt().add_(self.epsilon).view(self.accumulator_l[param].shape[0],1).expand(self.accumulator_l[param].shape[0],self.accumulator_r[param].shape[0])
                denom = torch.min(self.accumulator_r[param].sqrt().add_(self.epsilon),broadcasted_accumulator_l)
                param.data.addcdiv_(param.grad,denom,value=-self.lr)
            else:
                # Update Biases
                if i % k == 0:
                    self.sum_of_squared_bias_gradients[param].addcmul_(param.grad, param.grad, value=1)
                denom = self.sum_of_squared_bias_gradients[param].sqrt()
                param.data.addcdiv_(param.grad,denom.add_(self.epsilon),value=-self.lr) # Update
            count+=1
            
    def zero_grad(self):
        for param in self.model.parameters() :
            if param.grad is not None:
                param.grad.zero_()
                
                
class SM3IIOptimizer_adagrad:
    def __init__(self, model, lr=0.01, epsilon=0.01, preconditioner=[],device='cpu'):
        self.model = model
        self.lr = lr
        self.epsilon = epsilon
        self.sum_of_squared_gradients = {}
        self.accumulator_l = {}
        self.accumulator_r = {}
        self.sum_of_squared_bias_gradients = {}
        self.device = device
        self.preconditioner_zero_initialization()
        self.bias_preconditioner_zero_initialization()

    
    def preconditioner_zero_initialization(self):
        count = 0
        # count variable is created to skip the biases for now
        for param in self.model.parameters() :
            if count % 2 == 0:
                #self.sum_of_squared_gradients[param] = torch.zeros_like(param).to(DEVICE)
                self.accumulator_l[param] = torch.zeros_like(param.data[:,0]).to(self.device)
                self.accumulator_r[param] = torch.zeros_like(param.data[0,:]).to(self.device)
            count+=1
    
    def bias_preconditioner_zero_initialization(self):
        count = 0
        for param in self.model.parameters() :
            if count % 2 != 0:
                self.sum_of_squared_bias_gradients[param] = torch.zeros_like(param).to(self.device)
            count+=1
    def step(self,k=1,i=0):
        count=0 # this is done to skip the biases update
        for param in self.model.parameters() :
            if param.grad is None:
                continue
            if count % 2 == 0:
                # broadcast one of the accumulators into the shape of the other accumulator as well 
                broadcasted_accumulator_l = self.accumulator_l[param].view(self.accumulator_l[param].shape[0],1).expand(self.accumulator_l[param].shape[0],self.accumulator_r[param].shape[0])
                v_t = torch.minimum(self.accumulator_r[param],broadcasted_accumulator_l) + torch.square(param.grad)
                param.data.addcdiv_(param.grad, v_t.sqrt().add_(self.epsilon), value=-self.lr)
                # Update
                if i % k == 0:
                    self.accumulator_l[param] = torch.zeros_like(param.data[:,0]).to(self.device)
                    self.accumulator_r[param] = torch.zeros_like(param.data[0,:]).to(self.device)
                    
                    self.accumulator_l[param] = torch.max(self.accumulator_l[param], torch.max(v_t, dim=1).values)  # L: all rows 
                    self.accumulator_r[param] = torch.max(self.accumulator_r[param], torch.max(v_t, dim=0).values)  # R: all columns
            else:
                # Update Biases
                if i % k == 0:
                    self.sum_of_squared_bias_gradients[param].addcmul_(param.grad, param.grad, value=1)
                denom = self.sum_of_squared_bias_gradients[param].sqrt().add_(self.epsilon)
                param.data.addcdiv_(param.grad,denom,value=-self.lr) # Update
            count+=1
    
    def zero_grad(self):
        for param in self.model.parameters() :
            if param.grad is not None:
                param.grad.zero_()

class SM3Optimizer_adagrad:
    def __init__(self, model, lr=0.01, epsilon=0.01,preconditioner=[],device='cpu'):
        self.model = model
        self.lr = lr
        self.epsilon = epsilon
        self.accumulator_l = {}
        self.accumulator_r = {}
        self.device = device
        self.sum_of_squared_bias_gradients = {}
        if self.model.__class__.__name__ == 'LinearM' or self.model.__class__.__name__ == 'AE':
            self.train_parameters = model.parameters()
        elif self.model.__class__.__name__  == 'MobileViTForImageClassification': 
            self.train_parameters = model.head.parameters()
        elif net.__class__.__name__  == 'VisionTransformer':
            self.train_parameters = model.head.parameters()

        if preconditioner != []:
            self.preconditioner_server_initialization(preconditioner)
        else:
            self.preconditioner_zero_initialization()
        self.bias_preconditioner_zero_initialization()
        
    def check(self):
        for param in self.train_parameters:
            print("Parameter Object:", param)
            print("Parameter Data:", param.data)
            print("Parameter Gradient:", param.grad)
            print("Requires Grad:", param.requires_grad)
            break

    def bias_preconditioner_zero_initialization(self):
        count = 0
        for param in self.train_parameters:
            if count % 2 != 0:
                self.sum_of_squared_bias_gradients[param] = torch.zeros_like(param).to(self.device)
            count+=1

    def preconditioner_zero_initialization(self):
        count = 0
        # count variable is created to skip the biases for now
        for param in self.model.parameters():
            if count % 2 == 0:
                #self.sum_of_squared_gradients[param] = torch.zeros_like(param).to(DEVICE)
                self.accumulator_l[param] = torch.zeros_like(param.data[:,0]).to(self.device)
                self.accumulator_r[param] = torch.zeros_like(param.data[0,:]).to(self.device)
            count+=1

    def preconditioner_server_initialization(self,preconditioner):
        count = 0
        # count variable is created to skip the biases for now
        # How do you initialize this preconditioner?
        for param in self.train_parameters:
            if count % 2 == 0:
                _preconditioner = torch.from_numpy(preconditioner[count]).clone().to(self.device)
                self.accumulator_l[param] =  torch.max(_preconditioner, dim=1).values
                self.accumulator_r[param] =  torch.max(_preconditioner, dim=0).values
            count += 1

    def step(self,k=1,i=0):
        count=0 # this is done to skip the biases update
        for param in self.train_parameters:
            if param.grad is None:
                continue
            if count % 2 == 0:
                if i % k == 0:
                    self.accumulator_l[param] = self.accumulator_l[param] + torch.max(torch.square(param.grad), dim=1).values  # L: all rows 
                    self.accumulator_r[param] = self.accumulator_r[param] + torch.max(torch.square(param.grad), dim=0).values  # R: all columns
                # broadcast one of the accumulators into the shape of the other accumulator as well
                # for v_i do it here:
                broadcasted_accumulator_l = self.accumulator_l[param].view(self.accumulator_l[param].shape[0],1).expand(self.accumulator_l[param].shape[0], self.accumulator_r[param].shape[0])
                min_vals = torch.min(self.accumulator_r[param], broadcasted_accumulator_l)
                denom = min_vals.sqrt().add_(self.epsilon)
                param.data.addcdiv_(param.grad,denom,value=-self.lr) # Update
            else:
                # Update Biases
                if i % k == 0:
                    self.sum_of_squared_bias_gradients[param].addcmul_(param.grad, param.grad, value=1)
                denom = self.sum_of_squared_bias_gradients[param].sqrt().add_(self.epsilon)
                param.data.addcdiv_(param.grad, denom, value=-self.lr) # Update
            count+=1
    
    def zero_grad(self):
        for param in self.train_parameters:
            if param.grad is not None:
                param.grad.zero_()


class AdamOptimizer:
    def __init__(self, model, rounds, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, 
                    preconditioner=[], device='cpu', bias_correct = False):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.device = device
        self.bias_correction1 = 1
        self.bias_correction2 = 1
        self.moving_avg = {}
        self.squared_moving_avg = {}
        self.step_counter = 0
        self.bias_correct = bias_correct

        for param in self.model.parameters():
            self.moving_avg[param] = torch.zeros_like(param, device=self.device)
        
        def preconditioner_zero_initialization():
            for param in self.model.parameters():
                self.squared_moving_avg[param] = torch.zeros_like(param).to(self.device)

        def preconditioner_server_initialization(preconditioner):
            count = 0
            for param in self.model.parameters():
                if param.requires_grad: # my addition
                    self.squared_moving_avg[param] = torch.from_numpy(preconditioner[count]).clone().to(self.device)
                    count +=1
        
        if preconditioner != []:
            preconditioner_server_initialization(preconditioner)
            self.step_counter = rounds
        else:
            preconditioner_zero_initialization()

    #@profile
    def step(self, k=1, i=0):
        self.step_counter += 1
        for param in self.model.parameters():
            if param.grad is None:
                continue

            # Update moving averages of the gradients and the squared gradients
            self.moving_avg[param].mul_(self.beta1).add_(param.grad, alpha=1-self.beta1)
            
            if i % k == 0:  # delayed updates
                self.squared_moving_avg[param].mul_(self.beta2).addcmul_(param.grad, param.grad, value=1-self.beta2)
                if self.bias_correct: 
                    self.bias_correction2 = 1 - self.beta2 ** self.step_counter
            
            # Compute bias-corrected first and second moment estimates
            if self.bias_correct:
                self.bias_correction1 = 1 - self.beta1 ** self.step_counter
            m_hat = self.moving_avg[param] / self.bias_correction1
            v_hat = self.squared_moving_avg[param] / self.bias_correction2

            # Update parameters
            param.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.epsilon), value=-self.lr)

    def zero_grad(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()


class SM3Optimizer_adam:
    def __init__(self, model, rounds, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, 
                preconditioner=[], device='cpu', bias_correct = False):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.device = device
        self.bias_correction1 = 1
        self.bias_correction2 = 1
        self.bias_correction2_b = 1
        self.moving_avg = {}
        self.step_counter = 0
        
        self.accumulator_l = {}
        self.accumulator_r = {}
        self.sum_of_squared_bias_gradients = {}
        self.bias_correct = bias_correct
        
        self.device = device
        for param in self.model.parameters():
            self.moving_avg[param] = torch.zeros_like(param, device=self.device)
            
        if preconditioner != []:
            self.preconditioner_server_initialization(preconditioner)
            self.step_counter = rounds
        else:
            self.preconditioner_zero_initialization()
        self.bias_preconditioner_zero_initialization()
    
    def bias_preconditioner_zero_initialization(self):
        count = 0
        for param in self.model.parameters():
            if param.requires_grad:
                if count % 2 != 0:
                    self.sum_of_squared_bias_gradients[param] = torch.zeros_like(param).to(self.device)
                count+=1
    
    def preconditioner_zero_initialization(self):
        count = 0
        for param in self.model.parameters():
            if param.requires_grad:
                if count % 2 == 0:
                    #self.sum_of_squared_gradients[param] = torch.zeros_like(param).to(DEVICE)
                    self.accumulator_l[param] = torch.zeros_like(param.data[:,0]).to(self.device)
                    self.accumulator_r[param] = torch.zeros_like(param.data[0,:]).to(self.device)
                count+=1
    
    def preconditioner_server_initialization(self,preconditioner):
        count = 0
        for param in self.model.parameters():
            if param.requires_grad:
                if count % 2 == 0:
                    _preconditioner = torch.from_numpy(preconditioner[count]).clone().to(self.device)
                    self.accumulator_l[param] =  torch.max(_preconditioner, dim=1).values
                    self.accumulator_r[param] =  torch.max(_preconditioner, dim=0).values
                count+=1

    #@profile
    def step(self, k=1, i=0):
        self.step_counter += 1
        count = 0
        for param in self.model.parameters():
            if param.grad is None:
                continue
            
            self.moving_avg[param].mul_(self.beta1).add_(param.grad, alpha=1 - self.beta1)
            if self.bias_correct:
                self.bias_correction1 = 1 - self.beta1 ** self.step_counter
            m_hat = self.moving_avg[param] / self.bias_correction1
            
            if count % 2 == 0:
                if i % k == 0:
                    self.accumulator_l[param] = self.beta2 * self.accumulator_l[param] + (1-self.beta2) * torch.max(torch.square(param.grad), dim=1).values  # L: all rows
                    self.accumulator_r[param] = self.beta2 * self.accumulator_r[param] + (1-self.beta2) * torch.max(torch.square(param.grad), dim=0).values  # R: all columns
                    if self.bias_correct:
                        self.bias_correction2 = 1 - self.beta2 ** self.step_counter
                
                broadcasted_accumulator_l = self.accumulator_l[param].sqrt().add_(self.epsilon).view(self.accumulator_l[param].shape[0],1).expand(self.accumulator_l[param].shape[0],self.accumulator_r[param].shape[0])
                v_hat = torch.min(self.accumulator_r[param].sqrt().add_(self.epsilon),broadcasted_accumulator_l) / self.bias_correction2 ** 0.5
                
                param.data.addcdiv_(m_hat,v_hat,value=-self.lr)
            else:
                # Update Biases
                if i % k == 0:
                    if self.bias_correct:
                        self.bias_correction2_b = 1 - self.beta2 ** self.step_counter
                    self.sum_of_squared_bias_gradients[param].mul_(self.beta2).addcmul_(param.grad, param.grad, value=1 - self.beta2)

                denom = self.sum_of_squared_bias_gradients[param] / self.bias_correction2_b
                param.data.addcdiv_(m_hat, denom.sqrt().add_(self.epsilon), value=-self.lr) # Update
            
            count += 1
            
    def zero_grad(self):
        for param in self.model.parameters() :
            if param.grad is not None:
                param.grad.zero_()



class SM3Optimizer_adagrad_tensor:
    # The only difference with the previous implementation is that we need to decide how many accumulators do we keep according to the size of gradient tensor
    def __init__(self, model, lr=0.01, epsilon=0.01,preconditioner=[],device='cpu'):
        self.model = model
        self.lr = lr
        self.epsilon = epsilon
        self.accumulator = defaultdict(lambda: defaultdict(torch.Tensor))
        self.device = device
        self.sum_of_squared_bias_gradients = {}
        self.param_shape = {}
        for param in self.model.parameters():
            self.param_shape[id(param)] = len(param.shape)
        self.preconditioner_zero_initialization()
    
    def preconditioner_zero_initialization(self):
        # count variable is created to skip the biases for now
        for param in self.model.parameters():
            if param.requires_grad:
                param_id = id(param)
                #print(id(param))
                if self.param_shape[param_id] > 1:
                    c = 0
                    for i in param.shape:
                        self.accumulator[param_id][f'{param_id}_{c}'] = torch.zeros(i).to(self.device)
                        c+=1
                else:
                    self.sum_of_squared_bias_gradients[param] = torch.zeros_like(param).to(self.device)
    #@profile
    def step(self,k=1,i=0):
        for param in self.model.parameters():
            if param.grad is None:
                continue
            param_id = id(param)
            if self.param_shape[param_id] > 1:
                if i % k == 0:
                    c = 0
                    for _ in param.shape:
                        # Calculate the dimensions along which you want to calculate the max
                        max_dims = [j for j in range(len(param.shape)) if j != c]
                        self.accumulator[param_id][f'{param_id}_{c}'] =  self.accumulator[param_id][f'{param_id}_{c}'] + torch.amax(torch.square(param.grad), dim=max_dims)
                        c+=1
                # broadcast one of the accumulators into the shape of the other accumulator
                # for v_i do it here:
                broadcasted_accumulator = self.broadcasting_accumulators(self.accumulator[param_id])
                min_vals = broadcasted_accumulator[0]
                for l in range(1,len(broadcasted_accumulator)):
                    min_vals = torch.min(broadcasted_accumulator[l], min_vals)
                    # Calculate the denominator
                denom = min_vals.sqrt().add_(self.epsilon)
                param.data.addcdiv_(param.grad,denom,value=-self.lr) # Update
            else:
                # Update Biases
                if i % k == 0:
                    self.sum_of_squared_bias_gradients[param].addcmul_(param.grad, param.grad, value=1)
                denom = self.sum_of_squared_bias_gradients[param].sqrt().add_(self.epsilon)
                param.data.addcdiv_(param.grad,denom,value=-self.lr) # Update
    
    def zero_grad(self):
        """_summary_
        """
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
                
    def broadcasting_accumulators(self,accumulator):
        """_summary_

        Args:
            accumulator (dict): This contains 1D array of each gradient dimension

        Returns:
            broadcasted_accumulator: This contains the broadcasted version of each 1D accumulator
        """
        rank = len(accumulator.keys())
        broadcasted_accumulator = []
        for i, key in enumerate(accumulator.keys()):
            mod_shape = [1]* i + [accumulator[key].shape[0]] + [1]*(rank-i-1)
            broadcasted_accumulator.append(torch.reshape(accumulator[key],mod_shape))
        return broadcasted_accumulator
    
class SM3Optimizer_adam_tensor:
    # The only difference with the previous implementation is that we need to decide how many accumulators do we keep according to the size of gradient tensor
    def __init__(self, model, rounds, lr=0.01, beta1=0.9, beta2=0.999,epsilon=0.01,
                preconditioner=[],device='cpu', bias_correct = False):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.device = device
        self.bias_correction1 = 1
        self.bias_correction2 = 1
        self.bias_correction2_b = 1
        self.moving_avg = {}
        self.step_counter = 0
        self.accumulator = defaultdict(lambda: defaultdict(torch.Tensor))
        self.sum_of_squared_bias_gradients = {}
        self.param_shape = {}
        self.bias_correct = bias_correct
        self.SM3_param_count = 0
        self.non_SM3_param_count = 0

        for param in self.model.parameters():
            self.param_shape[id(param)] = len(param.shape)
        self.preconditioner_zero_initialization()
    
    def preconditioner_zero_initialization(self):
        # count variable is created to skip the biases for now
        for param in self.model.parameters():
            if param.requires_grad:
                param_id = id(param)
                #print(id(param))
                if self.param_shape[param_id] > 1:
                    c = 0
                    for i in param.shape:
                        self.accumulator[param_id][f'{param_id}_{c}'] = torch.zeros(i).to(self.device)
                        c+=1
                else:
                    self.sum_of_squared_bias_gradients[param_id] = torch.zeros_like(param).to(self.device)
                self.moving_avg[param_id] = torch.zeros_like(param).to(self.device)
    #@profile
    def step(self,k=1,i=0):
        self.step_counter += 1
        for param in self.model.parameters():
            if param.grad is None:
                continue
            param_id = id(param)
            self.moving_avg[param_id].mul_(self.beta1).add_(param.grad, alpha=1 - self.beta1)
            if self.bias_correct:
                self.bias_correction1 = 1 - self.beta1 ** self.step_counter
            m_hat = self.moving_avg[param_id] / self.bias_correction1
            
            if self.param_shape[param_id] > 1:
                
                if i % k == 0:
                    c = 0
                    for _ in param.shape:
                        # Calculate the dimensions along which you want to calculate the max
                        max_dims = [j for j in range(len(param.shape)) if j != c]
                        self.accumulator[param_id][f'{param_id}_{c}'] =  self.beta2 * self.accumulator[param_id][f'{param_id}_{c}'] + (1-self.beta2) * torch.amax(torch.square(param.grad), dim=max_dims)
                        c+=1
                    if self.bias_correct:
                        self.bias_correction2 = 1 - self.beta2 ** self.step_counter
                # broadcast one of the accumulators into the shape of the other accumulator

                broadcasted_accumulator = self.broadcasting_accumulators(self.accumulator[param_id])
                min_vals = broadcasted_accumulator[0]
                for l in range(1,len(broadcasted_accumulator)):
                    min_vals = torch.min(broadcasted_accumulator[l], min_vals)
                    # Calculate the denominator
                denom = min_vals.sqrt().add_(self.epsilon) / self.bias_correction2 ** 0.5

                param.data.addcdiv_(param.grad,denom,value=-self.lr) # Update
            
            else:
                # Update Biases
                if i % k == 0:
                    if self.bias_correct:
                        self.bias_correction2_b = 1 - self.beta2 ** self.step_counter
                    self.sum_of_squared_bias_gradients[param_id].mul_(self.beta2).addcmul_(param.grad, param.grad, value=1 - self.beta2)
                denom = self.sum_of_squared_bias_gradients[param_id] /self.bias_correction2_b
                param.data.addcdiv_(m_hat,denom.sqrt().add_(self.epsilon),value=-self.lr) # Update

    def zero_grad(self):
        """_summary_
        """
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
                
    def broadcasting_accumulators(self,accumulator):
        """_summary_

        Args:
            accumulator (dict): This contains 1D array of each gradient dimension

        Returns:
            broadcasted_accumulator: This contains the broadcasted version of each 1D accumulator
        """
        rank = len(accumulator.keys())
        broadcasted_accumulator = []
        for i, key in enumerate(accumulator.keys()):
            mod_shape = [1]* i + [accumulator[key].shape[0]] + [1]*(rank-i-1)
            broadcasted_accumulator.append(torch.reshape(accumulator[key],mod_shape))
        return broadcasted_accumulator