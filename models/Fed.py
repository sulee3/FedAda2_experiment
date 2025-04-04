#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np


class Fed():
    def __init__(self, w_glob, sigma = 1, clip = 1, batch_size = 32, diff_private = 0, 
                num_all_clients = 1) -> None:
        w_glob = {k: v.cpu().numpy() for k, v in w_glob.items()}
        self.sum_delta_square = [np.zeros_like(x) for x in w_glob.values()]
        self.global_params = [np.array(x) for x in w_glob.values()]
        self.sigma = sigma
        self.clip = clip
        self.batch_size = batch_size
        self.diff_private = diff_private
        self.num_all_clients = num_all_clients
        # For Server Adam 
        self.m_t = [np.zeros_like(x) for x in w_glob.values()]
        self.v_t = [np.zeros_like(x) for x in w_glob.values()]
        self.t = 0 # For bias correction Adam
    
    def FedAvg(self, param_updates, num_samples, lr):
        aggregated_delta = self.aggregate(param_updates, num_samples)
        self.global_params = [np.array(np.add(x, lr * y)) for (x, y) in zip(self.global_params, aggregated_delta)]
        parameters_aggregated = {k: torch.from_numpy(v) for k, v in zip(param_updates[0].keys(), self.global_params)}
        return parameters_aggregated

    def Fed_serv_adag(self, num_samples, param_updates, lr, tau, server_preconditioner_initilization=False, beta1=0):
        aggregated_delta = self.aggregate(param_updates, num_samples)
        #m_t = aggregated_delta # Implement EMA momentum for first moment for server Adagrad
        self.m_t = [np.add(np.multiply(beta1, m), np.multiply((1 - beta1), g)) for m, g in zip(self.m_t, aggregated_delta)]
        self.sum_delta_square = [np.array(np.add(x, np.square(y))) for (x, y) in zip(self.sum_delta_square, aggregated_delta)]
        denom = [np.array(np.add(np.sqrt(x), tau)) for x in self.sum_delta_square]
        b = [np.array(lr * np.divide(x, y)) for (x, y) in zip(self.m_t, denom)]
        parameters_aggregated = [np.array(np.add(x, y)) for (x, y) in zip(self.global_params, b)]
        self.global_params = parameters_aggregated
        if server_preconditioner_initilization:
            preconditioner = self.sum_delta_square
        else:
            preconditioner = []
        parameters_aggregated = {k: torch.from_numpy(v) for k, v in zip(param_updates[0].keys(), parameters_aggregated)}
        return parameters_aggregated, preconditioner

    def Fed_serv_adam(self, num_samples, param_updates, lr, tau, server_preconditioner_initilization=False, 
                        bias_correction=True, beta1=0.9, beta2=0.999):
        self.t += 1
        aggregated_delta = self.aggregate(param_updates, num_samples)
        #self.m_t = [beta1 * m + (1 - beta1) * g for m, g in zip(self.m_t, aggregated_delta)]
        #self.v_t = [beta2 * v + (1 - beta2) * (g ** 2) for v, g in zip(self.v_t, aggregated_delta)]
        self.m_t = [np.add(np.multiply(beta1, m), np.multiply((1 - beta1), g)) for m, g in zip(self.m_t, aggregated_delta)]
        self.v_t = [np.add(np.multiply(beta2, v), np.multiply((1 - beta2), np.square(g))) for v, g in zip(self.v_t, aggregated_delta)]

        if bias_correction:
            # m_hat = [m / (1 - beta1 ** self.t) for m in self.m_t]
            # v_hat = [v / (1 - beta2 ** self.t) for v in self.v_t]
            m_hat = [np.divide(m, (1 - beta1 ** self.t)) for m in self.m_t]
            v_hat = [np.divide(v, (1 - beta2 ** self.t)) for v in self.v_t]
        else:
            m_hat = self.m_t
            v_hat = self.v_t
        # denom = [np.sqrt(v) + tau for v in v_hat]
        # step = [m / d for m, d in zip(m_hat, denom)]
        denom = [np.add(np.sqrt(v), tau) for v in v_hat]
        step = [np.divide(m, d) for m, d in zip(m_hat, denom)]

        # Use + sign, not -, following Adaptive Federated Optimization Paper
        # parameters_aggregated = [p + lr * s for p, s in zip(self.global_params, step)]
        parameters_aggregated = [np.add(p, lr * s) for p, s in zip(self.global_params, step)]

        self.global_params = parameters_aggregated
        if server_preconditioner_initilization:
            preconditioner = self.v_t
        else:
            preconditioner = []
        parameters_aggregated = {k: torch.from_numpy(v) for k, v in zip(param_updates[0].keys(), parameters_aggregated)}
        return parameters_aggregated, preconditioner

    def aggregate(self, w, num_samples):
        if self.diff_private == 1:
            # l2-based clipping/rescaling
            def l2_norm_clip(param_updates, clip_value):
                for client_number in range(len(param_updates)):
                    # Concatenate all gradients for the client
                    flat_grads = []
                    for k in param_updates[client_number].keys():
                        flat_grads.append(param_updates[client_number][k].view(-1))
                    concatenated_grads = torch.cat(flat_grads)
                    
                    # Compute the L2 norm
                    l2_norm = torch.norm(concatenated_grads)
                    
                    # If the L2 norm exceeds the clip value, rescale the gradients
                    if l2_norm > clip_value:
                        scale_factor = clip_value / l2_norm
                        for k in param_updates[client_number].keys():
                            param_updates[client_number][k] *= scale_factor
                
                return param_updates
            
            w = l2_norm_clip(w, self.clip)

            # Averaging uniformly over all clients, w is pseudograd collection
            # w = param_updates (only participating client updates appended to [])
            # Deep copy in order to keep the keys 
            w_avg = copy.deepcopy(w[0])
            w_device = w_avg[list(w_avg.keys())[0]].device
            # Exclude the first client from the summation due to already deepcopying
            num_clients = range(1, len(w))

            for k in w_avg.keys():
                for i in num_clients:
                    w_avg[k] += w[i][k]
                w_avg[k] = torch.div(w_avg[k], len(w))
                w_avg[k] += torch.div(torch.normal(0, self.sigma * self.clip, size=w_avg[k].shape, device=w_device),len(w))
            w_avg = [x.cpu().numpy() for x in w_avg.values()]

        elif self.diff_private == 0:
            # Weighted average over samples allocated to clients
            num_examples_total = sum(num_samples)
            w_avg = copy.deepcopy(w[0])
            for i in w_avg.keys():
                w_avg[i] = num_samples[0] * w_avg[i]
            num_clients = range(1, len(w))
            new_num_samples = num_samples[1:]
            for k in w_avg.keys():
                for i,samples in zip(num_clients,new_num_samples):
                    w_avg[k] += samples * w[i][k]
                w_avg[k] = torch.div(w_avg[k], num_examples_total)
            w_avg = [x.cpu().numpy() for x in w_avg.values()]
        else: 
            raise ValueError("Unrecognized number for diff_privacy (1=Y, 0=N)")
        return w_avg


def param_update_l2_norms(param_updates):
    l2_norm_list = [] 
    for client_number in range(len(param_updates)):
        # Concatenate all gradients for the client
        flat_grads = []
        for k in param_updates[client_number].keys():
            flat_grads.append(param_updates[client_number][k].view(-1))
        concatenated_grads = torch.cat(flat_grads)
        
        # Compute the L2 norm
        l2_norm = torch.norm(concatenated_grads)
        l2_norm_list.append(l2_norm.cpu()) 
        #l2_norm_list.append([np.asarray(l2_norm.cpu())])    
    return l2_norm_list