#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # Differential Privacy arguments (add default values if issues arise)
    parser.add_argument('--sigma', type=float, help='sigma for differential privacy')
    parser.add_argument('--clip', type=float, help='clipping value for differential privacy')
    parser.add_argument('--diff_private', type=int, default=0, help='Differential privacy Y (1) or N (0)')

    # learning warm-up and decay
    parser.add_argument('--localwarmup', type=int, default=0, help='0 for no client warm-up and 1 for client warm-up')
    parser.add_argument('--globalwarmup', type=int, default=0, help='0 for no server warm-up and 1 for server warm-up')
    parser.add_argument('--cold_lr_client', type=float, default=0.0, help='initial lr for client before warm-up')
    parser.add_argument('--cold_lr_server', type=float, default=0.0, help='initial lr for server before warm-up')
    parser.add_argument('--local_warm_backprops', default=1, type=int, help='warmup ends after this many local backpropagation steps')
    parser.add_argument('--global_warmup_epochs_function_of_global_epoch_step', default=1, type=int, help='warmup ends after this many global epochs')
    parser.add_argument('--global_epoch', type=int, default=1, help='will be synchronized to global round epoch in main_fed.py file')

    # federated arguments
    parser.add_argument('--epochs', type=int, default=20, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")

    # model arguments
    parser.add_argument('--model', type=str, default='vit', help='model name')
    parser.add_argument('--fine_tune_mode', type=int, default=0, help='This is decides whether to fine tune the models last layer or all layers')
    
    # other arguments
    parser.add_argument('--base_project_name', type=str, default='BaseDbug', help="Base Project Name for Wandb")
    parser.add_argument('--reload_data', type=int, default=1, help="Reloads & Reprocesses data (1) or just uses previously processed files (0)")
    parser.add_argument('--vis_gradstats_atepoch', type=str, default="", help="View histograms of client pseudogradient l2 norm")
    parser.add_argument('--eval_every_kepochs', type=int, default=1, help="Evaluate data every k epochs to save time")
    parser.add_argument('--dataset', type=str, default='femnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--ada_mode', type=int, default=0, help='adaptivity mode [-1,0,1,2]')
    parser.add_argument('--update_delay', type=int, default=2, help='delay of local preconditioner update (in terms of local epochs)')
    parser.add_argument('--server_preconditioner', type=int, default=0, help='use server preconditioner')
    parser.add_argument('--client_opt', type=str, default='SM3_adam_noBC', help='client optimizer')
    parser.add_argument('--server_opt', type=str, default='Avg', help='server optimizer')
    
    # Client HPs
    parser.add_argument('--client_lr', type=float, default=0.001, help="learning rate for client optimizer")
    parser.add_argument('--client_eps', type=float, default=0.00001, help='epsilon for client optimizer')
    
    #Server HPs
    parser.add_argument('--server_lr', type=float, default=0.1, help='learning rate for server optimizer')
    parser.add_argument('--server_eps', type=float, default=0.001, help='epsilon for server optimizer')
    parser.add_argument('--server_beta1', type=float, default=0.9, help='beta1 for Adam, first moment EMA')
    parser.add_argument('--server_beta2', type=float, default=0.999, help='beta2 for EMA Adam')
    parser.add_argument('--server_beta1_adagrad', type=float, default=0, help='beta1 for Adagrad, first moment EMA')
    
    # for ada mode 1 : server_eps = 1e-5, client_eps = 1e-3
    # for ada mode 2 : server_eps = 1e-5, client_eps = 1e-3
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--client_momentum', type=float, default=0, help="Client SGD momentum (default: 0)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    
    args = parser.parse_args()
    return args
