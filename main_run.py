#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# Warning: Do NOT run with args.reload_data = 1 for more than one job simultaneously. 
# The reason is that the code is set to delete any pre-existing files, so this causes a lot of issues
# if multiple runs are trying to reload the same directory. 

# Intend to be identical to main_fed, with the exception that it saves and loads 
# already processed data, so it does not need to be reloaded every time. 
# This is important because some functions in data processing use numpy, 
# which requires CPU, and high-memory CPU machines appear to be in very limited
# supply. I waited an entire day and still, the jobs were in queue for non-iid 
# CIFAR.

# If args.reload_data = 1, reprocesses data (needed when client sizes change). 
# Otherwise, uses saved preprocessed data. 

import os
import pickle
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import time
import resource

import wandb
import math
from PIL import Image

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, AE, LinearM, MLP_classifier
from models.Fed import Fed, param_update_l2_norms
from models.test import test_img, test

from data.dataset import Mnist_data, Stackoverflow, GLDV2, Cifar_data, Cifar100_data, Femnist_data
import logging

# Vision Transformers
from transformers import MobileViTImageProcessor, MobileViTForImageClassification
from transformers import ViTImageProcessor, ViTForImageClassification
from timm.models.vision_transformer import vit_small_patch16_224

from PIL import Image
import requests
import random
from collections import Counter



#######################################
# Move to test

def get_dataloader_datacount(dataloader):
    length = 0
    for _ in dataloader.dataset:
        length += 1
    return length

# Function to get all labels and their statistics from a DataLoader
def get_labels_and_stats(dataloader):
    labels = []
    for _, targets in dataloader:
        labels.extend(targets.numpy())
    labels = np.array(labels)
    label_counts = Counter(labels)

    # Print all labels
    #print("Labels:", labels)
    # Print label statistics
    # print("Label Statistics:", label_counts)
    return labels, label_counts

#######################################

if __name__ == '__main__':
    args = args_parser()
    # print(args)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    random_seed = args.seed #1
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    # print(f"Random seed used: {random_seed}")

    def save_data(objects, filepath):
        if os.path.exists(filepath):
            os.remove(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(objects, f)

    def load_data(filepath):
        retries = 3
        for i in range(retries):
            try:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            except OSError as e:
                if e.errno == 116:  # Stale file handle
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    raise
        raise Exception("Failed to load data after multiple retries due to stale file handle.")
    
    # Is deleted every 90 or so days, so reprocessing data will occasionally be needed
    temp_directory = '/net/scratch/sulee/FedAda2_data/'
    os.makedirs(temp_directory, exist_ok=True)
    data_path = os.path.join(temp_directory, f'{args.dataset}_iid{args.iid}_data.pkl')

    # Check if we need to reload data or use preprocessed data
    if args.reload_data == 1 or not os.path.exists(data_path):
        # Load dataset
        if args.dataset == 'mnist':
            data = Mnist_data(args.num_users, args.iid, args.local_bs)
            in_size = (1, 28, 28)
            # model = AE
        elif args.dataset == 'stackoverflow':
            data = Stackoverflow(args.num_users, args.iid, args.local_bs)
            in_size = (1, 10000)
            # model = LinearM
        elif args.dataset == 'gldv2':
            data = GLDV2(args.num_users, args.iid, args.local_bs)
            in_size = (3, 224, 224)
        elif args.dataset == 'cifar':
            data = Cifar_data(args.num_users, args.iid, args.local_bs)
            in_size = (3, 224, 224) # (3, 32, 32) is resized before feeding ViT
        elif args.dataset == 'cifar100':
            data = Cifar100_data(args.num_users, args.iid, args.local_bs)
            in_size = (3, 224, 224) # (3, 32, 32) is resized before feeding ViT
        elif args.dataset == 'femnist':
            data = Femnist_data(args.num_users, args.iid, args.local_bs)

        # Load the datasets + split into federated users. 
        # testloader is just a single valloader for some datasets and is not used.
        trainloaders, valloaders, testloader = data.load_datasets() #<- OOM kill happens here
        save_data((data, trainloaders, valloaders, testloader), data_path)
    else:
        data, trainloaders, valloaders, testloader = load_data(data_path)

    dataiter = iter(trainloaders[0])
    images, labels = next(dataiter)
    img_size = images[0].shape

    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'AE':
        net_glob = AE().to(args.device)
    elif args.model == 'LinearM':
        net_glob = LinearM().to(args.device)
    elif args.model == 'vit':
        net_glob = vit_small_patch16_224(pretrained=True)
        # Freeze all layers except the final classification layer
        if args.fine_tune_mode == 1:
            for name, param in net_glob.named_parameters():
                param.requires_grad = False
        
        hd = 500
        nl = 12
        if args.dataset == 'cifar':
            net_glob.head = torch.nn.Linear(net_glob.head.in_features, 10) #MLP_classifier(input_dim = net_glob.head.in_features, output_dim = 10, hidden_dim = hd, num_layers = nl) #torch.nn.Linear(net_glob.head.in_features, 10)
        elif args.dataset == 'gldv2':
            net_glob.head = torch.nn.Linear(net_glob.head.in_features, 203) #MLP_classifier(net_glob.head.in_features, 203, hd, nl) #torch.nn.Linear(net_glob.head.in_features, 203)
        elif args.dataset == 'cifar100':
            net_glob.head = torch.nn.Linear(net_glob.head.in_features, 100)
        elif args.dataset == 'femnist':
            net_glob.head = torch.nn.Linear(net_glob.head.in_features, 62)

        if args.fine_tune_mode == 1:
            for name, param in net_glob.head.named_parameters():
                param.requires_grad = True

        # for name, param in net_glob.named_parameters():
        #    if param.requires_grad:
        #        print(f"Trainable parameter: {name} - Shape: {param.shape}")
        #    else:
        #        print(f"Non-trainable parameter: {name} - Shape: {param.shape}")
    else:
        exit('Error: unrecognized model')

    client_eps = args.client_eps
    server_eps = args.server_eps
    server_lr = args.server_lr
    client_lr = args.client_lr
    start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Start memory measurement
    
    if args.model == 'vit' and args.fine_tune_mode==1:
        w_glob = net_glob.head.state_dict()
    else:
        w_glob = net_glob.state_dict()

    if args.ada_mode == -1:
        local_optimizer = "sgd"
        server_optimizer = "Avg"
    else:
        local_optimizer = args.client_opt
        server_optimizer = args.server_opt

    if args.server_preconditioner and local_optimizer == "sgd":
        raise ValueError("Local Opt is sgd, but settings demand server-client preconditioner transmission.")
    if args.server_preconditioner and server_optimizer == "Avg":
        raise ValueError("Server Opt is Avg, but settings demand server-client preconditioner transmission.")

    def lr_schedule_global():
        if args.globalwarmup:
            if args.global_epoch < args.global_warmup_epochs_function_of_global_epoch_step:
                return args.cold_lr_server + (args.server_lr - args.cold_lr_server) * (args.global_epoch / args.global_warmup_epochs_function_of_global_epoch_step)
        return args.server_lr
    
    # Set project name here
    proj_name = args.base_project_name # proj_name = "fn5_"
    proj_name += f"{args.seed} {args.frac}"
    if args.diff_private:
        proj_name += f"DP_"
    proj_name += f"SO{server_optimizer}_"
    proj_name += f"CO{local_optimizer}_"
    if server_optimizer == "Adagrad":
        proj_name += f"beta1_{args.server_beta1_adagrad}_"
    if args.server_preconditioner and args.ada_mode != -1:
        proj_name += "PcndY_"
    else:
        proj_name += "PcndN_"
    proj_name += args.dataset
    proj_name += f"_gEpo{args.epochs}"
    if args.iid:
        proj_name += "_iid"
    else: 
        proj_name += "_noniid"
    if args.model == "vit":
        proj_name += f"_ft{args.fine_tune_mode}_{args.globalwarmup}{args.localwarmup}"
    # proj_name += f"_SC{args.globalwarmup}{args.localwarmup}{args.cold_lr_server}{args.cold_lr_client}till{args.global_warmup_epochs_function_of_global_epoch_step}{args.local_warmup_epochs_function_of_global_epoch_step}"
    # "_SCwarmlr_{args.globalwarmup}_{args.localwarmup}_cold{args.cold_lr_server}_{args.cold_lr_client}_tillGepo{args.global_warmup_epochs_function_of_global_epoch_step}_{args.local_warmup_epochs_function_of_global_epoch_step}"
    if args.ada_mode == 1:
        proj_name += f"_Del{args.update_delay}"

    run_name = f"nClr{args.client_lr}_Slr{args.server_lr}_le{args.local_ep}_bs{args.local_bs}_"
    if args.diff_private:
        run_name += f"cn{args.clip}_{args.sigma}"
    if args.ada_mode > -1:
        if server_optimizer != "Avg":
            run_name += f"Seps{args.server_eps}_"
        if local_optimizer != "sgd":
            run_name += f"Ceps{args.client_eps}"
        if args.ada_mode == 1:
            run_name += f"_Del{args.update_delay}"

    os.environ['WANDB_DIR'] = '/net/scratch/sulee/fedada2_all/wandblog/'
    wandb.init(
            # set the wandb project where this run will be logged
            project=proj_name,
            name=run_name,
            config=args.__dict__
            )
    # wandb.init(
    #         # set the wandb project where this run will be logged
    #         project=proj_name,
    #         name=run_name,
    #         # track hyperparameters and run metadata
    #         config={
    #         "time": "5-8-8pm",
    #         "model": args.model,
    #         "dataset": args.dataset, 
    #         "communication rounds": args.epochs,
    #         "bs": args.local_bs,
    #         "local epochs": args.local_ep,
    #         "seed": args.seed,
    #         "client lr": args.client_lr, 
    #         "server lr": args.server_lr,
    #         "client eps": args.client_eps,
    #         "server eps": args.server_eps,
    #         "ada mode": args.ada_mode,
    #         "frac sampled": args.frac,
    #         "num_tot_user": args.num_users,
    #         "local optimizer": local_optimizer,
    #         "server optimizer": server_optimizer,
    #         "delay parameter": args.update_delay,
    #         "server adam beta1": args.server_beta1,
    #         "server adam beta2": args.server_beta2,
    #         "DP Clip": args.clip,
    #         "DP Sigma": args.sigma,
    #         "server beta1 adagrad only": args.server_beta1_adagrad,
    #         "glob_warmup until": args.global_warmup_epochs_function_of_global_epoch_step,
    #         "loc_warmup until": args.local_warm_backprops,
    #         "transmitting server preconditioners": args.server_preconditioner}
    #         )
    
    num_samples = []
    server_preconditioner = []

    FedStrategy = Fed(w_glob, sigma = args.sigma, clip = args.clip, 
                        batch_size = args.local_bs, diff_private = args.diff_private, 
                        num_all_clients = args.num_users)

   
    # Take this outside the loop
    cli_datanum_train = []
    cli_datanum_val = []
    for client_idx in range(len(trainloaders)):
        cli_datanum_train.append(len(trainloaders[client_idx].dataset)) 
        cli_datanum_val.append(len(valloaders[client_idx].dataset))
    total_number_train = sum(cli_datanum_train)
    total_number_val = sum(cli_datanum_val)

    eval_every_kepochs = args.eval_every_kepochs

    for iter in range(args.epochs):
        args.global_epoch = iter
        tr_loss_locals = []
        tr_accuracy_locals = []
        val_loss_locals = []
        num_samples = []
        val_accuracy_locals = []
        update_params_locals =  []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # print('Selected {} users'.format(len(idxs_users)))

        if iter % eval_every_kepochs == 0:
            
            for client_idx in range(len(trainloaders)):
                tr_loss, tr_accuracy = test(net_glob, trainloaders[client_idx], args)
                val_loss, val_accuracy = test(net_glob, valloaders[client_idx], args)
                tr_loss_locals.append(copy.deepcopy(tr_loss * cli_datanum_train[client_idx] / total_number_train))
                tr_accuracy_locals.append(copy.deepcopy(tr_accuracy * cli_datanum_train[client_idx] / total_number_train))
                val_loss_locals.append(copy.deepcopy(val_loss * cli_datanum_val[client_idx] / total_number_val))
                val_accuracy_locals.append(copy.deepcopy(val_accuracy * cli_datanum_val[client_idx] / total_number_val))            

        for idx in idxs_users:
            result = {}
            local = LocalUpdate(args=args, dataset=trainloaders[idx])
            if args.model == 'LinearM':
                w = local._train_one_epoch_linear(net=copy.deepcopy(net_glob).to(args.device), rounds=iter, ada_mode=args.ada_mode, preconditioner=server_preconditioner)
            elif args.model == 'AE':
                w = local._train_one_epoch_ae(net=copy.deepcopy(net_glob).to(args.device), rounds=iter, ada_mode=args.ada_mode, preconditioner=server_preconditioner)
            elif args.model == 'vit':
                w = local._train_one_epoch_vit(net=copy.deepcopy(net_glob).to(args.device), rounds=iter, ada_mode=args.ada_mode, preconditioner=server_preconditioner)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            
            # Calculate parameter update
            for (key1, param1), (key2, param2) in zip(w.items(), w_glob.items()):
                if key1 == key2:  # make sure the parameters correspond to each other
                    result[key1] = param1 - param2.float().to(args.device)
            
            update_params_locals.append(result)
            num_samples.append(len(trainloaders[idx].dataset))

        l2_list_pseudograds = param_update_l2_norms(update_params_locals)
        l2_list_pseudograds = np.array([tensor.item() for tensor in l2_list_pseudograds])

        # global update
        if args.ada_mode == -1:
            # "FedAvg" implimentation is (allocated local data size/number)  
            # weighted average by default which differs from FedAvg 
            num_samples_ones = [1 for _ in num_samples]
            w_glob = FedStrategy.FedAvg(update_params_locals, num_samples_ones, 1)
            if args.globalwarmup:
                raise ValueError(f"Fedavg but global lr warm up requested")
        elif args.server_opt == "Avg":
            num_samples_ones = [1 for _ in num_samples]
            w_glob = FedStrategy.FedAvg(update_params_locals, num_samples_ones, 1)
            if args.globalwarmup:
                raise ValueError(f"Server optimizer is {args.server_opt} but global lr warm-up requested")
        elif args.server_opt == "weightedAvg":
            w_glob = FedStrategy.FedAvg(update_params_locals, num_samples, lr_schedule_global())
            if args.server_preconditioner:
                raise ValueError("Server Opt is weightedAvg, but settings demand server-client preconditioner transmission.")
        elif args.server_opt == "Adagrad":
            # Server Adagrad 
            w_glob, updated_server_preconditioner = FedStrategy.Fed_serv_adag(num_samples,
                                        update_params_locals,
                                        lr_schedule_global(),
                                        server_eps,
                                        server_preconditioner_initilization=args.server_preconditioner, 
                                        beta1=args.server_beta1_adagrad)
            if args.server_preconditioner:
                server_preconditioner = updated_server_preconditioner
        elif args.server_opt == "Adam_noBC":
            # Server Adam
            w_glob, updated_server_preconditioner = FedStrategy.Fed_serv_adam(num_samples,
                                        update_params_locals,
                                        lr_schedule_global(),
                                        server_eps,
                                        server_preconditioner_initilization=args.server_preconditioner, 
                                        bias_correction = False,
                                        beta1 = args.server_beta1,
                                        beta2 = args.server_beta2)
            if args.server_preconditioner:
                server_preconditioner = updated_server_preconditioner 
        elif args.server_opt == "Adam":
            # Server Adam, bias corrected 
            w_glob, updated_server_preconditioner = FedStrategy.Fed_serv_adam(num_samples,
                                        update_params_locals,
                                        lr_schedule_global(),
                                        server_eps,
                                        server_preconditioner_initilization=args.server_preconditioner, 
                                        bias_correction = True,
                                        beta1 = args.server_beta1,
                                        beta2 = args.server_beta2)
            if args.server_preconditioner:
                server_preconditioner = updated_server_preconditioner

        # -1 is forced FedAvg, 0 allows server/client opt freely chosen, and 1 activates delayed updates on top of 0. 
        if args.ada_mode > 2 or args.ada_mode < -1:
            raise ValueError("Do not recognize ada_mode, vals = -1,0,1 needed.") 

        # copy weight to net_glob
        if args.model == 'vit' and args.fine_tune_mode == 1:
            net_glob.head.load_state_dict(w_glob)
        else:
            net_glob.load_state_dict(w_glob)
            
        if iter % eval_every_kepochs == 0:
            tr_loss_avg = sum(tr_loss_locals) #/ len(tr_loss_locals)
            tr_accuracy_avg = sum(tr_accuracy_locals) #/ len(tr_accuracy_locals)
            val_loss_avg = sum(val_loss_locals) #/ len(val_loss_locals)
            val_accuracy_avg = sum(val_accuracy_locals) #/ len(val_accuracy_locals)
            
            # print("epoch", iter,'users', len(idxs_users), 'training loss', tr_loss_avg, 'training accuracy', tr_accuracy_avg,
            #         'test loss', val_loss_avg, 'test accuracy', val_accuracy_avg,
            #         'pseudograd l2 norms', l2_list_pseudograds, flush=True)

            wandb.log({'training loss': tr_loss_avg, 'training accuracy': tr_accuracy_avg,
                    'test loss': val_loss_avg, 'test accuracy': val_accuracy_avg,
                    'pseudogradient l2 norm mean': np.mean(l2_list_pseudograds),
                    'pseudogradient l2 norm standard deviation': np.std(l2_list_pseudograds)})

                 