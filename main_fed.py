#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from models.Nets import MLP, CNNMnist, CNNCifar, AE, LinearM
from models.Fed import Fed, param_update_l2_norms
from models.test import test_img, test
#from models.central_test import Test

#from torch.utils.tensorboard import SummaryWriter
from data.dataset import Mnist_data, Stackoverflow, GLDV2, Cifar_data
#from torchsummary import summary
import logging

# Vision Transformers
from transformers import MobileViTImageProcessor, MobileViTForImageClassification
from transformers import ViTImageProcessor, ViTForImageClassification
from timm.models.vision_transformer import vit_small_patch16_224

from PIL import Image
import requests

if __name__ == '__main__':
    args = args_parser()
    print(args)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # Set up logging
    #log_name = f'log_{args.dataset}_{args.model}_C{args.num_users}_iid{args.iid}_bs{args.local_bs}_ada{args.ada_mode}_frac{args.frac}.txt'
    log_name = f'log_[{args.ada_mode}].txt'
    logging.basicConfig(filename=log_name, level=logging.INFO, 
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    # load dataset and split users
    if args.dataset == 'mnist':
        data = Mnist_data(args.num_users, args.iid, args.local_bs)
        in_size = (1, 28, 28)
        #model = AE
    elif args.dataset == 'stackoverflow':
        data = Stackoverflow(args.num_users, args.iid, args.local_bs)
        in_size = (1, 10000)
        #model = LinearM
    elif args.dataset == 'gldv2':
        data = GLDV2(args.num_users, args.iid, args.local_bs)
        in_size = (3,224,224)
    elif args.dataset == 'cifar':
        data = Cifar_data(args.num_users, args.iid, args.local_bs) # Cifar_data_my, Cifar_data
        in_size = (3,224,224) #(3, 32, 32)

    trainloaders, valloaders, testloader = data.load_datasets() #<- OOM kill happens here
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
        # # Get the pretrained ViT
        # feature_extractor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
        # net_glob = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small").to(args.device)
        # Freeze all layers except the final classification layer
        if args.fine_tune_mode == 1:
            for name, param in net_glob.named_parameters():
                param.requires_grad = False
        
        if args.dataset == 'cifar':
            net_glob.head = torch.nn.Linear(net_glob.head.in_features, 10)
        elif args.dataset == 'gldv2':
            net_glob.head = torch.nn.Linear(net_glob.head.in_features, 203)
        
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
    #summary(net_glob, in_size)

    #start_time = time.time()
    client_eps = args.client_eps
    server_eps = args.server_eps
    server_lr = args.server_lr
    client_lr = args.client_lr
    start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Start memory measurement
    
    if args.model == 'vit' and args.fine_tune_mode==1:
        w_glob = net_glob.head.state_dict()
    else:
        w_glob = net_glob.state_dict()

    
    # Mine
    if args.ada_mode <= -1:
        local_optimizer = "sgd"
        server_optimizer = "Avg"
    else:
        local_optimizer = args.client_opt
        server_optimizer = args.server_opt

    if args.server_preconditioner and local_optimizer == "sgd":
        raise ValueError("Local Opt is sgd, but settings demand server-client preconditioner transmission.")
    if args.server_preconditioner and server_optimizer == "Avg":
        raise ValueError("Server Opt is Avg, but settings demand server-client preconditioner transmission.")
    if args.localwarmup and local_optimizer == "sgd":
        raise ValueError(f"Local optimizer is {local_optimizer} but local lr warm up requested")

    def lr_schedule_global():
        if args.globalwarmup:
            if args.global_epoch < args.global_warmup_epochs_function_of_global_epoch_step:
                return args.cold_lr_server + (args.server_lr - args.cold_lr_server) * (args.global_epoch / args.global_warmup_epochs_function_of_global_epoch_step)
        return args.server_lr
    
    # Set project name here
    proj_name = "RTest_"
    if args.diff_private:
        proj_name += "DiffPriv"
    proj_name += f"SerOpt{server_optimizer}_"
    proj_name += f"CliOpt{local_optimizer}_"
    if server_optimizer == "Adagrad":
        proj_name += f"beta1_{args.server_beta1_adagrad}_"
    if args.server_preconditioner and args.ada_mode != -1:
        proj_name += "PrecondY_"
    else:
        proj_name += "PrecondN_"
    proj_name += args.dataset
    proj_name += f"_gblEpo{args.epochs}"
    if args.iid:
        proj_name += "_iid"
    if args.model == "vit":
        proj_name += f"_ftune{args.fine_tune_mode}"
    proj_name += f"_locEpoch{args.local_ep}_serv_cli_warm_{args.globalwarmup}_{args.localwarmup}_w_cold_lr{args.cold_lr_server}_{args.cold_lr_client}_till{args.global_warmup_epochs_function_of_global_epoch_step}_{args.local_warmup_epochs_function_of_global_epoch_step}"


    run_name = f"Clr{args.client_lr}_Slr{args.server_lr}_locEpoch{args.local_ep}_bs{args.local_bs}_"
    if args.ada_mode > -1:
        if server_optimizer != "Avg":
            run_name += f"Ceps{args.client_eps}_"
        if local_optimizer != "sgd":
            run_name += f"Seps{args.server_eps}"
        if args.ada_mode == 1:
            run_name += f"_Delay{args.update_delay}"

    wandb.init(
            # set the wandb project where this run will be logged
            project=proj_name,
            name=run_name,
            # track hyperparameters and run metadata
            config={
            "time": "5-8-8pm",
            "model": args.model,
            "dataset": args.dataset, 
            "communication rounds": args.epochs,
            "bs": args.local_bs,
            "local epochs": args.local_ep,
            "client lr": args.client_lr, 
            "server lr": args.server_lr,
            "client eps": args.client_eps,
            "server eps": args.server_eps,
            "ada mode": args.ada_mode,
            "local optimizer": local_optimizer,
            "server optimizer": server_optimizer,
            "delay parameter": args.update_delay,
            "server adam beta1": args.server_beta1,
            "server adam beta2": args.server_beta2,
            "DP Clip": args.clip,
            "DP Sigma": args.sigma,
            "server beta1 adagrad only": args.server_beta1_adagrad,
            "transmitting server preconditioners": args.server_preconditioner}
            )
    
    # training
    loss_train = []
    loss_test  = []
    acc_test = []
    acc_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    num_samples = []
    server_preconditioner = []

    FedStrategy = Fed(w_glob, sigma = args.sigma, clip = args.clip, 
                        batch_size = args.local_bs, diff_private = args.diff_private, 
                        num_all_clients = args.num_users)

    # for idx in range(10):
    #     #print(trainloaders[idx])
    #     for inputs, labels in trainloaders[idx]:
    #         print(idx, labels)

    # For W&B logging (give to options)
    avg_last_n_epochs = args.avg_last_n_epochs
    tr_loss_hist = []
    tr_accuracy_hist = []
    val_loss_hist = []
    val_accuracy_hist = []

    # Visualize pseudogradient histogram at these epochs. 
    vis_gradstats_atepoch = [int(x) for x in args.vis_gradstats_atepoch.split(',')]
    max_epoch = max(vis_gradstats_atepoch)
    # Weights and Biases requires a dictionary to log
    histogram_dict = {}
    length_matrix = []

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
        print('select {} users'.format(len(idxs_users)))
        for idx in idxs_users:
            result = {}
            tr_loss, tr_accuracy = test(net_glob, trainloaders[idx], args)
            val_loss, val_accuracy = test(net_glob, valloaders[idx], args)
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
            tr_loss_locals.append(copy.deepcopy(tr_loss))
            tr_accuracy_locals.append(copy.deepcopy(tr_accuracy))
            val_loss_locals.append(copy.deepcopy(val_loss))
            val_accuracy_locals.append(copy.deepcopy(val_accuracy))

        l2_list_pseudograds = param_update_l2_norms(update_params_locals)
        l2_list_pseudograds = np.array([tensor.item() for tensor in l2_list_pseudograds])

        # global update
        if args.ada_mode == -1:
            # "FedAvg" is (local data count) weighted average by default, which strictly
            # speaking differs from the original FedAvg 
            num_samples_ones = [1 for _ in num_samples]
            w_glob = FedStrategy.FedAvg(update_params_locals, num_samples_ones, 1)
            if args.globalwarmup or args.localwarmup:
                raise ValueError(f"Fedavg but global or local lr warm up requested")
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
            
        # print loss
        tr_loss_avg = sum(tr_loss_locals) / len(tr_loss_locals)
        tr_accuracy_avg = sum(tr_accuracy_locals) / len(tr_accuracy_locals)
        val_loss_avg = sum(val_loss_locals) / len(val_loss_locals)
        val_accuracy_avg = sum(val_accuracy_locals) / len(val_accuracy_locals)
        
        print("epoch ", iter, 'training loss', tr_loss_avg, 'training accuracy', tr_accuracy_avg,
                'test loss', val_loss_avg, 'test accuracy', val_accuracy_avg,
                'pseudograd l2 norms', l2_list_pseudograds, flush=True)
        
        # Append current epoch's metrics to history lists
        tr_loss_hist.append(tr_loss_avg)
        tr_accuracy_hist.append(tr_accuracy_avg)
        val_loss_hist.append(val_loss_avg)
        val_accuracy_hist.append(val_accuracy_avg)
        # print(len(tr_loss_hist))

        # Calculate averages over the last n epochs if we have enough data
        if len(tr_loss_hist) == avg_last_n_epochs and iter != 0:
            tr_loss_avg_last_n = sum(tr_loss_hist) / avg_last_n_epochs
            tr_accuracy_avg_last_n = sum(tr_accuracy_hist) / avg_last_n_epochs
            val_loss_avg_last_n = sum(val_loss_hist) / avg_last_n_epochs
            val_accuracy_avg_last_n = sum(val_accuracy_hist) / avg_last_n_epochs
            
            #print(f"Logging to wandb at epoch {iter}")
            wandb.log({'training loss': tr_loss_avg_last_n, 'training accuracy': tr_accuracy_avg_last_n,
                    'test loss': val_loss_avg_last_n, 'test accuracy': val_accuracy_avg_last_n,
                    'pseudogradient l2 norm mean': np.mean(l2_list_pseudograds),
                    'pseudogradient l2 norm standard deviation': np.std(l2_list_pseudograds)})
            
            # Reset the history lists after logging
            tr_loss_hist = []
            tr_accuracy_hist = []
            val_loss_hist = []
            val_accuracy_hist = []
        elif iter == 0 or avg_last_n_epochs == 1:
            # Initial logging for the zeroth iteration
            wandb.log({'training loss:': tr_loss_avg, 'training accuracy': tr_accuracy_avg,
                    'test loss': val_loss_avg, 'test accuracy': val_accuracy_avg,
                    'pseudogradient l2 norm mean': np.mean(l2_list_pseudograds),
                    'pseudogradient l2 norm standard deviation': np.std(l2_list_pseudograds)})
        # else:
        #     print(f"Not enough data to log averages to wandb at epoch {iter}")

        #print('Round {:3d}, Average training loss {:.3f}, Average training accuracy {:.3f},  Average val loss {:.3f}, Average val accuracy {:.3f},'.format(iter, tr_loss_avg,tr_accuracy_avg,val_loss_avg,val_accuracy_avg))
        #writer.add_scalar('Average training Loss', tr_loss_avg, global_step=iter)
        #writer.add_scalar('Average training Accuracy', tr_accuracy_avg, global_step=iter)
        #writer.add_scalar('Average validation Loss', tr_loss_avg, global_step=iter)
        #writer.add_scalar('Average validation Accuracy', tr_accuracy_avg, global_step=iter)
        loss_train.append(tr_loss_avg)
        loss_test.append(val_loss_avg)
        acc_train.append(tr_accuracy_avg)
        acc_test.append(val_accuracy_avg)

                 