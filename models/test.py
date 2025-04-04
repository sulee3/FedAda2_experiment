#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def test(net, testloader,args):
    loss = 0
    acc = 0
    if net.__class__.__name__  == 'AE':
        loss,acc = test_ae(net,testloader,args)
    elif net.__class__.__name__  == 'LinearM' or net.__class__.__name__  == 'VisionTransformer': 
        loss,acc = test_linear(net,testloader,args)
    elif net.__class__.__name__  == 'MobileViTForImageClassification':
        loss,acc = test_vit(net,testloader,args)
    return loss, acc
    

def test_ae(net, testloader, args):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.MSELoss()
    correct, total, loss = 0, 0, 0.0
    net.eval().to(args.device)
    with torch.no_grad():
        for images, _ in testloader:
            images = images.to(args.device)
            outputs = net(images)
            images = images.reshape(-1, 784)
            data_count = len(testloader.dataset)
            loss += criterion(outputs, images).item() * data_count
            total += data_count
            # trainloaders[client_idx].dataset
    loss /= total #len(testloader)
    return loss, 0

def test_linear(net, testloader,args):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval().to(args.device)
    with torch.no_grad():
        for inputs,labels in testloader:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs = net(inputs)
            loss += criterion(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= total #len(testloader)
    accuracy = correct / total
    return loss, accuracy

def test_vit(net, testloader,args):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval().to(args.device)
    with torch.no_grad():
        for inputs,labels in testloader:
            inputs  = inputs.to(args.device)
            labels  = labels.to(args.device)
            outputs = net(inputs).logits
            loss += criterion(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= total #len(testloader)
    accuracy = correct / total
    return loss, accuracy

def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != 0:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

