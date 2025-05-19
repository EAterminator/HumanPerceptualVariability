import os
from sys import path
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from dataset_varMNIST import varMNIST_tensor, create_balanced_loader
from utils import (
    build_vit,
    build_vgg,
    build_cornet,
    build_mlp,
    build_lrm,
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from tqdm import tqdm
import copy
# return the trained model with the best valid_acc
def train_model(
        model,
        dataloader_train,
        dataloader_valid,
        criterion,
        optimizer,
        device,
        n_epochs=16,
        log=True
):
    model.train()
    best_valid_acc = 0
    best_model = None
    for epoch in range(n_epochs):
        running_loss = 0.0
        for features, labels in dataloader_train:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        list_valid_loss, list_valid_acc = evaluate_model(model, dataloader_valid, criterion, device)
        if log:
            for i in range(len(dataloader_valid)):
                print(f'epoch: {epoch}, dataset: {i}, valid_loss: {list_valid_loss[i]: .4f}, valid_acc: {list_valid_acc[i]: .4f}')
        # focus on the first validation dataset
        if list_valid_acc[0] > best_valid_acc:
            best_valid_acc = list_valid_acc[0]
            best_model = copy.deepcopy(model.state_dict())
            # print(f'best model updated, valid_acc: {best_valid_acc: .4f}')
    model.load_state_dict(best_model)
    return model

# evaluate the logistic regression model for each subject
def evaluate_model(model, dataloader, criterion, device):
    # dataloader is a list of dataloaders
    # compute the loss and accuracy for each dataloader
    model.eval()
    log_loss = []
    log_correct = []
    with torch.no_grad():
        for dataloader_i in dataloader:
            running_loss = 0.0
            correct = 0
            total = 0
            for features, labels in dataloader_i:
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            log_loss.append(running_loss / len(dataloader_i))
            log_correct.append(correct / total)
    return log_loss, log_correct


def finetune_group_level(model, dataloader_train, dataloader_valid, device):

    # build the logistic regression model
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    list_valid_loss, list_valid_acc_before = evaluate_model(model, dataloader_valid, criterion, device)
    for i in range(len(dataloader_valid)):
        print(f'before finetuning, dataset: {i}, valid_loss: {list_valid_loss[i]: .4f}, valid_acc: {list_valid_acc_before[i]: .4f}')

    model = train_model(
        model,
        dataloader_train,
        dataloader_valid,
        criterion,
        optimizer,
        device,
        n_epochs=16
    )

    list_valid_loss, list_valid_acc_after = evaluate_model(model, dataloader_valid, criterion, device)
    for i in range(len(dataloader_valid)):
        print(f'after finetuning, dataset: {i}, valid_loss: {list_valid_loss[i]: .4f}, valid_acc: {list_valid_acc_after[i]: .4f}')

    return model, list_valid_acc_before, list_valid_acc_after



def main(path_save):

    # load data
    path_DigitRecog = 'varMNIST/df_DigitRecog.csv'
    df_DigitRecog = pd.read_csv(path_DigitRecog)

    # build 5 models
    path_vit = 'ckpts/vit_mnist.pth'
    path_vgg = 'ckpts/small_vgg_mnist.pth'
    path_cornet = 'ckpts/cornet_z-mnist.pt'
    path_mlp = 'ckpts/mlp_mnist.pt'
    path_lrm = 'ckpts/lrm_mnist.pth'
    vit = build_vit(path_vit)
    vgg = build_vgg(path_vgg)
    cornet = build_cornet(path_cornet)
    mlp = build_mlp(path_mlp)
    lrm = build_lrm(path_lrm)
    models = [vit, vgg, cornet, mlp, lrm]
    model_names = ['vit', 'vgg', 'cornet', 'mlp', 'lrm']

    # varMNIST dataset
    path_image = 'varMNIST/images.pt'
    images = torch.load(path_image)
    varmnist_train = varMNIST_tensor(df_DigitRecog, images, subject='group', is_train=True)
    varmnist_valid = varMNIST_tensor(df_DigitRecog, images, subject='group', is_train=False)

    # MNIST dataset
    from torchvision.datasets import MNIST
    from torchvision import transforms
    mnist_train = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    mnist_valid = MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

    # build the dataloader
    dataloader_train = create_balanced_loader([varmnist_train, mnist_train], batch_size=128, num_samples=100000)
    dataloader_valid_mnist = torch.utils.data.DataLoader(mnist_valid, batch_size=128, shuffle=False)
    dataloader_valid_varmnist = torch.utils.data.DataLoader(varmnist_valid, batch_size=128, shuffle=False)
    dataloader_valid = [dataloader_valid_varmnist, dataloader_valid_mnist]

    # record the performance
    log_valid_acc_before = []
    log_valid_acc_after = []
    # finetune the model for each subject
    for model, model_name in zip(models, model_names):
        print(f'finetune the model for {model_name}')
        model_finetuned, valid_acc_before, valid_acc_after = finetune_group_level(model, dataloader_train, dataloader_valid, device)
        path_save_model = os.path.join(path_save, f'model_{model_name}.pth')
        log_valid_acc_before.append(valid_acc_before)
        log_valid_acc_after.append(valid_acc_after)
        torch.save(model_finetuned.state_dict(), path_save_model)

     # save the performance
    df_performance = pd.DataFrame({
        'model': model_names,
        'valid_acc_before': log_valid_acc_before,
        'valid_acc_after': log_valid_acc_after
    })
    path_performance = os.path.join(path_save, 'performance.csv')
    df_performance.to_csv(path_performance, index=False)


if __name__ == '__main__':
    path_save = '' # specify the path to save the results
    main(path_save)