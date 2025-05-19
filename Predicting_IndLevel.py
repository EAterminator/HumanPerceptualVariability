import os
from random import sample
from sys import path
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from dataset_varMNIST import varMNIST_tensor, create_balanced_loader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from Predicting_GroupLevel import (
    evaluate_model,
    train_model
)
from utils import (
    build_vit,
    build_vgg,
    build_cornet,
    build_mlp,
    build_lrm,
)


def finetune_subject_level(model, dataloader_train, dataloader_valid, device):

    # build the logistic regression model
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

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
        n_epochs=16,
        log=False
    )

    list_valid_loss, list_valid_acc_after = evaluate_model(model, dataloader_valid, criterion, device)
    for i in range(len(dataloader_valid)):
        print(f'after finetuning, dataset: {i}, valid_loss: {list_valid_loss[i]: .4f}, valid_acc: {list_valid_acc_after[i]: .4f}')

    return model, list_valid_acc_before, list_valid_acc_after


def main(path_save):

    # load data
    path_DigitRecog = 'varMNIST/df_DigitRecog.csv'
    df_DigitRecog = pd.read_csv(path_DigitRecog)
    list_subjects = df_DigitRecog['subject_id'].unique()

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

    # record the performance
    log_valid_acc_before = []
    log_valid_acc_after = []
    log_subject = []
    # loop over subjects. df_subject['subject_id'] is the subject id
    for i, subject_id in enumerate(list_subjects):

        # build the dataloader
        varmnist_train_i = varMNIST_tensor(df_DigitRecog, images, subject=i, is_train=True)
        varmnist_valid_i = varMNIST_tensor(df_DigitRecog, images, subject=i, is_train=False)

        if len(varmnist_train_i) < 100:
            print(f'skip subject {i} (id:{subject_id}) because of insufficient data')
            continue
        print(f'finetuning subject {i} (id:{subject_id}) ({i+1}/{len(list_subjects)})')
        print(f'train_dataset: {len(varmnist_train_i)}, valid_dataset: {len(varmnist_valid_i)}')

        dataloader_train = create_balanced_loader([varmnist_train_i, varmnist_train, mnist_train], [2,1,1], batch_size=128, num_samples=500)
    
        dataloader_valid_i = torch.utils.data.DataLoader(varmnist_valid_i, batch_size=128, shuffle=True)
        sampler_varmnist = torch.utils.data.RandomSampler(varmnist_valid, num_samples=128)
        dataloader_valid_varmnist = torch.utils.data.DataLoader(varmnist_valid, batch_size=128, sampler=sampler_varmnist)
        sampler_mnist = torch.utils.data.RandomSampler(mnist_valid, num_samples=128)
        dataloader_valid_mnist = torch.utils.data.DataLoader(mnist_valid, batch_size=128, sampler=sampler_mnist)
        dataloader_valid = [dataloader_valid_i, dataloader_valid_varmnist, dataloader_valid_mnist]

        # load the model
        path_save_group = '' # the path where you save the group models
        model_names = ['vit', 'vgg', 'cornet', 'mlp', 'lrm']
        models = []
        for model_name in model_names:
            path_model = os.path.join(path_save_group, f'model_{model_name}.pth')
            if model_name == 'vit':
                model = build_vit(path_model)
            elif model_name == 'vgg':
                model = build_vgg(path_model)
            elif model_name == 'cornet':
                model = build_cornet(path_model)
            elif model_name == 'mlp':
                model = build_mlp(path_model)
            elif model_name == 'lrm':
                model = build_lrm(path_model)
            models.append(model)
        
        log_valid_acc_before_subject = []
        log_valid_acc_after_subject = []
        # fineturn the model for each subject
        for model, model_name in zip(models, model_names):
            print(f'finetuning model {model_name} for subject {subject_id}')
            model_finetuned, valid_acc_before, valid_acc_after = finetune_subject_level(
                model, dataloader_train, dataloader_valid, device
            )
            path_save_model = os.path.join(path_save, f'model_{model_name}_subject_{subject_id}.pth')
            log_valid_acc_before_subject.append(valid_acc_before)
            log_valid_acc_after_subject.append(valid_acc_after)
            torch.save(model_finetuned.state_dict(), path_save_model)
        log_valid_acc_before.append(log_valid_acc_before_subject)
        log_valid_acc_after.append(log_valid_acc_after_subject)
        log_subject.append(subject_id)
    
    # save the performance
    df_performance = pd.DataFrame({
        'subject_id': log_subject,
        'valid_acc_before_vit': [x[0] for x in log_valid_acc_before],
        'valid_acc_before_vgg': [x[1] for x in log_valid_acc_before],
        'valid_acc_before_cornet': [x[2] for x in log_valid_acc_before],
        'valid_acc_before_mlp': [x[3] for x in log_valid_acc_before],
        'valid_acc_before_lrm': [x[4] for x in log_valid_acc_before],
        'valid_acc_after_vit': [x[0] for x in log_valid_acc_after],
        'valid_acc_after_vgg': [x[1] for x in log_valid_acc_after],
        'valid_acc_after_cornet': [x[2] for x in log_valid_acc_after],
        'valid_acc_after_mlp': [x[3] for x in log_valid_acc_after],
        'valid_acc_after_lrm': [x[4] for x in log_valid_acc_after],
    })
    path_performance = os.path.join(path_save, 'df_performance.csv')
    df_performance.to_csv(path_performance, index=False)


if __name__ == '__main__':
    path_save = '' # the path where you save the subject models
    main(path_save)