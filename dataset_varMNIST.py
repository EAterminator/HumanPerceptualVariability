import os
import sys

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class VarMNIST(Dataset):

    def __init__(self, df_DigitRecog, path_dataset, subject, is_train):            
        
        self.df_DigitRecog = df_DigitRecog
        list_subjects = self.df_DigitRecog['subject_id'].unique()
        self.path_dataset = path_dataset
        self.subject = subject  
        if subject != 'group':
            # extract selected subject 
            self.df_DigitRecog = self.df_DigitRecog[self.df_DigitRecog['subject_id'] == list_subjects[subject]]

        self.is_train = is_train
        self.df_DigitRecog = self.df_DigitRecog[self.df_DigitRecog['is_train'] == is_train]

    def __len__(self):
        return len(self.df_DigitRecog)
    
    def __getitem__(self, idx):
        # extract data for the idx-th row
        row = self.df_DigitRecog.iloc[idx]
        # load image
        path_img = os.path.join(self.path_dataset, row['stimulus'])
        img = Image.open(path_img).convert('L')
        img = transforms.ToTensor()(img)
        # load label
        label = row['response']
        return img, label
    

class varMNIST_tensor(Dataset):

    def __init__(self, df_DigitRecog, images, subject, is_train):            
        
        self.df_DigitRecog = df_DigitRecog
        list_subjects = self.df_DigitRecog['subject_id'].unique()
        self.images = images
        self.subject = subject  
        if subject != 'group':
            # extract selected subject 
            self.df_DigitRecog = self.df_DigitRecog[self.df_DigitRecog['subject_id'] == list_subjects[subject]]

        self.is_train = is_train
        self.df_DigitRecog = self.df_DigitRecog[self.df_DigitRecog['is_train'] == is_train]

    def __len__(self):
        return len(self.df_DigitRecog)
    
    def __getitem__(self, idx):
        # extract data for the idx-th row
        row = self.df_DigitRecog.iloc[idx]
        # load image
        img_idx = row['stimulus_id']
        img = self.images[img_idx]
        # load label
        label = row['response']
        return img, label

from torch.utils.data import WeightedRandomSampler, DataLoader, ConcatDataset

def create_balanced_loader(datasets, weights_dataset=None, batch_size=128, num_samples=None):
    """
    Create a balanced DataLoader from multiple datasets with a fixed number of samples.
    
    Parameters:
        datasets (list): List of datasets to combine.
        weights_dataset (list, optional): Relative weights for each dataset. Default is None.
        batch_size (int): Batch size for the DataLoader.
        num_samples (int): Total number of samples in the DataLoader.
    
    Returns:
        DataLoader: A DataLoader with balanced sampling and limited total samples.
    """
    # Combine all datasets into one
    combined_dataset = ConcatDataset(datasets)
    n_samples = [len(dataset) for dataset in datasets]
    n_total = sum(n_samples)
    
    if weights_dataset is None:
        weights_dataset = [1.0] * len(datasets)
    
    # Calculate per-sample weights
    weights_sample = []
    for i, _ in enumerate(datasets):
        weight = weights_dataset[i] * n_total / n_samples[i]
        weights_sample.extend([weight] * n_samples[i])
    
    # Create a weighted sampler with the specified number of samples
    if num_samples is None:
        num_samples = n_total
    sampler = WeightedRandomSampler(weights_sample, num_samples)
    loader = DataLoader(combined_dataset, batch_size=batch_size, sampler=sampler)
    
    return loader


if __name__ == '__main__':

    # load data
    path_DigitRecog = 'varMNIST/df_DigitRecog.csv'
    df_DigitRecog = pd.read_csv(path_DigitRecog)
    subject = 'group' # int or 'group'
    is_train = True
    # dataset = VarMNIST(df_DigitRecog, path_dataset, subject, is_train)
    images = torch.load('varMNIST/images.pt')
    dataset = varMNIST_tensor(df_DigitRecog, images, subject, is_train)

    # visualize data
    idx = 8
    img, label = dataset[idx]
    print(img.shape)
    print(label)
    import matplotlib.pyplot as plt
    plt.imshow(img.permute(1,2,0).detach(), cmap='gray')
    plt.show()

    # load mnist dataset
    from torchvision.datasets import MNIST
    mnist_dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

    # create balanced loader
    loader = create_balanced_loader([dataset, mnist_dataset], batch_size=128)
    # loader = DataLoader(dataset, batch_size=128, shuffle=True)

    import time
    t = time.time()
    # visualize loader
    for img, label in loader:
        # print(img.shape)
        # print(label)
        print(time.time()-t)
        # break
