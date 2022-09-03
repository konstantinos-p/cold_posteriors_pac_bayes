import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

'''
In this file we the class AbaloneDataset which is a torch object useful for training and testing.
'''


class AbaloneDataset(Dataset):
    '''Abalone Dataset'''

    def __init__(self,csv_file):
        '''
        Inputs:
            csv_file: Path to the csv file of abalone samples.
        '''
        self.abalone = pd.read_csv(csv_file, index_col=0)
        self.name = 'abalone'

    def __len__(self):
        return len(self.abalone)

    def __getitem__(self,idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (features, target) where target is the age of each abalone.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.abalone.iloc[idx, :-1]
        target = [self.abalone.iloc[idx,-1]]

        #Change features and targets to torch tensors
        features = torch.tensor(features).type(torch.FloatTensor)
        target = torch.tensor(target).type(torch.FloatTensor)

        return features,target

class DiamondsDataset(Dataset):
    '''Abalone Dataset'''

    def __init__(self,csv_file):
        '''
        Inputs:
            csv_file: Path to the csv file of abalone samples.
        '''
        self.diamonds = pd.read_csv(csv_file, index_col=0)
        self.name = 'diamonds'

    def __len__(self):
        return len(self.diamonds)

    def __getitem__(self,idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (features, target) where target is the age of each abalone.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.diamonds.iloc[idx].loc[self.diamonds.columns != 'price']
        target = [self.diamonds.iloc[idx].loc['price']]

        #Change features and targets to torch tensors
        features = torch.tensor(features).type(torch.FloatTensor)
        target = torch.tensor(target).type(torch.FloatTensor)

        return features,target

class kc_houseDataset(Dataset):
    '''KC House Dataset'''

    def __init__(self,csv_file):
        '''
        Inputs:
            csv_file: Path to the csv file of abalone samples.
        '''
        self.house = pd.read_csv(csv_file, index_col=0)
        self.name = 'kc_house'

    def __len__(self):
        return len(self.house)

    def __getitem__(self,idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (features, target) where target is the age of each abalone.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.house.iloc[idx].loc[self.house.columns != 'price']
        target = [self.house.iloc[idx].loc['price']]

        #Change features and targets to torch tensors
        features = torch.tensor(features).type(torch.FloatTensor)
        target = torch.tensor(target).type(torch.FloatTensor)

        return features,target

def data_split(full,test_size=0,train_size=0,suffix_size = 0,validation_size=0,seed_split=42):
    '''
     A function returning the an Abalone dataset split after some transformations.

     Parameters
     ----------
     full: The full dataset.
     test_size: The size of the suffix set.
     train_size: The size of the validation set.
     suffix_size: The size of the suffix set.
     validation_size: The size of the validation set.
     seed_split: The seed used in the random split. Fix this so that the results are reproducible.
     dataset_class: The dataset class to use when parsing the raw data.

     Returns
     -------
     test: The test set.
     train: The training set.
     train_suffix: The training suffix set.
     validation: The validation set.
     true: The true set.
     true_suffix: The true suffix set.
    '''
    if len(full) < test_size+train_size:
        raise ValueError('Sum of Test set {:n} and Train set {:n} sizes is larger than the full dataset size {:n}'
                         .format(test_size,train_size,len(full)) )

    test, train_tmp,true_tmp = random_split(full, [test_size,train_size,len(full)-test_size-train_size],
                                     generator=torch.Generator().manual_seed(seed_split))

    if len(train_tmp) < suffix_size+validation_size:
        raise ValueError('Sum of Suffix set {:n} and Validation set {:n} sizes is larger than the Training set size {:n}'
                         .format(suffix_size,validation_size,len(train_tmp)) )
    if len(true_tmp) < suffix_size:
        raise ValueError('Suffix set {:n} size is larger than the True set size {:n}'
                         .format(suffix_size,len(true_tmp)) )

    train_suffix, validation,train = random_split(train_tmp, [suffix_size,validation_size,len(train_tmp)-suffix_size-validation_size],
                                     generator=torch.Generator().manual_seed(seed_split))

    true_suffix, true = random_split(true_tmp, [suffix_size,len(true_tmp)-suffix_size],
                                     generator=torch.Generator().manual_seed(seed_split))

    return test,train,train_suffix,validation,true,true_suffix

def get_dataloaders(dataset=None,test_size=0,suffix_size = 0,
               validation_size=0,seed_split=42,batch_size=40):
    """
    Return dataloaders for all splits of the dataset 'dataset'.
    """

    test, train, train_suffix, validation, true, true_suffix = \
        data_split(full=dataset, test_size=test_size, train_size=test_size + validation_size,
                   suffix_size=suffix_size,
                   validation_size=validation_size, seed_split=seed_split)

    test_dataloader = DataLoader(test, batch_size=batch_size,
                                         shuffle=True)

    train_dataloader = DataLoader(train, batch_size=batch_size,
                                  shuffle=True)

    train_suffix_dataloader = DataLoader(train_suffix, batch_size=batch_size,
                                 shuffle=True)

    validation_dataloader = DataLoader(validation, batch_size=batch_size,
                                       shuffle=True)

    true_dataloader = DataLoader(true, batch_size=batch_size,
                                 shuffle=True)

    true_suffix_dataloader = DataLoader(true_suffix, batch_size=batch_size,
                                       shuffle=True)

    return test_dataloader,train_dataloader,train_suffix_dataloader,validation_dataloader,true_dataloader,true_suffix_dataloader



