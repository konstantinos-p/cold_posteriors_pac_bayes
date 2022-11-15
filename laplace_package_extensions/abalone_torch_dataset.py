import torch
import pandas as pd
from torch.utils.data import Dataset
'''
In this file we define the class AbaloneDataset which is a torch object useful for training and testing.
'''


class AbaloneDataset(Dataset):
    '''Abalone Dataset'''

    def __init__(self,csv_file):
        '''
        Inputs:
            csv_file: Path to the csv file of abalone samples.
        '''
        self.abalone = pd.read_csv(csv_file, index_col=0)

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
