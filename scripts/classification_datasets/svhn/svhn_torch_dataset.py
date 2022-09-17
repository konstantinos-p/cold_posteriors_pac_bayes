from torchvision.datasets import SVHN
from torch.utils.data import random_split
import torchvision
import torch
from torch.nn.functional import one_hot
import os
from torch.utils.data import DataLoader
from torchvision import transforms

'''
In this module I include all functions necessary to use the SVHN dataset.
'''

def SVHN_split(dir='dataset',mode='classification',validation_percentage=10,seed_split=42,image_transforms=False):
    '''
     A function returning a SVHN dataset split after some transformations.

     Parameters
     ----------
     dir: The directory where the SVHN raw data are located.
     mode: {'classification','regression'} Whether to return the labels are one hot encodings or int.
     validation_percentage: The size of the validation set in percentage points.
     seed_split: The seed used in the random split. Fix this so that the results are reproducible.

     Returns
     -------
     test: The test set.
     train: The training set.
     validation: The validation set.
     '''

    if 'svhn' not in os.listdir(dir):
        raise FileNotFoundError('The svhn folder doesn\'t exist in directory: '+dir )

    mean = [0.4380, 0.4440, 0.4730]
    std = [0.1751, 0.1771, 0.1744]

    default_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
        ])

    if image_transforms==False:
        image_transform = default_transform
    else:
        image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        transforms.Lambda(lambda x: normalize_svhn(x)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Pad(4),
        torchvision.transforms.RandomCrop(32)])

    if mode == 'classification':
        target_transform = None
    elif mode == 'regression':
        target_transform = transformation_one_hot

    #Create the test set
    test = SVHN(root=dir+'/svhn', download=True, split='test', transform=default_transform, target_transform=target_transform)

    #Create a temporary train set
    train_tmp = SVHN(root=dir+'/svhn', download=True, split='train', transform=image_transform, target_transform=target_transform)


    #Split the temporary training set into a training and validation set
    train, validation = random_split(train_tmp, [int(len(train_tmp)*(100-validation_percentage)/100),
                                                 len(train_tmp)-int(len(train_tmp)*(100-validation_percentage)/100)],
                                     generator=torch.Generator().manual_seed(seed_split))

    return test,train,validation

def transformation_one_hot(x):
    '''
    This function transforms labels into the one_hot encoding.
    '''
    x = torch.as_tensor(x)
    return one_hot(x,num_classes=10)

def get_dataloaders(validation_percentage=10,seed_split=42,batch_size=100,dir=None,mode='classification',image_transforms=False):
    """
    Return dataloaders for all splits of SVHN.
    """

    test, train, validation = \
        SVHN_split(dir=dir, mode=mode, validation_percentage=validation_percentage, seed_split=seed_split,image_transforms=image_transforms)

    test_dataloader = DataLoader(test, batch_size=batch_size,
                                         shuffle=True)

    train_dataloader = DataLoader(train, batch_size=batch_size,
                                  shuffle=True)

    validation_dataloader = DataLoader(validation, batch_size=batch_size,
                                       shuffle=True)
    return test_dataloader,train_dataloader,validation_dataloader

def normalize_svhn(data_tensor):
    '''re-scale image values to [-1, 1]'''
    return (data_tensor / 255.)