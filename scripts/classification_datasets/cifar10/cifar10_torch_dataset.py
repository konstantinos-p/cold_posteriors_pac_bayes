from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
import torchvision
import torch
from torch.nn.functional import one_hot
import os
from torch.utils.data import DataLoader

'''
In this module I include all functions necessary to use the CIFAR-10 dataset.
'''

def CIFAR10_split(dir='dataset',mode='classification',validation_percentage=10,seed_split=42,image_transforms=False):
    '''
     A function returning a CIFAR10 dataset split after some transformations.

     Parameters
     ----------
     dir: The directory where the CIFAR10 raw data are located.
     mode: {'classification','regression'} Whether to return the labels are one hot encodings or int.
     validation_percentage: The size of the validation set in percentage points.
     seed_split: The seed used in the random split. Fix this so that the results are reproducible.

     Returns
     -------
     test: The test set.
     train: The training set.
     validation: The validation set.
     '''

    if 'cifar10' not in os.listdir(dir):
        raise FileNotFoundError('The cifar10 folder doesn\'t exist in directory: '+dir )

    default_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    if image_transforms==False:
        image_transform = default_transform
    else:
        image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Pad(4),
        torchvision.transforms.RandomCrop(32)])

    if mode == 'classification':
        target_transform = None
    elif mode == 'regression':
        target_transform = transformation_one_hot

    #Create the test set
    test = CIFAR10(root=dir+'/cifar10', download=True, train=False, transform=default_transform, target_transform=target_transform)

    #Create a temporary train set
    train_tmp = CIFAR10(root=dir+'/cifar10', download=True, train=True, transform=image_transform, target_transform=target_transform)


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
    Return dataloaders for all splits of CIFAR10.
    """

    test, train, validation = \
        CIFAR10_split(dir=dir, mode=mode, validation_percentage=validation_percentage, seed_split=seed_split,image_transforms=image_transforms)

    test_dataloader = DataLoader(test, batch_size=batch_size,
                                         shuffle=True)

    train_dataloader = DataLoader(train, batch_size=batch_size,
                                  shuffle=True)

    validation_dataloader = DataLoader(validation, batch_size=batch_size,
                                       shuffle=True)
    return test_dataloader,train_dataloader,validation_dataloader