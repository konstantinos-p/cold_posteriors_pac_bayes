from torchvision.datasets import MNIST,EMNIST
from torch.utils.data import random_split
import torchvision
import torch
from torch.nn.functional import one_hot
import os
from torch.utils.data import DataLoader

'''
In this module I include all functions necessary to use the MNIST dataset together with the EMNIST extension.
'''

def MNIST_split(dir='dataset',mode='classification',suffix_size = 5000,validation_size=5000,seed_split=42,image_transforms=None):
    '''
     A function returning a MNIST dataset split after some transformations.

     Parameters
     ----------
     dir: The directory where the MNIST and EMNIST raw data are located.
     mode: {'classification','regression'} Whether to return the labels are one hot encodings or int.
     suffix_size: The size of the suffix set.
     validation_size: The size of the validation set.
     seed_split: The seed used in the random split. Fix this so that the results are reproducible.

     Returns
     -------
     test: The test set.
     train: The training set.
     train_suffix: The training suffix set.
     validation: The validation set.
     true: The true set.
     true_suffix: The true suffix set.
     '''

    if 'mnist' not in os.listdir(dir):
        raise FileNotFoundError('The mnist folder doesn\'t exist in directory: '+dir )
    if 'emnist' not in os.listdir(dir):
        raise FileNotFoundError('The emnist folder doesn\'t exist in directory: '+dir )

    if image_transforms==None:
        image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
        ])
    default_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
        ])

    if mode == 'classification':
        target_transform = None
    elif mode == 'regression':
        target_transform = transformation_one_hot

    #Create the test set
    test = MNIST(root=dir+'/mnist', download=True, train=False, transform=default_transform, target_transform=target_transform)

    #Create a temporary train set
    train_tmp = MNIST(root=dir+'/mnist', download=True, train=True, transform=image_transform, target_transform=target_transform)
    if len(train_tmp) < suffix_size+validation_size:
        raise ValueError('Sum of Training Suffix set {:n} and Validation set {:n} sizes is larger than the Training set size {:n}'.format(suffix_size,validation_size,len(train_tmp)) )


    #Create a temporary true set
    true_tmp = EMNIST(root=dir+'/emnist', download=True, split='digits', train=True,
                    transform=image_transform, target_transform=target_transform)
    if len(true_tmp) < suffix_size:
        raise ValueError('True Suffix set {:n} size is larger than the True set size {:n}'.format(suffix_size,len(true_tmp)) )


    #Split the temporary training set into a training,suffix and validation set
    train_suffix, validation,train = random_split(train_tmp, [suffix_size,validation_size,len(train_tmp)-validation_size-suffix_size],
                                     generator=torch.Generator().manual_seed(seed_split))

    #Split the temporary true set into a true and suffix set
    true_suffix, true, _ = random_split(true_tmp, [suffix_size,2*suffix_size,len(true_tmp)-3*suffix_size],
                                     generator=torch.Generator().manual_seed(seed_split))

    return test,train,train_suffix,validation,true,true_suffix

def transformation_one_hot(x):
    '''
    This function transforms labels into the one_hot encoding.
    '''
    x = torch.as_tensor(x)
    return one_hot(x,num_classes=10)

def get_dataloaders(suffix_size = 5000,
               validation_size=5000,seed_split=42,batch_size=40,dir=None,mode='classification'):
    """
    Return dataloaders for all splits of MNIST-10.
    """

    test, train, train_suffix, validation, true, true_suffix = \
        MNIST_split(dir=dir, mode=mode, suffix_size=suffix_size, validation_size=validation_size, seed_split=seed_split)

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