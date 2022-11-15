from torchvision.datasets import MNIST,EMNIST
from torch.utils.data import random_split,DataLoader
import torchvision
import torch
from torch.nn.functional import one_hot

"""
This script downloads the MNIST and EMNIST datasets and stores them in
the relevant folders, and run some tests. I check that:
1) The number of samples per dataset is correct.
2) That I can make the transform to one_hot encoding for the labels. This is needed when approximating the
classification problem as a regression problem.
3) I check in general that the format of the data samples is correct.
"""

def transformation_one_hot(x):
    '''
    This function transforms labels into the one_hot encoding.
    '''
    x = torch.as_tensor(x)
    return one_hot(x,num_classes=10)

#Download MNIST and apply a transoform to one hot encoding of the labels.
mnist = MNIST(root='dataset/mnist',download=True,train=True,transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]),target_transform=transformation_one_hot)
#Download EMNIST
emnist = EMNIST(root='dataset/emnist',download=True,split='digits',train=False,transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

#Test the random_split function that
train,validation = random_split(mnist, [int(len(mnist)*0.9),int(len(mnist)*0.1)], generator=torch.Generator().manual_seed(42))

#Print the length of this script
print(len(train))
print(len(validation))

#What is the format of the signals x and labels y in these datasets?
#MNIST
mnist_dataloader = DataLoader(mnist, batch_size=1,
                            shuffle=True)
x_mnist,y_mnist = next(iter(mnist_dataloader))
#EMNIST
emnist_dataloader = DataLoader(emnist, batch_size=1,
                            shuffle=True)
x_emnist,y_emnist = next(iter(emnist_dataloader))


end =1