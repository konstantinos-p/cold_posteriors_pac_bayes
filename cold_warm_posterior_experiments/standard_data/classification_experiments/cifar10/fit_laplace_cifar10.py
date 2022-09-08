from torch import nn
from scripts.classification_datasets.cifar10.cifar10_torch_dataset import get_dataloaders
from utils.multiple_standard_data_experiments_utils import estimate_all_metrics_plain
import os
from utils.laplace_evaluation_utils import zero_one_loss,ECE_wrapper,NLLLoss_with_log_transform
from utils.wide_resnet_utils import FixupWideResNet
import torch

'''
This script estimates the required metrics for the laplace approximation of the cifar10 dataset.
'''

#Hyperparameters
loss_fn = nn.CrossEntropyLoss()
nll_with_log_tranform = NLLLoss_with_log_transform()
batch_size = 128
grid_lambda = 20
min_temperature=0.1
max_temperature=20
grid_prior_variance = 20
min_prior_variance=0.1
max_prior_variance=20
n_samples_metrics_la = 100
hessian_structure = 'kron'
image_transforms = False

#Get dataset
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/cifar10'
dir_cifar10= '/Users/Kostas/PycharmProjects/cold-warm-posteriors/scripts/classification_datasets/cifar10/dataset'

test_dataloader,train_dataloader,validation_dataloader = get_dataloaders(dir=dir_cifar10,batch_size=batch_size,
                                                                         image_transforms=image_transforms)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Change directory
os.chdir(path)

#Set model and estimate bounds
model = FixupWideResNet(22, 4, 10, dropRate=0.3)
model.to(device)
estimate_all_metrics_plain(train_dataloader,test_dataloader,validation_dataloader,model,likelihood='classification',
                           loss_functions_test=[nll_with_log_tranform,ECE_wrapper,zero_one_loss],
                           loss_functions_test_names=['nll', 'ECE', 'zero_one'],grid_lambda=grid_lambda,
                           min_temperature=min_temperature,max_temperature=max_temperature,grid_prior_variance=None,
                           min_prior_variance=min_prior_variance,max_prior_variance=max_prior_variance,
                           n_samples=n_samples_metrics_la,hessian_structure=hessian_structure)

