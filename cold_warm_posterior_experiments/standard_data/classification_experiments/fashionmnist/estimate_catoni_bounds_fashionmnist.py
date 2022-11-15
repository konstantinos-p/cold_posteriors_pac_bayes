from torch import nn
import torch
from scripts.classification_datasets.fashionmnist.fashionmnist_torch_dataset import get_dataloaders
import os
from utils.multiple_standard_data_experiments_utils import estimate_all_bounds_catoni
from utils.model_utils import CNN_nobatchnorm
from utils.laplace_evaluation_utils import NLLLoss_with_log_transform,zero_one_loss,ECE_wrapper

'''
This script estimates the B_mixed,B_original bounds for the fashionmnist dataset.
'''

#Hyperparameters
loss_fn = zero_one_loss
nll_with_log_tranform = NLLLoss_with_log_transform()
batch_size = 40
prior_variance=torch.linspace(0,0,0)
min_temperature=1e-7
max_temperature=1e+4
grid_lambda = 20
n_samples_test_set_la = 100
image_transforms = False

#Get dataset
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/fashionmnist/results'

dir_fashionmnist= '/Users/Kostas/PycharmProjects/cold-warm-posteriors/scripts/classification_datasets/fashionmnist/dataset'


test_dataloader,train_dataloader,validation_dataloader = get_dataloaders(dir=dir_fashionmnist,batch_size=batch_size,
                                                                         image_transforms=image_transforms)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Change directory
os.chdir(path)

#Set model and estimate bounds
model = CNN_nobatchnorm()
model.to(device)
estimate_all_bounds_catoni(prior_variance=prior_variance,train_dataloader = train_dataloader,
                           test_dataloader=test_dataloader,grid_lambda=grid_lambda,min_temperature=min_temperature,
                           max_temperature=max_temperature,n_samples=n_samples_test_set_la,model=model,
                           loss_fn_bound=[loss_fn],loss_functions_test=[nll_with_log_tranform,ECE_wrapper,zero_one_loss],
                           loss_functions_test_names=['nll','ECE','zero_one'])

