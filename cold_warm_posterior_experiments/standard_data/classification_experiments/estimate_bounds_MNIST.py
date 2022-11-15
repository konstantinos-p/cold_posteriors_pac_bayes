from torch import nn
import torch
from scripts.classification_datasets.mnist.mnist_torch_dataset import get_dataloaders
import os
from utils.multiple_standard_data_experiments_utils import estimate_all_bounds
from utils.model_utils import LeNet
from utils.laplace_evaluation_utils import NLLLoss_with_log_transform,zero_one_loss,ECE_wrapper

'''
This script estimates the B_mixed,B_original bounds for the MNIST-10 dataset.
'''

#Hyperparameters
loss_fn = nn.CrossEntropyLoss()
nll_with_log_tranform = NLLLoss_with_log_transform()
batch_size = 40
prior_variance=torch.linspace(0.0001,0.1,1)
min_temperature=0.1
max_temperature=20
grid_lambda = 20
n_samples_test_set_la = 100

#Get dataset
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/MNIST'
dir_MNIST= '/Users/Kostas/PycharmProjects/cold-warm-posteriors/scripts/classification_datasets/mnist/dataset'

test_dataloader,train_dataloader,train_suffix_dataloader,validation_dataloader,true_dataloader,true_suffix_dataloader = \
    get_dataloaders(dir=dir_MNIST,mode='classification',suffix_size = 5000,validation_size=5000,seed_split=42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Change directory
os.chdir(path)

#Set model and estimate bounds
model = LeNet()
model.to(device)
estimate_all_bounds(prior_variance=prior_variance,true_dataloader=true_dataloader,
                    train_suffix_dataloader=train_suffix_dataloader,test_dataloader=test_dataloader,
                        grid_lambda=grid_lambda,min_temperature=min_temperature,max_temperature=max_temperature,
                    n_samples=n_samples_test_set_la,model=model,likelihood='classification',loss_fn_bound=[loss_fn],
                    loss_functions_test=[nll_with_log_tranform,ECE_wrapper,zero_one_loss],
                    loss_functions_test_names=['nll','ECE','zero_one'])

