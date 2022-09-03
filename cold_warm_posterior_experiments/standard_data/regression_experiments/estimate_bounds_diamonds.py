from torch import nn
import torch
from scripts.regression_datasets.regression_torch_datasets import DiamondsDataset,get_dataloaders
import os
from utils.multiple_standard_data_experiments_utils import estimate_all_bounds
from utils.model_utils import NeuralNetwork
from utils.laplace_evaluation_utils import Gaussian_nll_predictive

'''
This script estimates the B_approx,B_mixed,B_original bounds for the Diamonds dataset.
'''

#Hyperparameters
loss_fn = nn.MSELoss()
batch_size = 40
prior_variance=torch.linspace(0.00001,0.1,2)
min_temperature=0.1
max_temperature=10
grid_lambda = 20
n_samples_test_set_la = 100
input_dimensions  = 9

#Get dataset
dataset= DiamondsDataset('/Users/Kostas/PycharmProjects/cold-warm-posteriors/scripts/regression_datasets'
                              '/diamonds_dataset/dataset/data.csv')

test_dataloader,train_dataloader,train_suffix_dataloader,validation_dataloader,true_dataloader,true_suffix_dataloader = \
    get_dataloaders(dataset=dataset,test_size=10788,suffix_size = 1000,
               validation_size=5394,seed_split=42)

#Change directory
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/regression_experiments/diamonds'
os.chdir(path)

#Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Set model and estimate bound
model = NeuralNetwork(input_size=input_dimensions)
model.to(device)
estimate_all_bounds(prior_variance=prior_variance,true_dataloader=true_dataloader,
                    train_suffix_dataloader=train_suffix_dataloader,test_dataloader=test_dataloader,
                        grid_lambda=grid_lambda,min_temperature=min_temperature,max_temperature=max_temperature,
                    n_samples=n_samples_test_set_la,model=model,likelihood='regression',loss_fn_bound=[loss_fn],
                    loss_functions_test=[Gaussian_nll_predictive],loss_functions_test_names=['GaussianNLL'],n_f=10,n_XY=10)

