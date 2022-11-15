from torch import nn
from scripts.regression_datasets.regression_torch_datasets import kc_houseDataset,get_dataloaders
from utils.multiple_standard_data_experiments_utils import estimate_prior_and_posterior_Roy
from utils.laplace_evaluation_utils import Gaussian_nll_predictive
from utils.model_utils import NeuralNetwork
import os
import torch

'''
This script trains multiple neural networks on the KC_House dataset and saves them in new folders in the current directory.
'''

number_of_networks = 10
loss_fn = nn.MSELoss()
learning_rate = 1e-3
batch_size = 40
epochs_prefix = 10
epochs_suffix = 1

#Directory in which to save the results
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/regression_experiments/kc_house'

#Directory containing the dataset
dataset= kc_houseDataset('/Users/Kostas/PycharmProjects/cold-warm-posteriors/scripts/regression_datasets'
                              '/kc_house_dataset/dataset/data.csv')

#Get the dataloaders
test_dataloader,train_dataloader,train_suffix_dataloader,validation_dataloader,true_dataloader,true_suffix_dataloader = \
    get_dataloaders(dataset=dataset,test_size=4323,suffix_size = 400,
               validation_size=2161,seed_split=42)

#Get and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Train networks
for i in range(number_of_networks):
    print('Starting new model.')
    os.chdir(path)
    folder_name = 'model_' + str(i)
    model = NeuralNetwork(input_size=18)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    estimate_prior_and_posterior_Roy(loss_fn_train=[loss_fn],loss_fn_test=[Gaussian_nll_predictive],
                                     loss_fn_test_names=['GaussianNLL'],epochs_prefix=epochs_prefix,
                                     epochs_suffix=epochs_suffix,train_suffix_dataloader=train_suffix_dataloader,
                                     train_dataloader=train_dataloader,validation_dataloader=validation_dataloader,
                                     test_dataloader=test_dataloader,folder_name=folder_name,model=model,optimizer=optimizer)








