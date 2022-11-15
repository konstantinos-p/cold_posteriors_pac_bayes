from torch import nn
from scripts.classification_datasets.mnist.mnist_torch_dataset import get_dataloaders
from utils.multiple_standard_data_experiments_utils import estimate_prior_and_posterior_Roy
import os
from utils.laplace_evaluation_utils import zero_one_loss,ECE_wrapper
from utils.model_utils import LeNet
import torch
'''
This script trains multiple LeNet networks on the MNIST dataset and saves them in new folders in the current directory.
'''

number_of_networks = 10
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
batch_size = 40
epochs_prefix = 10
epochs_suffix = 1

#Directory in which to save the models
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/MNIST'

#Path to the MNIST dataset
dir_MNIST= '/Users/Kostas/PycharmProjects/cold-warm-posteriors/scripts/classification_datasets/mnist/dataset'

#Define dataloaders
test_dataloader,train_dataloader,train_suffix_dataloader,validation_dataloader,true_dataloader,true_suffix_dataloader = \
    get_dataloaders(dir=dir_MNIST,mode='classification',suffix_size = 5000,validation_size=5000,seed_split=42)

#Get and set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Train Models
for i in range(number_of_networks):
    print('Starting new model.')
    os.chdir(path)
    folder_name = 'model_' + str(i)
    model = LeNet()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    estimate_prior_and_posterior_Roy(loss_fn_train=[loss_fn],loss_fn_test=[loss_fn,ECE_wrapper,zero_one_loss],
                                     loss_fn_test_names=['nll','ECE','zero_one'],
                                     epochs_prefix=epochs_prefix,epochs_suffix=epochs_suffix,
                                     train_suffix_dataloader=train_suffix_dataloader,
                                     train_dataloader=train_dataloader,validation_dataloader=validation_dataloader,
                                     test_dataloader=test_dataloader,folder_name=folder_name,model=model,
                                     optimizer=optimizer)








