from torch import nn
from scripts.classification_datasets.fashionmnist.fashionmnist_torch_dataset import get_dataloaders
from utils.multiple_standard_data_experiments_utils import estimate_prior_and_posterior_plain
import os
from utils.laplace_evaluation_utils import zero_one_loss,ECE_wrapper
from utils.model_utils import LeNet_nobatchnorm
import torch

'''
This script trains multiple ResNet networks on the fashionmnist dataset and saves them in new folders in the current directory.
'''

number_of_networks = 10
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
batch_size = 40
epochs = 10
image_transforms = True


#Directory in which to save the models
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/fashionmnist/results'

#Path to the svhn dataset
dir_fashionmnist= '/services/scratch/mistis/kpitas/projects/cold-warm-posteriors/scripts/classification_datasets/fashionmnist/dataset'

#Define dataloaders
test_dataloader,train_dataloader,validation_dataloader = get_dataloaders(dir=dir_fashionmnist,batch_size=batch_size,
                                                                         image_transforms=image_transforms)

#Get and set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Train Models
for i in range(number_of_networks):
    print('Starting new model.')
    os.chdir(path)
    folder_name = 'model_' + str(i)
    model = LeNet_nobatchnorm()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = None
    estimate_prior_and_posterior_plain(loss_fns=[loss_fn,ECE_wrapper,zero_one_loss],
                                     loss_fns_names=['nll','ECE','zero_one'],
                                     epochs=epochs,train_dataloader=train_dataloader,
                                     validation_dataloader=validation_dataloader,
                                     test_dataloader=test_dataloader,folder_name=folder_name,model=model,
                                     optimizer=optimizer)








