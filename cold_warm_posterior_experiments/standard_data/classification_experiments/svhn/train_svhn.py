from torch import nn
from scripts.classification_datasets.svhn.svhn_torch_dataset import get_dataloaders
from utils.multiple_standard_data_experiments_utils import estimate_prior_and_posterior_plain
import os
from utils.laplace_evaluation_utils import zero_one_loss,ECE_wrapper
from utils.wide_resnet_utils import FixupWideResNet
import torch
from utils.model_utils import resnet_cifar10style_scheduler

'''
This script trains multiple ResNet networks on the svhn dataset and saves them in new folders in the current directory.
'''

number_of_networks = 1
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-1
batch_size = 128
epochs = 300
image_transforms = False
weight_decay = 5e-4

#Directory in which to save the models
path = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/cold_warm_posterior_experiments/standard_data/' \
       'classification_experiments/svhn/results'

#Path to the svhn dataset
dir_svhn= '/Users/Kostas/PycharmProjects/cold-warm-posteriors/scripts/classification_datasets/svhn/dataset'

#Define dataloaders
test_dataloader,train_dataloader,validation_dataloader = get_dataloaders(dir=dir_svhn,batch_size=batch_size,
                                                                         image_transforms=image_transforms)

#Get and set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Train Models
for i in range(number_of_networks):
    print('Starting new model.')
    os.chdir(path)
    folder_name = 'model_' + str(i)
    model = FixupWideResNet(22, 4, 10, dropRate=0.4)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=weight_decay)
    scheduler = resnet_cifar10style_scheduler(optimizer=optimizer,max_epochs=epochs)
    estimate_prior_and_posterior_plain(loss_fns=[loss_fn,ECE_wrapper,zero_one_loss],
                                     loss_fns_names=['nll','ECE','zero_one'],
                                     epochs=epochs,train_dataloader=train_dataloader,
                                     validation_dataloader=validation_dataloader,
                                     test_dataloader=test_dataloader,folder_name=folder_name,model=model,
                                     optimizer=optimizer,scheduler=scheduler)








