from torch import nn
import torch
from utils.model_utils import train_with_epochs
from laplace import Laplace
from utils.model_utils import LeNet_nobatchnorm
from scripts.classification_datasets.fashionmnist.fashionmnist_torch_dataset import get_dataloaders
from utils.laplace_evaluation_utils import zero_one_loss

number_of_networks = 10
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
epochs = 10

'''
In this script we test the K-FAC Laplace approximation for the fashionmnist dataset.
We use it to make a rough estimate of the time requirements for optimizing the DNN and computing the Laplace approximation.
We check whether the Laplace approximation can be computed, as Resnets typically contain Batchnorm layers which can't be handled
by the Laplace-Redux package. Instead here we use a small FixupResnet.
'''


dir_fashionmnist= '/services/scratch/mistis/kpitas/projects/cold-warm-posteriors/scripts/classification_datasets/fashionmnist/dataset'

test_dataloader,train_dataloader,validation_dataloader = get_dataloaders(dir=dir_fashionmnist)

model = LeNet_nobatchnorm()

#Get and set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)

train_with_epochs(train_dataloader,validation_dataloader, model, [loss_fn,zero_one_loss],['nll','zero_one'],
                  optimizer,epochs=epochs,prior_mean=None,gamma=1)

la = Laplace(model.eval(), likelihood='classification',prior_precision = 1,subset_of_weights='all',hessian_structure='kron')

la.fit(train_dataloader)

end = 1