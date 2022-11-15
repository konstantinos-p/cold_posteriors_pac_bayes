import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scripts.regression_datasets.regression_torch_datasets import AbaloneDataset,data_split
# noinspection PyUnresolvedReferences
from model.model_utils import NeuralNetwork, train_loop, test_loop

'''
In this script we train a simple neural network on the proposed Abalone dataset. We then save the weights
of the network so as to learn a Laplace approximation from them.
'''


_,train_dataset,_,validation_dataset,_,_ = \
    data_split(AbaloneDataset('/Users/Kostas/PycharmProjects/cold-warm-posteriors/scripts/regression_datasets/'
                              'abalone_dataset/dataset/data.csv'),test_size=835,train_size=835+418,suffix_size = 0,
               validation_size=418,seed_split=42)


train_dataloader = DataLoader(train_dataset, batch_size=40,
                            shuffle=True)

validation_dataloader = DataLoader(validation_dataset, batch_size=40,
                            shuffle=True)



model = NeuralNetwork()

#hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 5

# Initialize the loss function
loss_fn = nn.MSELoss()

#Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


epochs = 10
for t in range(epochs):
    print("Epoch {:>d}\n-------------------------------".format(t+1))
    train_loop(train_dataloader, model, loss_fn, optimizer)
    print("Validation Error: \n Avg loss: {:>8f} \n".format(test_loop(validation_dataloader, model, loss_fn)) )
print("Done!")

torch.save(model.state_dict(),'model_regression/state_dict.pt')