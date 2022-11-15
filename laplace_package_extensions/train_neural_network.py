import torch
from torch import nn
from torch.utils.data import DataLoader
from laplace_package_extensions.abalone_torch_dataset import AbaloneDataset
from laplace_package_extensions.model.model_utils import NeuralNetwork, train_loop, test_loop

'''
In this script we train a simple neural network on the proposed Abalone dataset. We then save the weights
of the network so as to learn a Laplace approximation from them.
'''


train_dataset = AbaloneDataset('abalone/train.csv')
validation_dataset = AbaloneDataset('abalone/validation.csv')


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
    test_loop(validation_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(),'model/state_dict.pt')