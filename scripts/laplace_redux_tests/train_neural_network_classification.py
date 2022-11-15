import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scripts.classification_datasets.mnist.mnist_torch_dataset import MNIST_split
# noinspection PyUnresolvedReferences
from model_classification.model_utils import NeuralNetwork, train_loop, test_loop

'''
In this script we train a simple neural network on the MNIST dataset. We then save the weights
of the network so as to learn a Laplace approximation from them.
'''

dir_MNIST = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/scripts/classification_datasets/mnist/dataset'

test,train_dataset,train_suffix,validation_dataset,true,true_suffix = MNIST_split(dir=dir_MNIST,mode='classification',suffix_size = 5000,validation_size=5000,seed_split=42)


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
loss_fn = nn.CrossEntropyLoss()

#Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


epochs = 10
for t in range(epochs):
    print("Epoch {:>d}\n-------------------------------".format(t+1))
    train_loop(train_dataloader, model, loss_fn, optimizer)
    loss,acc = test_loop(validation_dataloader, model, loss_fn)
    print("Validation Error: \n Avg loss: {:>8f} Acc {:>8f}".format(loss,acc) )
print("Done!")

torch.save(model.state_dict(),'model_classification/state_dict.pt')