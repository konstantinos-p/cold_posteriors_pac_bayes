import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scripts.regression_datasets.regression_torch_datasets import DiamondsDataset,data_split


'''
In this script we train a simple neural network on the proposed Diamonds dataset. Specifically in the proposed data
split we significantly reduce the number of training samples so as to be able to estimate the Moment term in the Alquier
bound. The goal of this script is therefore to establish that we are able to learn a useful regressor given the small
number of training samples.
'''


_,train_dataset,_,validation_dataset,_,_ = \
    data_split(DiamondsDataset('dataset/data.csv'),test_size=10788,train_size=10788+5394,suffix_size = 0,validation_size=5394,seed_split=42)


train_dataloader = DataLoader(train_dataset, batch_size=40,
                            shuffle=True)

validation_dataloader = DataLoader(validation_dataset, batch_size=40,
                            shuffle=True)


def NeuralNetwork():
    '''
    A small fully connected neural network.
    '''

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(9, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 1),
    )

model = NeuralNetwork()

#hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 5

# Initialize the loss function
loss_fn = nn.MSELoss()

#Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


#Training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.float())
        loss = loss_fn(pred, y.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print("loss: {:>7f}  [{:>5d}/{:>5d}]".format(loss,current,size))

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.float())
            test_loss += loss_fn(pred, y.float()).item()

    test_loss /= num_batches
    correct /= size
    print("Validation Error: \n Avg loss: {:>8f} \n".format(test_loss))


epochs = 10
for t in range(epochs):
    print("Epoch {:>d}\n-------------------------------".format(t+1))
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(validation_dataloader, model, loss_fn)
print("Done!")