from torch import nn
import torch
'''
In this small script we define the neural network class used in the Laplace-Redux tests as
well as some helper functions.
'''

def NeuralNetwork():
    '''
    A small fully connected neural network.
    '''

    return nn.Sequential(

        nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(6),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.Flatten(),
        nn.Linear(256, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
        )




def load_model(path):
    '''
    A helper function that creates a neural network loads pretrained weights from path
    and returns the model
    Inputs:
        path: The path to the state dictionary file.
    Outputs:
        model: The model where we have loaded the pretrained weights.
    '''
    model = NeuralNetwork()
    model.load_state_dict(torch.load(path))
    model.eval()

    return model

#Training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    '''
    A training routine.
    '''
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.float())
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print("loss: {:>7f}  [{:>5d}/{:>5d}]".format(loss,current,size))

def test_loop(dataloader, model, loss_fn):
    '''
    A testing routine.
    '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss,correct = 0,0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    return test_loss,correct

