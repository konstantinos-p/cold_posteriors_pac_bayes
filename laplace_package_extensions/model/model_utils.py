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
        nn.Flatten(),
        nn.Linear(8, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 1),
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
        loss = loss_fn(pred, y.float())

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
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.float())
            test_loss += loss_fn(pred, y.float()).item()

    test_loss /= num_batches
    correct /= size
    print("Validation Error: \n Avg loss: {:>8f} \n".format(test_loss))

