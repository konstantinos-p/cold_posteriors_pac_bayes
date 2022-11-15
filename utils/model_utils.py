from torch import nn
import torch
from torch.nn.utils import parameters_to_vector
from torch.optim.lr_scheduler import MultiplicativeLR

'''
In this small script we define the neural network classes used in the Laplace-Redux tests as
well as some helper functions.
'''

def NeuralNetwork(input_size,hidden1=100,hidden2=100):
    '''
    A small fully connected neural network.
    '''

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 1),
    )

def LeNet():
    '''
    The LeNet architecture.
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

def LeNet_nobatchnorm():
    '''
    The LeNet architecture.
    '''

    return nn.Sequential(

        nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.Flatten(),
        nn.Linear(256, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
        )

def CNN_nobatchnorm():
    '''
    A slightly larger CNN architecture than LeNet.
    '''

    return nn.Sequential(

        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1152, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
        )

def CNN_nobatchnorm_svhn():
    '''
    A slightly larger CNN architecture than LeNet.
    '''

    return nn.Sequential(

        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2048, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
        )

def train_loop(dataloader, model, loss_fn, optimizer,prior_mean=None,gamma=1):
    '''
    A training routine.
    '''
    loss_fn = loss_fn[0]
    size = len(dataloader.dataset)

    device = next(model.parameters()).device

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        if prior_mean == None:
            loss = loss_fn(pred, y)
        else:
            loss = loss_fn(pred, y)+gamma*torch.square(torch.linalg.norm(parameters_to_vector(model.parameters())-prior_mean,2))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print("loss: {:>7f}  [{:>5d}/{:>5d}]".format(loss,current,size))

def test_loop(dataloader, model, loss_fns,loss_fn_names):
    '''
    A testing routine.
    '''
    results = {}

    preds = []
    ys = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for X, y in dataloader:

            X,y = X.to(device),y.to(device)

            pred = model(X)
            preds.append(pred)
            ys.append(y)


    preds = torch.cat(preds, dim=0)
    ys = torch.cat(ys, dim=0)

    for loss_fn, loss_name in zip(loss_fns, loss_fn_names):
        results[loss_name] = loss_fn(preds, ys).item()

    return results

def train_with_epochs(train_dataloader,validation_dataloader, model, loss_fn,loss_fn_names, optimizer,epochs=10,
                      prior_mean=None,gamma=1,scheduler=None):
    '''
    A training routine for multiple epochs.
    '''
    for t in range(epochs):
        print("Epoch {:>d}\n-------------------------------".format(t + 1))
        train_loop(train_dataloader, model, loss_fn, optimizer,prior_mean=prior_mean,gamma=gamma)
        validation_losses = test_loop(validation_dataloader, model, loss_fn,loss_fn_names)
        print("Validation Error: \n")
        print_losses(validation_losses)
        if scheduler != None:
            scheduler.step()
    print("Done!")

def resnet_cifar10style_scheduler(optimizer,max_epochs):
    '''
    Returns a common scheduler for Resnets trained on cifar10. The learning rate is multiplied by 0.1 at the point of
    50%,75% and approximately 87% of the epochs.

    Parameters
    ----------
    optimizer: The torch optimizer used for training.
    max_epochs: The number of epochs used for training.

    Returns
    -------
    cifar10style_scheduler: the created scheduler.
    '''

    lmbda = lambda epoch: 0.1 if epoch == int(max_epochs*0.5) or epoch == int(max_epochs*0.75) or epoch == int(max_epochs*0.87) else 1

    return MultiplicativeLR(optimizer, lr_lambda=lmbda)

def resnet_cifar100style_scheduler(optimizer,max_epochs):
    '''
    Returns a common scheduler for Resnets trained on cifar10. The learning rate is multiplied by 0.1 at the point of
    50%,75% and approximately 87% of the epochs.

    Parameters
    ----------
    optimizer: The torch optimizer used for training.
    max_epochs: The number of epochs used for training.

    Returns
    -------
    cifar10style_scheduler: the created scheduler.
    '''

    lmbda = lambda epoch: 0.2 if epoch == int(max_epochs*0.30) or epoch == int(max_epochs*0.60) or epoch == int(max_epochs*0.80) else 1

    return MultiplicativeLR(optimizer, lr_lambda=lmbda)

def print_losses(losses):
    '''
    This function prints the losses that have been calculated using the test_loop function. The results correspond to
    a single dataloader and model but possibly multiple losses. The test_loop function returns a dictionary of the losses
    and their values. The losses that I am using are the
    zero-one loss named 'zero_one'
    negative log likelihood named 'nll'
    ECE (expected calibration error) named 'ECE'

    Parameters
    ----------
    losses: A dictionary with the losses, for a given model and dataloader. The possible keys are 'zero_one','nll' and
    'ECE'

    Returns
    -------

    '''

    for key in losses.keys():
        print("{}: {:>8f} \n".format(key,losses[key]))

    return


