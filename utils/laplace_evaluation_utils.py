import torch
import copy
import numpy as np
from torch.nn.modules.loss import _WeightedLoss
from torch import Tensor
from typing import Optional
from torch.nn import functional as F
import time
import datetime
from netcal.metrics import ECE

def risk_evaluation_loop(dataloader, la_model, loss_fn,model_averaging=True,mode='posterior',n_samples=100):
    """
    This function estimates the average risk of the Laplace approximation for a dataset.

    Parameters
    ----------
    dataloader : A dataloader pointing to the the dataset for which we will the evaluate risk.
    la_model : The Laplace approximation model that we will evaluate.
    loss_fn : The loss function used to measure risk.
    model_averaging : Make predictions based on the Bayesian model average predictive or not.
    mode : Whether to sample from the prior or the posterior.
    n_samples = Number of samples when estimating the posterior.

    Returns
    -------
    test_loss : The risk of the Laplace approximation model over the dataset.
    """

    num_batches = len(dataloader)
    test_loss = 0

    device = la_model._device

    with torch.no_grad():
        for X, y in dataloader:

            X, y = X.to(device), y.to(device)

            if model_averaging == True:
                if la_model.likelihood == 'regression':
                    pred,_ = la_model(X,pred_type='nn',mode=mode,model_averaging=model_averaging,n_samples=n_samples)
                elif la_model.likelihood == 'classification':
                    pred = la_model(X, pred_type='nn', mode=mode, model_averaging=model_averaging,n_samples=n_samples)
                test_loss += loss_fn(pred, y).item()
            elif model_averaging == False:
                pred_samples = la_model(X, pred_type='nn', mode=mode, model_averaging=model_averaging,n_samples=n_samples)
                if la_model.likelihood == 'classification':
                    y = torch.reshape(y, (y.shape[0], 1))
                y = torch.reshape(y,(1,y.shape[0],y.shape[1]))
                y = y.repeat(pred_samples.shape[0],1,1)
                y = torch.reshape(y,(-1,y.shape[2]))
                if la_model.likelihood == 'classification':
                    y = torch.reshape(y, ([-1]))
                pred_samples = torch.reshape(pred_samples, (-1, pred_samples.shape[2]))
                test_loss += loss_fn(pred_samples, y).item()


    test_loss /= num_batches
    return test_loss

def risk_per_sample_loop(dataloader, la_model, loss_fn,mode='posterior'):
    """
    This function estimates and returns the per sample risk of the Laplace approximation for a dataset.

    Parameters
    ----------
    dataloader : A dataloader pointing to the the dataset for which we will the evaluate risk.
    la_model : The Laplace approximation model that we will evaluate.
    loss_fn : The loss function used to measure risk.
    mode : Sample from the prior or the posterior.

    Returns
    -------
    test_loss_no_reduction : The risk of the Laplace approximation model over the dataset, per sample.
    """

    num_batches = len(dataloader)
    loss_fn_noreduction = copy.deepcopy(loss_fn)
    loss_fn_noreduction.reduction='none'
    test_loss_no_reduction = []

    la_model.g_rand.seed()
    device = la_model._device

    with torch.no_grad():
        for X, y in dataloader:

                X, y = X.to(device), y.to(device)

                state = la_model.g_rand.get_state()
                if la_model.likelihood == 'regression':
                    pred,_ = la_model(X,pred_type='nn',mode=mode,n_samples=1)
                elif la_model.likelihood == 'classification':
                    pred = la_model(X, pred_type='nn', mode=mode, n_samples=1)
                la_model.g_rand.set_state(state)
                test_loss_no_reduction.append(loss_fn_noreduction(pred, y))

    test_loss_no_reduction = torch.cat(test_loss_no_reduction,dim=0)

    return test_loss_no_reduction

class metrics_different_temperatures:
    """
    This class estimates the risk of the Laplace approximation for a given dataset and at different temperatures
    \lambda.

    Parameters
    ----------
    dataloader : A dataloader pointing to the the dataset for which we will the evaluate risk.
    la_model : The Laplace approximation model that we will evaluate.
    """
    def __init__(self,dataloader, la_model):
        self.dataloader = dataloader
        self.la_model = la_model
        self.loss_fns = None
        self.model_averaging = None
        self.mode = None
        self.n_samples =None
        self.losses= None
        self.results = {}

    def estimate(self, loss_fns,loss_fn_names,mode='posterior',lambdas=None,n_samples=100):
        """
        Estimate risk at different temperatures

        Parameters
        ----------
        loss_fn : The loss function used to measure risk.
        model_averaging: Whether the risk is computed with model averaging.
        mode : Sample from the prior or the posterior.
        lambdas: The grid of lambdas over which to evaluate.
        """
        if lambdas == None:
            raise ValueError('Parameter \'lambdas\' cannot have value None.')
        for loss_name in loss_fn_names:
            if loss_name in list(self.results.keys()):
                raise ValueError('Loss '+ loss_name +' has already been estimated. Cannot reestimate loss.')

        self.loss_fns = loss_fns
        self.loss_fn_names = loss_fn_names
        self.mode = mode
        self.n_samples =n_samples

        #Initialize dictionary of results
        for loss_name in loss_fn_names:
            self.results[loss_name] = []


        for lambda_t in lambdas:


            self.la_model.temperature = lambda_t
            if self.la_model.likelihood == 'regression':
                self.metrics_regression()
            elif self.la_model.likelihood == 'classification':
                self.metrics_classification()

        for loss_name in loss_fn_names:
            self.results[loss_name] = torch.FloatTensor(self.results[loss_name])
            self.results[loss_name] = [lambdas.cpu().detach().numpy(),np.reshape(self.results[loss_name].cpu().detach().numpy(),(-1))]

        return

    def metrics_classification(self):
        """
        This function estimates metrics of the Laplace approximation for a dataset, when the Laplace approximation
        has been fitted for classification tasks.
        """

        preds = []
        ys = []

        device = self.la_model._device

        with torch.no_grad():
            for X, y in self.dataloader:

                X, y = X.to(device), y.to(device)

                pred = self.la_model(X, pred_type='nn', mode=self.mode, model_averaging=True,
                                n_samples=self.n_samples)
                preds.append(pred)
                ys.append(y)

        preds = torch.cat(preds, dim=0)
        ys = torch.cat(ys, dim=0)

        for loss_fn, loss_name in zip(self.loss_fns, self.loss_fn_names):
            self.results[loss_name].append(loss_fn(preds, ys).item())

        return

    def metrics_regression(self):
        """
        This function estimates metrics of the Laplace approximation for a dataset, when the Laplace approximation
        has been fitted for regression tasks.
        """

        preds = []
        ys = []

        device = self.la_model._device

        with torch.no_grad():
            for X, y in self.dataloader:

                X, y = X.to(device), y.to(device)

                pred = self.la_model(X, pred_type='nn', mode=self.mode, model_averaging=False,
                                   n_samples=self.n_samples)

                y = torch.reshape(y,(1,y.shape[0],y.shape[1]))
                y = y.repeat(pred.shape[0],1,1)

                preds.append(pred)
                ys.append(y)

        preds = torch.cat(preds, dim=1)
        ys = torch.cat(ys, dim=1)

        for loss_fn, loss_name in zip(self.loss_fns, self.loss_fn_names):
            self.results[loss_name].append(loss_fn(preds, ys).item())

        return


class NLLLoss_with_log_transform(_WeightedLoss):
    """
    Modification of the torch.nn.NLLLoss class such that the input
    probabilities are on log scale.
    """

    __constants__ = ['ignore_index', 'reduction']
    #ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super(NLLLoss_with_log_transform, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(torch.log(input), target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)

def zero_one_loss(pred,y):
    """
    The zero-one loss.
    """
    return torch.tensor((pred.argmax(1) != y).type(torch.float).sum().item()/pred.shape[0])

def ECE_wrapper(pred,y):
    """
    Wrapper for the ECE metric from the netcal package.
    """
    return torch.tensor(ECE(bins=15).measure(pred.cpu().detach().numpy(), y.cpu().detach().numpy()))


def Gaussian_nll_predictive(pred,y,sigma=1):
    """
    The Gaussian NLL at test time, given the definitions in the paper. We compute the NLL for each sample
    of the posterior, we then average and then apply the -log(x).
    """
    square_diff = torch.exp(-(1/2)*torch.square((pred-y)/sigma))/(sigma*2.5066282)
    if square_diff.dim() ==2:
        square_diff = torch.unsqueeze(square_diff,dim=0)
    mean = torch.mean(square_diff,dim=0)
    squeezed = torch.squeeze(mean)
    mean_per_sample = torch.mean(-torch.log(squeezed))
    return mean_per_sample


class Timer:
    """
    This function times loops and estimates the remaining time for the loop completion.
    """
    def __init__(self,total_iterations,round_s=True):
        self.total_iterations = total_iterations
        self.iteration=0
        self.start_time = 0
        self.current_time= 0
        self.round_s = round_s

    def time(self):
        if self.start_time==0:
            self.start_time = time.time()
        else:
            self.current_time = time.time()
            self.iteration+=1
            if self.round_s == True:
                average_time = int((self.current_time-self.start_time)/self.iteration)
            else:
                average_time = (self.current_time - self.start_time) / self.iteration
            print('Iteration: '+str(self.iteration)+', average execution time is: '+str(datetime.timedelta(seconds = average_time))+', remaining time is: '+str(datetime.timedelta(seconds = average_time*(self.total_iterations-self.iteration)))+'.')


