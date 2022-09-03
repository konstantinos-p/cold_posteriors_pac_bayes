# noinspection PyUnresolvedReferences
from model.model_utils import load_model, test_loop
from scripts.regression_datasets.regression_torch_datasets import AbaloneDataset,data_split
from torch.utils.data import DataLoader
from torch import nn
from laplace import Laplace

'''
Load a trained neural network compute the Laplace approximation and run some test.

Notes:

when using the last layer version you can’t sample from the prior directly.
self.n_samples doesn’t have a value before applying LA fit

backpack rejects models that are not nn.sequential

last layer works even in more complex models because it determines layers in a different way

the mean is not set before applying fit

prior mean default value should be a vector the size of the parameters

It always returns the predictive distribution. In PAC-Bayes we need *samples* from the predictive, so as to average outside of the -loglikelihood

make explicit the generator of the random vectors of the LA so as to make the results reproducible

'''


#Dataset
_,train_dataset,_,validation_dataset,_,_ = \
    data_split(AbaloneDataset('/Users/Kostas/PycharmProjects/cold-warm-posteriors/scripts/regression_datasets'
                              '/abalone_dataset/dataset/data.csv'),test_size=835,train_size=835+418,suffix_size =
    0,validation_size=418,seed_split=42)


train_dataloader = DataLoader(train_dataset, batch_size=40,
                            shuffle=True)

validation_dataloader = DataLoader(validation_dataset, batch_size=40,
                            shuffle=True)


model = load_model('model/state_dict.pt')

# Initialize the loss function
loss_fn = nn.MSELoss()

#Check that the network is indeed trained
test_loop(validation_dataloader, model, loss_fn)

#Fit a Laplace approximation
la = Laplace(model, likelihood='regression',prior_precision = 100,subset_of_weights='all',hessian_structure ='diag')
la.fit(train_dataloader)

#Make some predictions
X_test,_ = next(iter(validation_dataloader))
f_mu, f_var = la(X_test,pred_type='nn')

end = 1