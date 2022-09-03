# noinspection PyUnresolvedReferences
from model_classification.model_utils import NeuralNetwork, train_loop, test_loop,load_model
from scripts.classification_datasets.mnist.mnist_torch_dataset import MNIST_split
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


dir_MNIST = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/scripts/classification_datasets/mnist/dataset'

test,train_dataset,train_suffix,validation_dataset,true,true_suffix = \
    MNIST_split(dir=dir_MNIST,mode='classification',suffix_size = 5000,validation_size=5000,seed_split=42)


train_dataloader = DataLoader(train_dataset, batch_size=40,
                            shuffle=True)

validation_dataloader = DataLoader(validation_dataset, batch_size=40,
                            shuffle=True)



model = load_model('model_classification/state_dict.pt')

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

#Check that the network is indeed trained
loss, acc = test_loop(validation_dataloader, model, loss_fn)
print("Validation Error: \n Avg loss: {:>8f} Acc {:>8f}".format(loss, acc))

#Fit a Laplace approximation
la = Laplace(model, likelihood='classification',prior_precision = 100,subset_of_weights='all',hessian_structure ='diag')
la.fit(validation_dataloader)

#Make some predictions
X_test,_ = next(iter(validation_dataloader))
f_mu = la(X_test,pred_type='nn')

end = 1