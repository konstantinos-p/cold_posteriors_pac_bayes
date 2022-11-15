import torch
from torch.nn.utils import parameters_to_vector
from utils.model_utils import LeNet
from scripts.classification_datasets.mnist.mnist_torch_dataset import get_dataloaders
from laplace_package_extensions.laplace_extension import Laplace

'''
This script tests the outputs of the Laplace approximation for a very high value of \lambda
("frozen" essentially a deterministic network) and the actual deterministic network on which the Laplace approximation is fitted.
We see that there are some differences in the outputs between the deterministic network and the "frozen" Laplace
approximation model. We located the source of the difference in the laplace_package_extensions.baselapalce_extension file
in the function _nn_posterior_predictive_samples line 691. Whenever new samples are loaded in the model for evaluation
there are some small deviations from the actual deterministic values. These deviations, we believe, accumulate and make
the predictions between the deterministic and LA "frozen" model also different.
'''


#Load data and model
dir_MNIST= '/Users/Kostas/PycharmProjects/cold-warm-posteriors/scripts/classification_datasets/mnist/dataset'

test_dataloader,train_dataloader,train_suffix_dataloader,validation_dataloader,true_dataloader,true_suffix_dataloader = \
    get_dataloaders(dir=dir_MNIST,mode='classification',suffix_size = 5000,validation_size=5000,seed_split=42)

#Define model
model = LeNet()
model.load_state_dict(torch.load('MNIST/results/MNIST_lambda_0_to_1000/model_0/prior_mean.pt',map_location=torch.device('cpu')))

#Load datasamples
X,y = next(iter(test_dataloader))

#Make predictions (the LA baseclass applies a softmax internally, the deterministic model doesn't)
preds_model = model(X)
preds_model = torch.softmax(preds_model,dim=-1)


a  = parameters_to_vector(model.parameters()).detach()

#Fit Laplace approximation
la = Laplace(model, likelihood='classification', prior_precision=1,
                  subset_of_weights='all',
                  hessian_structure='isotropic', prior_mean=0)
la.fit(train_suffix_dataloader)


#Make \lambda very large
la.temperature = 10000000000000
preds_la = la(X, pred_type='nn', mode='posterior', model_averaging=True,
                     n_samples=100)

#Estimate difference between deterministic and LA "frozen" predictions
diff = preds_model.detach()-preds_la.detach()

end = 1