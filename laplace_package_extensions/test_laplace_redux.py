from laplace_package_extensions.model.model_utils import load_model, test_loop
from laplace_package_extensions.abalone_torch_dataset import AbaloneDataset
from torch.utils.data import DataLoader
from torch import nn
from laplace_package_extensions.laplace_extension import Laplace

'''
Load a trained neural network compute the Laplace approximation and run some test.
'''

#Dataset
train_dataset = AbaloneDataset('abalone/train.csv')
validation_dataset = AbaloneDataset('abalone/validation.csv')


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
la = Laplace(model, likelihood='regression',prior_precision = 10000,subset_of_weights='all',hessian_structure ='isotropic')
la.fit(train_dataloader)

#Make some predictions
X_test,_ = next(iter(validation_dataloader))
f_mu, f_var = la(X_test,pred_type='nn',mode='posterior')

#Change temperature and prior precision and make some predictions
la.temperature = 0.1
la.prior_precision = 10
f_mu, f_var = la(X_test,pred_type='nn',mode='posterior')

#Generate samples from the predictive without computing the mean prediction
f = la(X_test,pred_type='nn',mode='posterior',model_averaging=False)



end = 1