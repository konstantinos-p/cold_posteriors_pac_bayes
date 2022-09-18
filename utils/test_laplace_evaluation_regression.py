from torch.utils.data import DataLoader
from torch import nn

from laplace_package_extensions.model.model_utils import load_model, test_loop
from scripts.regression_datasets.regression_torch_datasets import AbaloneDataset,data_split

from utils.laplace_evaluation_utils import metrics_different_temperatures,Gaussian_nll_predictive
from utils.bound_evaluation_utils_old import bound_estimator

import pickle

'''
In this script we test the bound and laplace utils which are necessary for testing the cold-warm posterio effect.
both modules are used to compute the different bound type (B_approximate,B_mixed,B_original)
'''

#Set paths
abalone_dir = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/scripts/regression_datasets/abalone_dataset/dataset/data.csv'
model_dir = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/scripts/laplace_redux_tests/model_regression/'


#Dataset
test,train_dataset,train_suffix,validation_dataset,true_dataset,true_suffix = \
    data_split(AbaloneDataset(abalone_dir),test_size=835,train_size=835+418,suffix_size =
    0,validation_size=418,seed_split=42)


train_dataloader = DataLoader(train_dataset, batch_size=40,
                            shuffle=True)

validation_dataloader = DataLoader(validation_dataset, batch_size=40,
                            shuffle=True)
true_dataloader = DataLoader(true_dataset, batch_size=40,
                            shuffle=True)


model = load_model(model_dir+'state_dict.pt')

# Initialize the loss function
loss_fn = nn.MSELoss()

#Check that the network is indeed trained
print('MAP validation error')
test_loop(validation_dataloader, model, loss_fn)

#Set hyperparameters
prior_variance=0.1
min_temperature=0.1
max_temperature=10

#Fit bound estimator and set hyperparameters
bound_estimator = bound_estimator(model,loss_fn,true_dataloader,train_dataloader)
bound_estimator.fit()
bound_estimator.prior_variance=prior_variance

#Estimate bounds
#Estimate the B_approx bound
bound_estimator.estimate(bound_types={'approximate'}, grid_lambda=20,min_temperature=min_temperature,
                                    max_temperature=max_temperature)

#Estimate the B_mixed bound
bound_estimator.estimate(bound_types={'mixed'}, grid_lambda=20,min_temperature=min_temperature,
                                   max_temperature=max_temperature)

#Estimate the B_orig bound
bound_estimator.estimate(bound_types={'original'}, grid_lambda=20,min_temperature=min_temperature,
                                  max_temperature=max_temperature)

#Estimate validation set risk
risk_estimator = metrics_different_temperatures(validation_dataloader, bound_estimator.la)
risk_estimator.estimate([Gaussian_nll_predictive], ['GaussianNLL'],mode='posterior',lambdas=bound_estimator.lambdas,
                                                                n_samples=100)


#Save hyperparameters
bound_estimator.save('results/regression/bound_log.pkl')

#Save data
all_results = {**bound_estimator.results, **risk_estimator.results}
results_file = open("results/regression/results.pkl", "wb")
pickle.dump(all_results, results_file)
results_file.close()

end = 1