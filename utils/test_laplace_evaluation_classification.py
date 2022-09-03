from torch.utils.data import Dataset, DataLoader
from torch import nn

from scripts.laplace_redux_tests.model_classification.model_utils import load_model, test_loop
from scripts.classification_datasets.mnist.mnist_torch_dataset import MNIST_split

from utils.laplace_evaluation_utils import metrics_different_temperatures,NLLLoss_with_log_transform,zero_one_loss,ECE_wrapper
from utils.bound_evaluation_utils import bound_estimator

import pickle

'''
In this script we test the bound and laplace utils which are necessary for testing the cold-warm posterior effect.
both modules are used to compute the different bound type (B_approximate,B_mixed,B_original)
'''

#Set paths
dir_MNIST = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/scripts/classification_datasets/mnist/dataset'
model_dir = '/Users/Kostas/PycharmProjects/cold-warm-posteriors/scripts/laplace_redux_tests/model_classification/'


#Dataset
test,train_dataset,train_suffix,validation_dataset,true,true_suffix = \
    MNIST_split(dir=dir_MNIST,mode='classification',suffix_size = 5000,validation_size=5000,seed_split=42)



train_dataloader = DataLoader(train_dataset, batch_size=40,
                            shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=40,
                            shuffle=True)
true_dataloader = DataLoader(true, batch_size=40,
                            shuffle=True)
true_suffix_dataloader = DataLoader(true_suffix, batch_size=40,
                            shuffle=True)


model = load_model(model_dir+'state_dict.pt')

# Initialize the loss function
cross_entropy = nn.CrossEntropyLoss()
nll_with_log_tranform = NLLLoss_with_log_transform()

#Check that the network is indeed trained
loss,acc = test_loop(validation_dataloader, model, cross_entropy)
print("Validation Error: \n Avg loss: {:>8f} Acc {:>8f}".format(loss,acc) )

#Set hyperparameters
prior_variance=0.01
min_temperature=0.001
max_temperature=1000

#Fit bound estimator and set hyperparameters
bound_estimator = bound_estimator(model,nll_with_log_tranform,true_dataloader,train_dataloader,
                                  likelihood='classification')
bound_estimator.fit()
bound_estimator.prior_variance = prior_variance


#Estimate bounds
bound_estimator.estimate(bound_types={'original','mixed'}, grid_lambda=3,min_temperature=min_temperature,
                                   max_temperature=max_temperature,n_XY=1,n_f=2)

#Estimate validation set risk
risk_estimator = metrics_different_temperatures(validation_dataloader, bound_estimator.la)
risk_estimator.estimate([nll_with_log_tranform,ECE_wrapper,zero_one_loss],
                                                                ['nll','ECE','zero_one'],
                                                                mode='posterior',lambdas=bound_estimator.lambdas,
                                                                n_samples=100)


#Save data
all_results = {**bound_estimator.results, **risk_estimator.results}
results_file = open("results/classification/results.pkl", "wb")
pickle.dump(all_results, results_file)
results_file.close()


end = 1