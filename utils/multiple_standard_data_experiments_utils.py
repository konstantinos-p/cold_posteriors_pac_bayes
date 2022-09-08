from utils.model_utils import train_with_epochs,NeuralNetwork,test_loop
import torch
from glob import glob
import os
from torch.nn.utils import parameters_to_vector
import pickle
from utils.laplace_evaluation_utils import metrics_different_temperatures,Timer
import numpy as np
from utils.bound_evaluation_utils import bound_estimator
from laplace.curvature import AsdlGGN
from laplace_package_extensions.laplace_extension import Laplace
from os import path

'''
This file contains scripts that help train multiple deep neural networks on the regression datasets.
'''

def estimate_prior_and_posterior_Roy(loss_fn_train,loss_fn_test,loss_fn_test_names,epochs_prefix,epochs_suffix,train_suffix_dataloader,
                                     train_dataloader,validation_dataloader,test_dataloader,folder_name,model,optimizer):
    '''
    This function estimates a prior and posterior mean to be used later in the Laplace approximation. The approach is
    similar to the one in @article{dziugaite2020role,
    title={On the role of data in PAC-Bayes bounds},
    author={Dziugaite, Gintare Karolina and Hsu, Kyle and Gharbieh, Waseem and Roy, Daniel M},
    journal={arXiv preprint arXiv:2006.10929},
    year={2020}
    }

    Parameters
    ----------
    loss_fn_train: torch Function
        The loss function used for training.
    loss_fn_test: Python List
        The loss function used for testing.
    loss_fn_test_names: Python List of Strings
        The names of the loss functions used.
    epochs_prefix: float
        The number of epochs used for training the prefix (this coincides with the training set as usually defined)
    epochs_suffix: float
        The number of epochs used for training the suffix.
    train_suffix_dataloader: torch.Dataloader
        The Dataloader of the training suffix set.
    train_dataloader:   torch.Dataloader
        The Dataloader of the training set.
    validation_dataloader: torch.Dataloader
        The Dataloader of the validation set.
    test_dataloader:    torch.Dataloader
        The Dataloader of the test set.
    folder_name:    string
        Full path to folder used for saving the results.
    model:  torch.Sequential.model
        The deterministic model for which we want to find MAP estimates
    optimizer:  torch.optimizer
        The optimizer used for training.
    '''

    # Create folder if it doesn't exist yet.

    if not folder_name + '/' in glob("*/"):
        os.mkdir(folder_name)
    else:
        raise FileExistsError('The folder ' + folder_name + ' already exists in directory ' + os.getcwd())

    # Train on prefix
    train_with_epochs(train_dataloader, validation_dataloader, model, loss_fn_train,loss_fn_names=['train_loss'],
                      optimizer=optimizer, epochs=epochs_prefix,prior_mean=None,gamma=1)
    test_loss_prior = test_loop(dataloader=test_dataloader,model=model,loss_fns=loss_fn_test,loss_fn_names=loss_fn_test_names)
    # Save prior mean and turn it into vector.
    torch.save(model.state_dict(), folder_name + '/prior_mean.pt')
    prior_mean = parameters_to_vector(model.parameters()).detach()

    # Finetune on suffix
    train_with_epochs(train_suffix_dataloader, validation_dataloader, model, loss_fn_train,loss_fn_names=['train_loss'],
                      optimizer=optimizer, epochs=epochs_suffix,prior_mean=prior_mean,gamma=1)
    test_loss_posterior = test_loop(dataloader=test_dataloader, model=model,loss_fns=loss_fn_test,loss_fn_names=loss_fn_test_names)
    # Save posterior mean.
    torch.save(model.state_dict(), folder_name + '/posterior_mean.pt')

    #Save log for model
    log_model = {'test_loss_prior':test_loss_prior,'test_loss_posterior':test_loss_posterior}
    log_file = open(folder_name + '/model_log.pkl', "wb")
    pickle.dump(log_model, log_file)
    log_file.close()

    return


def estimate_all_bounds(prior_variance,true_dataloader,train_suffix_dataloader,test_dataloader,
                        grid_lambda,min_temperature,max_temperature,n_samples,model,likelihood,loss_fn_bound,loss_functions_test,
                        loss_functions_test_names,n_f=2,n_XY=2):
    """
    This script finds all models in a given folder estimates the different bounds and saves them in the same folder.

    Parameters
    ----------
    prior_variance: torch.tensor
        The prior variances for which to estimate the bounds.
    true_dataloader: torch.Dataloader
        The Dataloader of the true set.
    train_dataloader:   torch.Dataloader
        The Dataloader of the training set.
    test_dataloader:    torch.Dataloader
        The Dataloader of the test set.
    grid_lambda: float
        The number of samples between min_temperature and max_temperature for which the bounds are evaluated.
    min_temperature: float
        The minimum value of parameter \lambda.
    max_temperature: float
        The maximum value of parameter \lambda.
    n_samples: float
        The number of Monte Carlo samples when estimating the test risk.
    model:  torch.Sequential.model
        The deterministic model on which we will fit our LA and estimate bounds.
    likelihood: {'regression','classification'}
        Whether we are in regression or classification mode. Required by the Laplace-Redux package.
    loss_fn_bound:  Python list
        The loss used for training the deterministic model.
    loss_functions_test: Python list
        The losses used for testing.
    loss_functions_test_names:  Python list of strings
        The names of the loss functions used for testing.
    n_f:    float
        The number of samples from the posterior when estimating the Moment term by MC.
    n_XY:   float
        The number of samples from the true data distribution when estimating the Moment term by MC. Note that
        each sample is actually X,Y each of size #train or #trainsuffix (usually the lastone).
    """

    timer = Timer(len(glob("*/")))
    folders = glob("*/")
    folders.sort()
    for folder in folders:
        timer.time()

        # Load prior mean
        model.load_state_dict(torch.load(folder + 'prior_mean.pt'))
        prior_mean = parameters_to_vector(model.parameters()).detach()

        # Load posterior mean
        model.load_state_dict(torch.load(folder + 'prior_mean.pt'))

        for prior_var, iter in zip(prior_variance, np.arange(len(prior_variance))):
            print('Variance #'+str(iter)+'/'+str(len(prior_variance)))
            bound_estimator_obj = bound_estimator(model, loss_fn_bound[0], true_dataloader, train_suffix_dataloader,
                                                  prior_mean=prior_mean,likelihood=likelihood)
            bound_estimator_obj.fit()
            bound_estimator_obj.prior_variance = prior_var.item()

            if likelihood == 'regression':
                # Estimate the B_approximate bound
                bound_estimator_obj.estimate(bound_types={'approximate'}, grid_lambda=grid_lambda,
                                             min_temperature=min_temperature,
                                             max_temperature=max_temperature,n_f=n_f,n_XY=n_XY)

            # Estimate the B_mixed and B_original bound
            bound_estimator_obj.estimate(bound_types={'mixed', 'original'}, grid_lambda=grid_lambda,
                                         min_temperature=min_temperature,
                                         max_temperature=max_temperature,n_f=n_f,n_XY=n_XY)

            # Estimate validation set risk
            risk_estimator = metrics_different_temperatures(test_dataloader, bound_estimator_obj.la)
            risk_estimator.estimate(loss_functions_test,loss_functions_test_names , mode='posterior',
                                    lambdas=bound_estimator_obj.lambdas,
                                    n_samples=n_samples)

            # Save hyperparameters
            bound_estimator_obj.save(folder + 'bound_log_' + str(iter) + '.pkl')

            # Save results
            all_results = {**bound_estimator_obj.results, **risk_estimator.results}
            results_file = open(folder + 'results_' + str(iter) + '.pkl', "wb")
            pickle.dump(all_results, results_file)
            results_file.close()

def estimate_prior_and_posterior_plain(loss_fns,loss_fns_names,epochs,
                                     train_dataloader,validation_dataloader,test_dataloader,folder_name,model,optimizer,scheduler=None):
    '''
    This function estimates a prior and posterior mean to be used later in the Laplace approximation. It follows the
    standard train,validation,test splits as well as training procedure from the literature.


    Parameters
    ----------
    loss_fns: Python List
        The loss functions.
    loss_fns_names: Python List of Strings
        The names of the loss functions.
    epochs: float
        The number of epochs used for training
    train_dataloader:   torch.Dataloader
        The Dataloader of the training set.
    validation_dataloader: torch.Dataloader
        The Dataloader of the validation set.
    test_dataloader:    torch.Dataloader
        The Dataloader of the test set.
    folder_name:    string
        Full path to folder used for saving the results.
    model:  torch.Sequential.model
        The deterministic model for which we want to find MAP estimates
    optimizer:  torch.optimizer
        The optimizer used for training.
    scheduler: torch.optim.lr_scheduler
        The scheduler used to adjust the learning rate during training.

    '''

    # Create folder if it doesn't exist yet.

    if not folder_name + '/' in glob("*/"):
        os.mkdir(folder_name)
    else:
        raise FileExistsError('The folder ' + folder_name + ' already exists in directory ' + os.getcwd())

    # Save prior mean and turn it into vector.
    torch.save(model.state_dict(), folder_name + '/prior_mean.pt')

    # Train on prefix
    train_with_epochs(train_dataloader, validation_dataloader, model, loss_fns,loss_fn_names=loss_fns_names,
                      optimizer=optimizer, epochs=epochs,prior_mean=None,gamma=1,scheduler=scheduler)

    train_loss_posterior = test_loop(dataloader=train_dataloader, model=model,loss_fns=loss_fns,loss_fn_names=loss_fns_names)
    test_loss_posterior = test_loop(dataloader=test_dataloader, model=model,loss_fns=loss_fns,loss_fn_names=loss_fns_names)
    validation_loss_posterior = test_loop(dataloader=validation_dataloader, model=model,loss_fns=loss_fns,loss_fn_names=loss_fns_names)

    # Save posterior mean.
    torch.save(model.state_dict(), folder_name + '/posterior_mean.pt')

    #Save log for model
    log_model = {'train_loss_posterior': train_loss_posterior,
                 'test_loss_posterior': test_loss_posterior,
                 'validation_loss_posterior': validation_loss_posterior}

    log_file = open(folder_name + '/model_log.pkl', "wb")
    pickle.dump(log_model, log_file)
    log_file.close()

    return

def estimate_all_metrics_plain(train_dataloader,test_dataloader,validation_dataloader,model,likelihood,loss_functions_test,
                        loss_functions_test_names,grid_lambda=100,min_temperature=0.1,max_temperature=100,
                        grid_prior_variance=None,min_prior_variance=0.0001,max_prior_variance=1,n_samples=100,
                        hessian_structure='kron',subset_of_weights='all'):

    """
    This script finds all models in a given folder estimates the different metrics for varying temperature levels
    and saves them in the same folder.

    Parameters
    ----------
    prior_variance: torch.tensor
        The prior variances for which to estimate the bounds.
    train_dataloader:    torch.Dataloader
        The Dataloader of the train set.
    test_dataloader:    torch.Dataloader
        The Dataloader of the test set.
    validation_dataloader:   torch.Dataloader
        The Dataloader of the training set.
    grid_lambda: float
        The number of samples between min_temperature and max_temperature for which the bounds are evaluated.
    min_temperature: float
        The minimum value of parameter \lambda.
    max_temperature: float
        The maximum value of parameter \lambda.
    grid_prior_variance: float,None
        The number of samples between min_temperature and max_temperature for which the bounds are evaluated.
    min_prior_variance: float
        The minimum value of parameter \lambda.
    max_prior_variance: float
        The maximum value of parameter \lambda.
    n_samples: float
        The number of Monte Carlo samples when estimating the test risk.
    model:  torch.Sequential.model
        The deterministic model on which we will fit our LA and estimate bounds.
    likelihood: {'regression','classification'}
        Whether we are in regression or classification mode. Required by the Laplace-Redux package.
    loss_functions_test: Python list
        The losses used for testing.
    loss_functions_test_names:  Python list of strings
        The names of the loss functions used for testing.
    subset_of_weights: 'all'
        Which weights to fit the Laplace approximation over.
    """

    timer = Timer(len(glob("*/")))
    folders = glob("*/")
    folders.sort()

    lambdas = torch.linspace(min_temperature, max_temperature, grid_lambda).to(next(model.parameters()).device)

    for folder in folders:
        timer.time()

        # Load posterior mean
        model.load_state_dict(torch.load(folder + 'posterior_mean.pt'))
        la = Laplace(model.eval(), likelihood=likelihood, prior_precision=1, subset_of_weights=subset_of_weights,
                     hessian_structure=hessian_structure, backend=AsdlGGN)
        la.fit(train_dataloader)

        if grid_prior_variance ==None:
            la.optimize_prior_precision()
            prior_variances = torch.tensor(1/la.prior_precision)
        else:
            prior_variances = torch.linspace(min_prior_variance, max_prior_variance, grid_prior_variance).to(next(model.parameters()).device)

        results = []
        for prior_var, iter in zip(prior_variances, np.arange(len(prior_variances))):

            print('Variance #'+str(iter)+'/'+str(len(prior_variances)))
            #bound_estimator_obj.prior_variance = prior_var.item()
            la.prior_precision = 1/prior_var.item()

            # Estimate test set risk
            risk_estimator = metrics_different_temperatures(test_dataloader, la)
            risk_estimator.estimate(loss_functions_test,loss_functions_test_names , mode='posterior',
                                    lambdas=lambdas,
                                    n_samples=n_samples)
            test_metrics = risk_estimator.results

            # Estimate validation set risk
            risk_estimator = metrics_different_temperatures(validation_dataloader, la)
            risk_estimator.estimate(loss_functions_test, loss_functions_test_names, mode='posterior',
                                    lambdas=lambdas,
                                    n_samples=n_samples)
            validation_metrics = risk_estimator.results

            results.append({'test':test_metrics,'validation':validation_metrics,'prior_var':prior_var.item()})

        # Save results
        results_file = open(folder + 'la_metrics.pkl', "wb")
        pickle.dump(results, results_file)
        results_file.close()

        #Save readme file
        path.exists(folder+'hyperparameters.txt')
        with open('hyperparameters.txt', 'w') as f:
            f.write('The results were obtained with the following hyperparameters \n\n grid_lambda: {} \n min_temperature: {} \n max_temperature: {} \n grid_prior_variance: {} \n min_prior_variance: {} \n max_prior_variance: {} \n n_samples: {} \n hessian_structure: {} \n subset_of_weights: {}' .format(grid_lambda,
                        min_temperature,max_temperature,grid_prior_variance,min_prior_variance,max_prior_variance,
                        n_samples,hessian_structure,subset_of_weights))

