import numpy as np

'''
This file contains all functions used to compute the optimized Alquier bound, on which all approximations have been made.
(B_{approx} bound)
'''

def posterior_variance(lambda_bound,h,d,prior_variance):
    '''
    This function computes the optimal posterior variance \sigma^2_{\hat{\rho}}.

    Inputs:
        lambda_bound: The regularization parameter lambda
        h: The curvature of the loss landscape
        d: The number of parameters in the predictive model
        prior_variance: The prior variance
    Outputs:
        y: The optimal posterior variance
    '''

    y = 1/(2*lambda_bound*h/d+1/prior_variance)

    return y

def empirical_risk(lambda_bound,h,d,prior_variance):
    '''
    This function computes the 2nd order Taylor expansion
    approximation to the optimal empirical risk.

    Inputs:
        lambda_bound: The regularization parameter lambda
        h: The curvature of the loss landscape
        d: The number of parameters in the predictive model
        prior_variance: The prior variance
    Outputs:
        y: The 2nd order Taylor expansion approximation to the optimal empirical risk
    '''

    y = posterior_variance(lambda_bound,h,d,prior_variance)*h

    return y

def moment(lambda_bound,d,oracle_l2,epsilon_variance,x_variance,prior_variance):

    '''
    This function computes the upper bound to the moment term in the Alquier bound.
    Inputs:
        lambda_bound: The regularization parameter lambda
        d: The number of parameters in the predictive model
        oracle_l2: The oracle norm of the weights of the function generating the dataset labels.
        epsilon_variance: The noise of the generated labels.
        x_variance: The variance of the generating function gradients per sample.
        prior_variance: The prior variance
    Ouputs:
        y: Upper bound to the moment term
    '''
    y = x_variance*(prior_variance*d+oracle_l2)/(1-2*lambda_bound*x_variance*prior_variance)+epsilon_variance

    return y

def kl(lambda_bound,d,h,prior_variance,prior_posterior_distance,delta):
    '''
    This function computes the optimized KL term in the Alquier bound (based on a second order Taylor expansion of the
    loss).
    Inputs:
        lambda_bound: The regularization parameter lambda
        d: The number of parameters in the predictive model
        h: The curvature of the loss landscape
        prior_variance: The prior variance
        prior_posterior_distance: The squared l2 norm of the differnce between the prior and posterior weights.
        delta: The bound holds with probability 1-delta.
    Outputs:
        y: The KL term
    '''
    y = (d/prior_variance)*posterior_variance(lambda_bound,h,d,prior_variance)
    y += (1/prior_variance)*prior_posterior_distance
    y += -d-d*np.log(posterior_variance(lambda_bound,h,d,prior_variance))
    y += d*np.log(prior_variance)
    y = (1/2)*y
    y += np.log(1/delta)
    y = (1/lambda_bound)*y

    return y

def bound(lambda_bound,h,d,prior_variance,oracle_l2,epsilon_variance,x_variance,prior_posterior_distance,delta):
    '''
    This function computes the complete Alquier bound to which all possible approximations have been made. (B_{approx} bound).
    Inputs:
        lambda_bound: The regularization parameter lambda
        h: The curvature of the loss landscape
        d: The number of parameters in the predictive model
        prior_variance: The prior variance
        oracle_l2: The oracle norm of the weights of the function generating the dataset labels.
        epsilon_variance:  The noise of the generated labels.
        x_variance: The variance of the generating function gradients per sample.
        prior_posterior_distance: The squared l2 norm of the differnce between the prior and posterior weights.
        delta: The bound holds with probability 1-delta.
    Outputs:
        y: The approximate bound values.
    '''
    y = empirical_risk(lambda_bound,h,d,prior_variance)
    y += moment(lambda_bound,d,oracle_l2,epsilon_variance,x_variance,prior_variance)
    y += kl(lambda_bound,d,h,prior_variance,prior_posterior_distance,delta)

    return y