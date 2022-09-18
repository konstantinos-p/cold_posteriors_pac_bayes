import torch
from laplace_package_extensions.laplace_extension import Laplace
import numpy as np
from utils.laplace_evaluation_utils import risk_evaluation_loop, risk_per_sample_loop
import pickle


class bound_estimator_diagonal:
    """
    Baseclass for bound estimators.

    Parameters
    ----------
    model : The deterministic model.
    loss_fn : The loss function used for training the deterministic model.
    train_dataloader : The dataloader for the set used for bound evaluation, it could be
    the training set but in most cases it is the trainsuffix set.
    prior_variance : The prior variance of the Laplace approximation.
    delta : The bound holds with 1-delta probability, usually delta=0.05.
    likelihood : Whether the LA is fitted for classification or regression.
    prior_mean : The prior mean.
    """

    def __init__(self, model, loss_fn, train_dataloader, prior_variance=0.1,
                 delta=0.05, likelihood='regression', prior_mean=0.):

        self.effective_bound_dependencies = set()
        self.model = model
        self.device = next(model.parameters()).device
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.delta = delta
        self.likelihood = likelihood

        # To be estimated by running self.fit().
        self.n = 0
        self.d = 0
        self.h = 0
        self.norm_diff = 0
        self.map_emp_risk = 0
        self.la = None

        # To be set when running self.evaluate().
        self.lambdas = None
        self.n_samples_MC_emp = 0

        # To be modified when running self.evaluate().
        self.results = {}

        # To be set when computing the bound terms
        self.bound_terms = {}

        # Prior mean
        self.prior_mean = prior_mean

        # Prior variance affects and depends on other variables
        self.prior_variance = prior_variance

    @property
    def prior_mean(self):
        return self._prior_mean

    @prior_mean.setter
    def prior_mean(self, prior_mean):
        self._prior_mean = prior_mean
        if self.la != None:
            self.la.prior_mean = self.prior_mean
            self.norm_diff = torch.linalg.norm(self.la.posterior_mean - self.la.prior_mean) ** 2

    def estimate(self):
        """
        Computes the bounds.
        """
        return NotImplementedError

    def compute_terms(self):
        """
        Computes the bound terms for the given lambdas and adds them to the terms_dictionary.
        """
        return NotImplementedError

    def fit(self):
        '''
        Use the deterministic model to estimate important common variables for the bounds.
        '''
        # Estimate MAP risk
        self.map_emp_risk = self.test_loop(self.train_dataloader, self.model, self.loss_fn)

        # Fit Laplace approximation
        self.la = Laplace(self.model, likelihood=self.likelihood, prior_precision=1 / self.prior_variance,
                          subset_of_weights='all',
                          hessian_structure='isotropic', prior_mean=self.prior_mean)
        self.la.fit(self.train_dataloader)
        self.la.optimize_prior_precision()
        self.prior_variance = 1 /self.la.prior_precision

        # Get number of training data
        self.n = self.la.n_data

        # Get number of parameters
        self.d = self.la.n_params

        # Get curvature at minimum
        self.h = self.la.H[0] * self.la.n_data * self.la.n_params

        # Get norm of difference between prior and posterior
        self.norm_diff = torch.linalg.norm(self.la.posterior_mean - self.la.prior_mean) ** 2

        # Get norm of vectorised posterior weights and generator weights
        self.norm_generator = torch.linalg.norm(self.la.posterior_mean) ** 2

        return


    def empirical_risk(self):
        '''
        The empirical risk term.
        '''
        return self.map_emp_risk + (1 / (
        2 * self.lambdas.cpu().numpy() * self.h.cpu().numpy() / self.n + 1 / self.prior_variance)) * (
                                   self.h.cpu().numpy() / self.n)

    def kl(self):
        '''
        The KL term.
        '''
        a = (self.d / self.prior_variance) * (
        1 / (2 * self.lambdas.cpu().numpy() * self.h.cpu().numpy() / self.n + 1 / self.prior_variance))
        b = (1 / self.prior_variance) * self.norm_diff.cpu().numpy()
        c = -self.d - self.d * np.log(
            (1 / (2 * self.lambdas.cpu().numpy() * self.h.cpu().numpy() / self.n + 1 / self.prior_variance)))
        d = self.d * np.log(self.prior_variance)

        return (1 / (self.lambdas.cpu().numpy() * self.n)) * ((1 / 2) * (a + b + c + d) + np.log(1 / self.delta))

    def empirical_risk_MC(self):
        '''
        The empirical risk term estimated with Monte Carlo sampling.
        '''
        risks = []
        previous_temp = self.la.temperature
        for lambda_t in self.lambdas:
            self.la.temperature = lambda_t
            risks.append(risk_evaluation_loop(dataloader=self.train_dataloader, la_model=self.la, loss_fn=self.loss_fn,
                                              model_averaging=False
                                              , mode='posterior', n_samples=self.n_samples_MC_emp))

        self.la.temperature = previous_temp[0]

        risks = torch.FloatTensor(risks).cpu().numpy()
        risks = np.reshape(risks, (-1))

        return risks

    def test_loop(self, dataloader, model, loss_fn):
        '''
        Estimate the loss of the deterministic neural network.
        '''
        num_batches = len(dataloader)
        test_loss = 0

        device = next(model.parameters()).device

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)

                pred = model(X)
                test_loss += loss_fn(pred, y).item()

        test_loss /= num_batches
        return test_loss


class bound_estimator_alquier(bound_estimator_diagonal):
    """
    Estimator for the Alquier family of bounds

    Parameters
    ----------
    model : The deterministic model.
    loss_fn : The loss function used for training the deterministic model.
    prior_variance : The prior variance of the Laplace approximation.
    true_dataloader : The dataloader for the true dataset.
    train_dataloader : The dataloader for the set used for bound evaluation, it could be
    the training set but in most cases it is the trainsuffix set.
    dataset_aleatoric : The epsilon parameter, when modelling the data labeling function.
    delta : The bound holds with 1-delta probability, usually delta=0.05.
    likelihood : Whether the LA is fitted for classification or regression.
    prior_mean : The prior mean.
    """

    def __init__(self, model, loss_fn, true_dataloader, train_dataloader, prior_variance=0.1, dataset_aleatoric=1,
                 delta=0.05, likelihood='regression', prior_mean=0.):
        super().__init__(model=model, loss_fn=loss_fn, train_dataloader=train_dataloader, prior_variance=prior_variance,
                         delta=delta, likelihood=likelihood, prior_mean=prior_mean)
        self.bound_dependencies = {'approximate': {'empirical_risk', 'moment', 'kl'},
                                   'mixed': {'empirical_risk', 'moment_MC', 'kl'},
                                   'original': {'empirical_risk_MC', 'moment_MC', 'kl'}}

        self.true_dataloader = true_dataloader
        self.dataset_aleatoric = dataset_aleatoric

        # To be estimated by running self.fit().
        self.norm_generator = 0
        self.sigma_x = 0
        self.min_temperature_reduced = None
        self.max_temperature_reduced = None

        # To be set when running self.evaluate().
        self.bound_types = set()
        self.n_XY = 0
        self.n_f = 0
        self.multipliers = torch.FloatTensor()

    @property
    def prior_variance(self):
        return self._prior_variance

    @prior_variance.setter
    def prior_variance(self, prior_variance):
        self._prior_variance = prior_variance
        if self.min_temperature_reduced != None and self.max_temperature_reduced != None:
            self.min_temperature_reduced = 0
            self.max_temperature_reduced = 1 / (2 * self.n * self.sigma_x * self.prior_variance)
        if self.la != None:
            self.la.prior_precision = 1 / self.prior_variance

    def save(self, path):
        """
        This function saves the bound parameters at the 'path' location.
        """

        log = {
            'n': self.n,
            'd': self.d,
            'h': self.h,
            'norm_diff': self.norm_diff,
            'norm_generator': self.norm_generator,
            'sigma_x': self.sigma_x,
            'multipliers': self.multipliers.cpu().detach().numpy(),
            'map_emp_risk': self.map_emp_risk
        }
        for term in list(self.bound_terms.keys()):
            log[term] = self.bound_terms[term].cpu().detach().numpy()

        results_file = open(path, "wb")
        pickle.dump(log, results_file)
        results_file.close()

        return

    def fit(self):
        '''
        Use the deterministic model to estimate important common variables for the bounds.
        '''
        super().fit()

        # Get norm of vectorised posterior weights and generator weights
        self.norm_generator = torch.linalg.norm(self.la.posterior_mean) ** 2
        if self.likelihood == 'regression':
            # Get the weight variance of the per sample gradients.
            la_true = Laplace(self.model, likelihood='regression', prior_precision=1 / self.prior_variance,
                              subset_of_weights='all',
                              hessian_structure='isotropic')
            la_true.fit(self.true_dataloader)
            self.sigma_x = la_true.H[0]

            # Get reduced set of temperatures to evaluate based on the constraints of the B_approximate bound
            self.min_temperature_reduced = 0
            self.max_temperature_reduced = 1 / (2 * self.n * self.sigma_x * self.prior_variance)

        return

    def estimate(self, bound_types={'original'}
                 , n_XY=10, n_f=10, n_samples_MC_emp=100, min_temperature=0.1
                 , max_temperature=10, grid_lambda=100):
        """
        Computes the Alquier bound values.

        Parameters
        ----------
        bound_types: {'original','mixed','approximate'}
            The bound type to be estimated.
        n_samples_MC_emp: int
            The number of Monte Carlo samples used when estimating the empirical risk.
        min_temperature: float
            The minimum temperature to evaluate.
        max_temperature: float
            The maximum temperature to evaluate.
        grid_lambda: int
            The number of linspace samples between the minimum and the maximum temperature.
        n_XY: int
            The number of X,Y Monte Carlo samples used when estimating the moment term.
        n_f: int
            The number of f Monte Carlo samples used when estimating the moment term.
        """

        if bound_types.intersection({'approximate', 'mixed', 'original'}) == set():
            raise ValueError(
                'Invalid bounds types: ' + str(bound_types) + '. Bound types are: approximate, mixed and original.')
        elif 'approximate' in bound_types and ('mixed' in bound_types or 'original' in bound_types):
            raise ValueError(
                'Cannot compute approximate bound and mixed or original bound at the same time.')
        else:
            self.bound_types = bound_types.intersection({'approximate', 'mixed', 'original'})
        if 'approximate' in self.bound_types and self.likelihood == 'classification':
            raise ValueError('Cannot compute approximate bound when self.likelihood == \'classification\'.')

        self.effective_bound_dependencies = set.union(*list(map(self.bound_dependencies.get, self.bound_types)))
        if self.effective_bound_dependencies == set():
            raise ValueError('\'effective_bound_dependencies\' is an empty set. No bounds can be computed')

        self.n_XY = n_XY
        self.n_f = n_f
        self.n_samples_MC_emp = n_samples_MC_emp

        if 'approximate' in bound_types:
            self.lambdas = torch.linspace(self.min_temperature_reduced, self.max_temperature_reduced, grid_lambda).to(
                self.device)
            self.lambdas = self.lambdas[1:-1]
        else:
            self.lambdas = torch.linspace(min_temperature, max_temperature, grid_lambda).to(self.device)

        # Compute terms for the different bounds.
        self.compute_terms()

        for bound in self.bound_types:
            # Sum the components of each bound
            bound_vals = torch.stack(list(map(self.bound_terms.get, self.bound_dependencies[bound])), dim=0).sum(dim=0)
            # Save the lambda and bound values
            self.results[bound] = [self.lambdas.cpu().detach().numpy(), bound_vals.cpu().detach().numpy()]

        return

    def moment(self):
        '''
        The moment term.
        '''
        return self.sigma_x.cpu().numpy() * (self.prior_variance * self.d + self.norm_generator.cpu().numpy()) / (
        1 - 2 * self.lambdas.cpu().numpy() * self.n * self.sigma_x.cpu().numpy() * self.prior_variance) \
               + self.dataset_aleatoric

    def moment_MC(self):
        '''
        The moment term estimated with Monte Carlo sampling.
        '''

        multipliers = []
        for i in range(self.n_XY):

            risks = risk_per_sample_loop(self.true_dataloader, self.la, self.loss_fn, mode='prior')
            mean_risk = torch.mean(risks)
            for j in range(self.n_f):
                multipliers.append(mean_risk - torch.mean(risks[np.random.choice(len(risks), self.n, replace=False)]))

        self.multipliers = torch.FloatTensor(multipliers).to(self.device)
        self.multipliers = torch.reshape(self.multipliers, (1, -1))

        lambdas = torch.reshape(self.lambdas, (-1, 1))
        res = (1 / (self.n * np.reshape(lambdas.cpu().numpy(), (-1)))) * (np.log(1 / self.multipliers.shape[1])
                                                                          + torch.logsumexp(
            self.n * lambdas @ self.multipliers, dim=1).cpu().numpy())

        return res

    def compute_terms(self):
        """
        Computes the bound terms for the given lambdas and adds them to the terms_dictionary.
        """

        if 'kl' in self.effective_bound_dependencies:
            self.bound_terms['kl'] = torch.tensor(self.kl())
        if 'empirical_risk' in self.effective_bound_dependencies:
            self.bound_terms['empirical_risk'] = torch.tensor(self.empirical_risk())
        if 'empirical_risk_MC' in self.effective_bound_dependencies:
            self.bound_terms['empirical_risk_MC'] = torch.tensor(self.empirical_risk_MC())
        if 'moment' in self.effective_bound_dependencies:
            self.bound_terms['moment'] = torch.tensor(self.moment())
        if 'moment_MC' in self.effective_bound_dependencies:
            self.bound_terms['moment_MC'] = torch.tensor(self.moment_MC())


        return


class bound_estimator_catoni(bound_estimator_diagonal):
    """
    Estimator for the Catoni classification bound

    Parameters
    ----------
    model : The deterministic model.
    loss_fn : The loss function used for training the deterministic model.
    prior_variance : The prior variance of the Laplace approximation.
    train_dataloader : The dataloader for the set used for bound evaluation, it could be
    the training set but in most cases it is the trainsuffix set.
    delta : The bound holds with 1-delta probability, usually delta=0.05.
    prior_mean : The prior mean.
    """

    def __init__(self, model, loss_fn, train_dataloader, prior_variance=0.1,
                 delta=0.05, prior_mean=0.):
        super().__init__(model=model, loss_fn=loss_fn, train_dataloader=train_dataloader, prior_variance=prior_variance,
                         delta=delta, likelihood='classification', prior_mean=prior_mean)
        self.bound_dependencies = {'mixed': {'empirical_risk', 'kl'},
                                   'original': {'empirical_risk_MC', 'kl'}}

        # To be estimated by running self.fit().


        # To be set when running self.evaluate().
        self.bound_types = set()

    @property
    def prior_variance(self):
        return self._prior_variance

    @prior_variance.setter
    def prior_variance(self, prior_variance):
        self._prior_variance = prior_variance
        if self.la != None:
            self.la.prior_precision = 1 / self.prior_variance

    def save(self, path):
        """
        This function saves the bound parameters at the 'path' location.
        """

        log = {
            'n': self.n,
            'd': self.d,
            'h': self.h,
            'norm_diff': self.norm_diff,
            'map_emp_risk': self.map_emp_risk
        }
        for term in list(self.bound_terms.keys()):
            log[term] = self.bound_terms[term].cpu().detach().numpy()

        results_file = open(path, "wb")
        pickle.dump(log, results_file)
        results_file.close()

        return

    def estimate(self, bound_types={'original'}
                 , n_samples_MC_emp=100, min_temperature=0.1
                 , max_temperature=10, grid_lambda=100):
        """
        Computes the Catoni bound values.

        Parameters
        ----------
        bound_types: {'original','mixed'}
            The bound type to be estimated.
        n_samples_MC_emp: int
            The number of Monte Carlo samples used when estimating the empirical risk.
        min_temperature: float
            The minimum temperature to evaluate.
        max_temperature: float
            The maximum temperature to evaluate.
        grid_lambda: int
            The number of linspace samples between the minimum and the maximum temperature.
        """

        if bound_types.intersection({'mixed', 'original'}) == set():
            raise ValueError(
                'Invalid bounds types: ' + str(bound_types) + '. Bound types are: mixed, and original.')
        else:
            self.bound_types = bound_types.intersection({'approximate', 'mixed', 'original'})

        if 'approximate' in self.bound_types and self.likelihood == 'classification':
            raise ValueError('Cannot compute approximate bound when self.likelihood == \'classification\'.')

        self.effective_bound_dependencies = set.union(*list(map(self.bound_dependencies.get, self.bound_types)))
        if self.effective_bound_dependencies == set():
            raise ValueError('\'effective_bound_dependencies\' is an empty set. No bounds can be computed')

        self.n_samples_MC_emp = n_samples_MC_emp

        self.lambdas = torch.linspace(min_temperature, max_temperature, grid_lambda).to(self.device)

        # Compute terms for the different bounds.
        self.compute_terms()

        for bound in self.bound_types:
            # Sum the components of each bound
            linear_term_values = torch.stack(list(map(self.bound_terms.get, self.bound_dependencies[bound])),
                                             dim=0).sum(dim=0)
            bound_vals = self.nonlinear_catoni_function(linear_term_values, self.lambdas)
            # Save the lambda and bound values
            self.results[bound] = [self.lambdas.cpu().detach().numpy(), bound_vals.cpu().detach().numpy()]

        return

    def nonlinear_catoni_function(self,linear_term_values, lambdas):
        """
        This function computes the non-linear function in the Catoni bound
        Phi^{-1}_{beta}(x) = (1-exp(-beta*x))/(1-exp(-beta)).

        Parameters
        ----------
        linear_term_values: The values of the linear E L_{X,Y}(f)+(1/lambda*n)KL(hat{rho}|pi) term.
        lambdas: The temperature values.
        """
        return (1 - torch.exp(-lambdas * linear_term_values)) / (1 - torch.exp(-lambdas))

    def compute_terms(self):
        """
        Computes the bound terms for the given lambdas and adds them to the terms_dictionary.
        """

        if 'kl' in self.effective_bound_dependencies:
            self.bound_terms['kl'] = torch.tensor(self.kl())
        if 'empirical_risk' in self.effective_bound_dependencies:
            self.bound_terms['empirical_risk'] = torch.tensor(self.empirical_risk())
        if 'empirical_risk_MC' in self.effective_bound_dependencies:
            self.bound_terms['empirical_risk_MC'] = torch.tensor(self.empirical_risk_MC())

        return

