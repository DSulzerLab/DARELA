import sys
import autograd.numpy as anp
from autograd import elementwise_grad, jacobian
from autograd.scipy import stats
from alive_progress import alive_bar

# Lower constraint
def _lower_constraint(lower):
    def fun(x):
        return lower + anp.exp(x)
    return fun

# Interval constraint
def _interval_constraint(lower, upper):
    def fun(x):
        sigmoid = 1 / (1 + anp.exp(x))
        return lower + (upper - lower) * sigmoid
    return fun

# Shared parameter constraints and prior settings
shared_settings = {
    "pstf": [_interval_constraint(0, 0.025), [0.0065, 0.003]],
    "taustf": [_interval_constraint(5, 50), [7.5, 1]],
    "pstd": [_interval_constraint(-0.025, 0), [-0.0031, 0.0003]],
    "taustd": [_interval_constraint(5, 50), [30.833, 13.57]],
    "pltf": [_interval_constraint(0, 0.025), [0.0003, 0.0005]],
    "taultf": [_interval_constraint(600, 1200), [900, 30]],
    "pltd": [_interval_constraint(-0.025, 0), [-0.0003, 0.0005]],
    "taultd": [_interval_constraint(600, 1200), [900, 30]],
    "Vm": [_lower_constraint(2.0), [5.0, 1.5]],
    "Km": [_lower_constraint(0.1), [0.2, 0.01]],
    "kS": [_interval_constraint(0.8, 1), [0.9, 0.001]],
    "kE": [_interval_constraint(0.8, 1.2), [1.0, 0.06]],
    "kads1": [_interval_constraint(0, 0.2), [0.045, 0.011]],
    "kads2": [_interval_constraint(0, 0.2), [0.086, 0.054]],
    "kads3": [_interval_constraint(0, 0.2), [0.08125, 0.009]]
}

# Uniform release DAp settings (for SUR and STUR)
uniform_release = {
    "DAp": [_lower_constraint(0.2), [0.5, 0.1]]
}

# Discrete release DAp settings (for STDR)
discrete_release = {
    "DAp": [_lower_constraint(0.2), [2.29, 0.4]]
}

# Parameter fitting bank
# Functions to assist with ADVI computations
class FitParams:
    def __init__(self, params, discrete = False):
        # Unconstrained parameter values
        self.mu = anp.zeros(len(params))
        self.omega = anp.zeros(len(params))

        # Get parameter constraints and prior distribution settings
        settings = shared_settings.copy()
        if discrete: settings.update(discrete_release)
        else: settings.update(uniform_release)
        param_settings = list(zip(*[settings[key] for key in params]))
        self.constraints, prior_settings = param_settings
        self.prior_mu, self.prior_omega = list(zip(*prior_settings))
        self.prior_mu = anp.array(self.prior_mu)
        self.prior_omega = anp.array(self.prior_omega)

        # Gradient functions
        self.grad_transform = elementwise_grad(self.transform)
        self.grad_jac = elementwise_grad(self.jac)

        # Adaptive optimizer variables
        self.s_mu = 0
        self.s_omega = 0
        self.alpha = 0.1
        self.eta = 1
        self.i = 1
        self.tau = 1
        self.epsilon = 1e-16
    
    # Draw unconstrained parameter samples
    # η -> ζ
    def draw(self, eta: anp.ndarray) -> anp.ndarray:
        zeta = self.mu + eta * anp.exp(self.omega)
        return zeta
    
    # Transform unconstrained samples to constrained parameter space
    # ζ -> θ
    def transform(self, zeta: anp.ndarray) -> anp.ndarray:
        theta = anp.array([self.constraints[i](zeta[i]) for i in range(zeta.shape[0])])
        return theta
    
    # Get Gaussian priors of constrained samples
    # log p(θ)
    def priors(self, theta: anp.ndarray) -> anp.ndarray:
        return stats.norm.logpdf(theta, self.prior_mu, self.prior_omega).sum()
    
    # Get Jacobian of unconstrained samples
    # log|det J_T^-1(ζ)|
    def jac(self, zeta: anp.ndarray) -> anp.ndarray:
        det_jacobian = anp.linalg.det(jacobian(self.transform)(zeta))
        log_abs_det_jacobian = anp.log(anp.abs(det_jacobian))
        return log_abs_det_jacobian
    
    # Get entropy of samples
    # log q(ζ)
    def entropy(self) -> anp.ndarray:
        return (0.5 * anp.log(2 * anp.pi * anp.exp(self.omega)) + 0.5).sum()
    
    # Update parameters using gradients
    def update(self, grad_mu, grad_omega):
        if self.i == 1:
            self.s_mu = grad_mu ** 2
            self.s_omega = grad_omega ** 2
        self.s_mu = self.alpha * (grad_mu ** 2) + (1 - self.alpha) * self.s_mu
        self.s_omega = self.alpha * (grad_omega ** 2) + (1 - self.alpha) * self.s_omega

        rho_mu = self.eta * (self.i ** (-0.5 + self.epsilon)) * ((self.tau + anp.sqrt(self.s_mu)) ** -1)
        rho_omega = self.eta * (self.i ** (-0.5 + self.epsilon)) * ((self.tau + anp.sqrt(self.s_omega)) ** -1)

        self.mu += (rho_mu * grad_mu)
        self.omega += (rho_omega * grad_omega)

        self.i += 1

# Parameter fitting engine
# Uses automatic differential variational inference (ADVI)
class FitEngine:
    params: FitParams
    def __init__(self, model):
        self.model = model
        self.grad_log_joint = elementwise_grad(self.log_joint)
    
    # Initialize fitting parameters
    def load(self, data, time, params, discrete = False):
        self.data = data
        self.time = time
        for param in params:
            try:
                assert param in shared_settings or param in uniform_release
            except AssertionError:
                raise ValueError("{} is not a valid parameter name", param)

        self.params = FitParams(params, discrete)
    
    # Compute the log joint (comprised of log prior + log likelihood)
    def log_joint(self, theta):
        log_prior = self.params.priors(theta)
        DA = self.model._solve_fit(self.time, theta)
        log_likelihood = stats.norm.logpdf(self.data, DA, 0.05).mean()
        return log_prior + log_likelihood
    
    # Compute the evidence lower bound (ELBO)
    # M: number of Monte Carlo samples (1 by default)
    def ELBO(self, M = 1):
        elbo = 0
        grad_mu = 0
        grad_omega = 0
        # For each sample
        for _ in range(M):
            # Get transformed parameter values
            eta = anp.random.normal(size = self.params.mu.shape[0])
            zeta = self.params.draw(eta)
            theta = self.params.transform(zeta)

            # Compute ELBO components
            log_joint = self.log_joint(theta)
            log_jac = self.params.jac(zeta)
            entropy = self.params.entropy()
            elbo += (log_joint + log_jac + entropy)

            # Compute ELBO gradients
            grad_joint = self.grad_log_joint(theta)
            grad_transform = self.params.grad_transform(zeta)
            grad_jac = self.params.grad_jac(zeta)
            grad_mu += (grad_joint * grad_transform + grad_jac)
            grad_omega += (grad_mu * eta * anp.exp(self.params.omega) + 1)
        
        return elbo / M, grad_mu / M, grad_omega / M

    # Run ADVI parameter estimate
    # M: number of Monte Carlo samples (1 by default)
    def run(self, M = 1):
        # Initialize ELBO tracking variables
        previous_elbo = 1e5
        delta_elbo = 1e5
        epsilon = 0.01
        max_iter = 1000

        # While ELBO continues to improve
        with alive_bar(title = "Running parameter estimation...") as bar:
            while delta_elbo > epsilon and self.params.i < max_iter:
                # Compute ELBO and gradients
                elbo = anp.nan
                iterations = 0
                while anp.isnan(elbo) and iterations < 10:
                    elbo, grad_mu, grad_omega = self.ELBO(M)
                    iterations += 1
                
                # Exit fitting if ELBO is still NaN
                if anp.isnan(elbo):
                    sys.stderr.write("Unable to converge to optimal parameters. Try reducing the number of parameters for fitting.")
                    exit(-1)
                
                # Update ELBO tracker
                # print(f"Iteration {self.params.i}: {elbo}")
                delta_elbo = anp.abs(elbo - previous_elbo)
                previous_elbo = elbo
                
                # Use gradients to update parameters
                self.params.update(grad_mu, grad_omega)
                
                bar()
        
        # Exit fitting if ELBO did not converge after max_iter
        if self.params.i == max_iter:
            sys.stderr.write("Unable to converge to optimal parameters. Try reducing the number of parameters for fitting.")
            exit(-1)

        # Set converged parameters back in model
        zeta = self.params.mu.copy()
        theta = self.params.transform(zeta)
        self.model._set_fit(theta)
