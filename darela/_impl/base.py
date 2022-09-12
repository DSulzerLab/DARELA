import autograd.numpy as anp
from .fit import FitEngine

# Base model constructor class
# Used as constructor for ODEMOdel and PDEModel
class BaseModel:
    dt: float # Discretization time step
    kGamma = 1 # DA adsorption rate constant
    params: dict # Parameter dictionary

    # Base initialization
    def __init__(self, kinetics: bool = True):
        self.kinetics = kinetics
        self.fit_engine = FitEngine(self)
        self.params = {}

    # FSCV experiment parameters
    def initialize(self, bursts: list, experiment_params: list):
        self.ti = bursts.copy() # Burst start durations
        self.f = experiment_params[0] # Stimulus frequency
        self.NP = experiment_params[1] # Number of pulses ber burst
        self.I = experiment_params[2] # Stimulus current
    
    # Load model release parameters 
    def release_params(self, params: dict):
        self.params.update(params)
    
    # Load model kinetic parameters
    def kinetic_params(self, params: dict):
        # Assert that kinetic types are specified
        assert 'ktypes' in params 

        # Assert that length of kick components/time constants are equal
        # to number of kinetic types specified
        assert len(params['ktypes']) == len(params['k'])
        assert len(params['ktypes']) == len(params['tau'])

        # Load each kick component/time constant as separate parameter from lists
        self.params['ktypes'] = params['ktypes'].copy()
        for i in range(len(params['ktypes'])):
            k_key = f'k{i + 1}'
            tau_key = f'tau{i + 1}'
            self.params[k_key] = params['k'][i]
            self.params[tau_key] = params['tau'][i]

    # Fit requested parameters to given FSCV trace
    def fit(self, data: anp.ndarray, time: anp.ndarray, params: list, discrete = False):
        # Mark parameters which need to be fit
        self.fit_params = params.copy()
        
        # Adjust kinetic parameters to indicate short-term/long-term or facilitation/depression
        adjusted_params = params.copy()
        kinetic_names = ['k', 'tau']
        for i in range(len(adjusted_params)):
            if adjusted_params[i][:-1] in kinetic_names and adjusted_params[i][-1].isnumeric():
                index = int(adjusted_params[i][-1]) - 1
                adjusted_params[i] = adjusted_params[i][:-1] + self.params['ktypes'][index]
        
        # Run fitting engine
        self.fit_engine.load(data, time, adjusted_params, discrete)
        self.fit_engine.run()
    
    # Solve kinetics ODE separately
    # Meant to be used for long-term analysis
    def solve_kinetics(self, ic: anp.ndarray, time: float, experiment_params: list, bursts = None):
        # Initialize stimulation
        self.f = experiment_params[0] # Stimulus frequency
        self.NP = experiment_params[1] # Number of pulses ber burst
        self.dt = 1
        if bursts is not None:
            self.dt /= self.f
            self.ti = bursts.copy()
        else:
            self.ti = []
        
        self.inc_solve([time])

        # Solve kinetics ODE system
        H = anp.zeros((3, self.t.shape[0]))
        H[:, 0] = ic
        for t in range(1, self.t.shape[0]):
            H[:, t] = self._solve_kinetics(H[:, t - 1], self.S[t - 1])
        
        return H
    
    # Incomplete function for solving model
    def inc_solve(self, time: anp.ndarray):
        # Initialize stimulation pattern
        self._set_stimulation(time[-1])

        # Initialize kinetics
        self._set_kinetics()
    
    # Incomplete function for solving model
    # Used when running fitting engine
    def _inc_solve_fit(self, time: anp.ndarray, theta: anp.ndarray):
        # Initialize stimulation pattern
        self._set_stimulation(time[-1])

        # Initialize fitting parameters
        self._set_fit(theta)
        
        # Initialize kinetics
        self._set_kinetics()
    
    # Set parameters to current estimated value
    # Called from fitting engine for each iteration
    def _set_fit(self, theta: anp.ndarray):
        assert(len(self.fit_params) == theta.shape[0])
        for i in range(len(self.fit_params)):
            self.params[self.fit_params[i]] = theta[i]
    
    # Time vector + stimulation pattern
    def _set_stimulation(self, end):
        self.nt = int(end / self.dt) + 1
        self.t = anp.linspace(0, end, self.nt)
        self.S = anp.zeros(self.t.shape)
        for start in self.ti:
            self.S += anp.heaviside((self.t - start).round(2), 1) * anp.heaviside((start + (self.NP / self.f) - self.t).round(2), 1)
    
    # Combine individual kick components/time constants into individual vectors
    # Used for solving model
    def _set_kinetics(self):
        k_vector = []
        tau_vector = []
        for i in range(len(self.params['ktypes'])):
            k_vector.append(self.params[f'k{i + 1}'])
            tau_vector.append(self.params[f'tau{i + 1}'])
        self.params['k'] = anp.array(k_vector)
        self.params['tau'] = anp.array(tau_vector)
    
    # DA kinetics ODE
    # dHj/dt
    def _solve_kinetics(self, H: anp.ndarray, S: float):
        kick = self.f * self.params['k'] * H * S
        decay = (1 - S) * (1 - H) / self.params['tau']
        H_dt = H + self.dt * (kick + decay)
        return H_dt
    
    # DA electrode coupled ODE system
    # d[DA]_E/dt
    # dΓ_[DA]/dt
    def _solve_electrode(self, DAs: float, DAe: float, GammaDA: float):
        DAe_dt = DAe + (self.params['kS'] * DAs - self.params['kE'] * DAe + self.kGamma * GammaDA) * self.dt
        GammaDA_dt = GammaDA + (self.params['kads1'] * DAe - self.params['kads2'] * DAe * GammaDA - self.params['kads3'] * GammaDA) * self.dt
        return anp.array([DAe_dt]), anp.array([GammaDA_dt])
