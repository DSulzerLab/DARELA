import autograd.numpy as anp
from .base import BaseModel

# PDE Model constructor class
# Used as constructor for STUR and STDR
class PDEModel(BaseModel):
    D = 240. # Diffusion coeffient
    P: anp.ndarray # Discrete release sites

    def __init__(self, kinetics: bool = True, discrete: bool = False):
        super(PDEModel, self).__init__(kinetics)
        self.discrete = discrete
    
    # FSCV experiment parameters
    def initialize(self, bursts: list, f: float, NP: float, I: float):
        super().initialize(bursts, [f, NP, I])
    
    # Spatial geometry parameters
    def geometry(self, Rl: float, Rd: float):
        self.Rl = Rl # Radius of cylinder
        self.Rd = Rd # Radius of dead space

        # Spatial discretization
        self.dR = 1 # in μm
        self.nR = int(self.Rl / self.dR) + 1
        self.R = anp.linspace(0, self.Rl, self.nR)
        self.eta = anp.heaviside(((self.Rl - self.Rd) - self.R).round(2), 1.0) # Dead space discretization with Heaviside theta
        
        # Temporal discretization
        self.dt = (self.dR ** 2) / (2 * self.D)
    
    # Parameter estimation
    def fit(self, data: anp.ndarray, time: anp.ndarray, params: list):
        super().fit(data, time, params, self.discrete)

    # Integrate PDE model over requested time series
    def solve(self, time, kinetics_state = None):
        super().inc_solve(time)
        
        # Initialize arrays
        DA = anp.zeros((self.nR, self.nt))
        U = anp.zeros(self.nR)
        W = anp.zeros(self.nR)
        DAe = anp.zeros(self.nt)
        GammaDA = anp.zeros(self.nt)
        if self.kinetics: 
            H = anp.zeros((len(self.params['ktypes']), self.nt))
            if kinetics_state is not None: # Input state for resuming kinetics
                assert(len(kinetics_state) == len(self.params['ktypes']))
                H[:, 0] = kinetics_state
            else: # Default state
                H[:, 0] = 1.
        
        # Integrate PDEs + ODEs
        for t in range(1, self.nt):
            # Cylindrical diffusion
            # d[DA]/dt
            U[1:-1] = (self.R[1:-1] / (2 * self.R[1:-1] - 1)) * DA[2:, t - 1] + ((self.R[1:-1] - 1) / (2 * self.R[1:-1] - 1)) * DA[:-2, t - 1] - DA[1:-1, t - 1]
            DA[1:-1, t] = DA[1:-1, t - 1] + self.D * U[1:-1] * self.dt

            # DA kinetics
            # dHj/dt
            if self.kinetics:
                H[:, t] = self._solve_kinetics(H[:, t - 1], self.S[t - 1])
                A = anp.prod(H[:, t])

            # DA release and DAT uptake
            # d[DA]/dt
            release = self.params['DAp'] * self.I * self.f * self.S[t - 1]
            if self.kinetics: release *= A
            if self.discrete: release = release * self.P[1:-1]
            DAT = (self.params['Vm'] * DA[1:-1, t - 1]) / (DA[1:-1, t - 1] + self.params['Km'])
            W[1:-1] = self.eta[1:-1] * (release - DAT) * self.dt 
            DA[1:-1, t] += W[1:-1]

            # Boundary conditions and negative values
            DA[0, t] = DA[1, t]
            DA[-1, t] = DA[-2, t]
            DA[DA[:, t] < 0, t] = 0
        
            # DA at electrode + adsorption
            # d[DA]_E/dt
            # dΓ_[DA]/dt
            DAe[t], GammaDA[t] = self._solve_electrode(DA[-1, t - 1], DAe[t - 1], GammaDA[t - 1])

        # Interpolate for requested time points
        DAe = anp.interp(time, self.t, DAe)
        return DAe
    
    # Integrate PDE model over requested time series
    # Used when running fitting engine
    # Removes in-place vector operations for gradient computation
    def _solve_fit(self, time: anp.ndarray, theta: anp.ndarray):
        super()._inc_solve_fit(time, theta)

        # Initialize arrays
        DA = anp.zeros((self.nR, 1))
        DAe = anp.array([0.])
        GammaDA = anp.array([0.])
        if self.kinetics:
            H = anp.zeros((3, 1))
            H[:, 0] = 1.
        
        # Integrate PDEs + ODEs
        for t in range(1, self.nt):
            # Cylindrical diffusion
            # d[DA]/dt
            U = (self.R[1:-1] / (2 * self.R[1:-1] - 1)) * DA[2:, t - 1] + ((self.R[1:-1] - 1) / (2 * self.R[1:-1] - 1)) * DA[:-2, t - 1] - DA[1:-1, t - 1]
            DA_dt = DA[1:-1, t - 1] + self.D * U * self.dt

            # DA kinetics
            # dHj/dt
            if self.kinetics:
                H_dt = self._solve_kinetics(H[:, t - 1], self.S[t - 1])
                H_dt = H_dt[:, anp.newaxis]
                H = anp.hstack((H, H_dt))
                A = anp.prod(H[:, t])

            # DA release and DAT uptake
            # d[DA]/dt
            release = self.params['DAp'] * self.I * self.f * self.S[t - 1]
            if self.kinetics: release *= A
            if self.discrete: release = release * self.P[1:-1]
            DAT = (self.params['Vm'] * DA[1:-1, t - 1]) / (DA[1:-1, t - 1] + self.params['Km'])
            W = self.eta[1:-1] * (release - DAT) * self.dt 
            DA_dt += W

            # Boundary conditions and negative values
            DA_dt = anp.concatenate((anp.array([DA_dt[0]]), DA_dt, anp.array([DA_dt[-1]])))
            DA_dt = anp.where(DA_dt > 0, DA_dt, 0)
            DA_dt = DA_dt[:, anp.newaxis]
            DA = anp.hstack((DA, DA_dt))
        
            # DA at electrode + adsorption
            # d[DA]_E/dt
            # dΓ_[DA]/dt
            DAe_dt, GammaDA_dt = self._solve_electrode(DA[-1, t - 1], DAe[t - 1], GammaDA[t - 1])
            DAe = anp.append(DAe, DAe_dt)
            GammaDA = anp.append(GammaDA, GammaDA_dt)

        # Interpolate for requested time points
        index = anp.clip(anp.searchsorted(self.t, time), 1, len(self.t) - 1)
        df = DAe[index] - DAe[index - 1]
        delta = time - self.t[index - 1]
        DAe = anp.where((df == 0), DAe[index], DAe[index - 1] + (delta / self.dt) * df)
        return DAe
