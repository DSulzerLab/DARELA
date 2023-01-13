import autograd.numpy as anp
from .base import BaseModel

# ODE Model constructor class
# Used as constructor for SUR
class ODEModel(BaseModel):
    def __init__(self, kinetics: bool = True):
        super(ODEModel, self).__init__(kinetics)
    
    # FSCV experiment parameters
    def initialize(self, bursts: list, f: float, NP: float, I: float):
        super().initialize(bursts, [f, NP, I])
        self.dt = 1 / self.f
    
    # Loss factor of DA
    def loss(self, L: float):
        self.L = L
    
    # Parameter estimation
    def fit(self, data: anp.ndarray, time: anp.ndarray, params: list):
        super().fit(data, time, params, False)
     
    # Integrate ODE model over requested time series
    def solve(self, time: anp.ndarray):
        super().inc_solve(time)

        # Initialize arrays
        DAs = anp.zeros(self.t.shape)
        DAe = anp.zeros(self.t.shape)
        GammaDA = anp.zeros(self.t.shape)
        if self.kinetics:
            H = anp.zeros((3, self.t.shape[0]))
            H[:, 0] = 1.

        # Integrate ODEs
        for t in range(1, self.t.shape[0]):
            # DA kinetics 
            # dHj/dt
            if self.kinetics:
                H[:, t] = self._solve_kinetics(H[:, t - 1], self.S[t - 1])
                A = anp.prod(H[:, t])

            # DA release and DAT uptake 
            # d[DA]_S/dt
            release = self.L * self.params['DAp'] * self.I * self.f * self.S[t - 1]
            if self.kinetics: release *= A
            DAT = (self.params['Vm'] * DAs[t - 1]) / (DAs[t - 1] + self.params['Km'])
            DAs[t] = DAs[t - 1] + (release - DAT) * self.dt

            # DA at electrode + adsorption
            # d[DA]_E/dt
            # dΓ_[DA]/dt
            DAe[t], GammaDA[t] = self._solve_electrode(DAs[t - 1], DAe[t - 1], GammaDA[t - 1])
        
        # Interpolate for requested time points
        DAs = anp.interp(time, self.t, DAs)
        DAe = anp.interp(time, self.t, DAe)
        return DAs, DAe
    
    # Integrate ODE model over requested time series
    # Used when running fitting engine
    # Removes in-place vector operations for gradient computation
    def _solve_fit(self, time: anp.ndarray, theta: anp.ndarray):
        super()._inc_solve_fit(time, theta)

        # Initialize arrays
        DAs = anp.array([0.])
        DAe = anp.array([0.])
        GammaDA = anp.array([0.])
        if self.kinetics:
            H = anp.zeros((3, 1))
            H[:, 0] = 1.

        # Integrate ODEs
        for t in range(1, self.t.shape[0]):
            # DA kinetics 
            # dHj/dt
            if self.kinetics:
                H_dt = self._solve_kinetics(H[:, t - 1], self.S[t - 1])
                H_dt = H_dt[:, anp.newaxis]
                H = anp.hstack((H, H_dt))
                A = anp.prod(H[:, t])

            # DA release and DAT uptake 
            # d[DA]_S/dt
            release = self.L * self.params['DAp'] * self.I * self.f * self.S[t - 1]
            if self.kinetics: release *= A
            DAT = (self.params['Vm'] * DAs[t - 1]) / (DAs[t - 1] + self.params['Km'])
            DAs_dt = DAs[t - 1] + (release - DAT) * self.dt
            DAs = anp.append(DAs, DAs_dt)

            # DA at electrode + adsorption
            # d[DA]_E/dt
            # dΓ_[DA]/dt
            DAe_dt, GammaDA_dt = self._solve_electrode(DAs[t - 1], DAe[t - 1], GammaDA[t - 1])
            DAe = anp.append(DAe, DAe_dt)
            GammaDA = anp.append(GammaDA, GammaDA_dt)
        
        # Interpolate for requested time points
        index = anp.clip(anp.searchsorted(self.t, time), 1, len(self.t) - 1)
        df = DAe[index] - DAe[index - 1]
        delta = time - self.t[index - 1]
        DAe = anp.where((df == 0), DAe[index], DAe[index - 1] + (delta / self.dt) * df)
        return DAe
