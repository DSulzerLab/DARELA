import autograd.numpy as anp
from typing import Iterable

from ._impl import *

# Simple Uniform Release Model
class SUR(ODEModel): 
    def __init__(self, kinetics: bool = True):
        super(SUR, self).__init__(kinetics)

# Spatiotemporal Uniform Release Model
class STUR(PDEModel):
    def __init__(self, kinetics: bool = True):
        super(STUR, self).__init__(kinetics, discrete = False)

# Spatiotemporal Discrete Release Model
class STDR(PDEModel): 
    def __init__(self, kinetics: bool = True):
        super(STDR, self).__init__(kinetics, discrete = True)
    
    # Discrete release sites
    def release_sites(self, Ri: Iterable[float]):    
        self.P = anp.zeros(self.nR)
        for i in Ri:
            index = anp.argmin(anp.abs(i - self.R))
            self.P[index] += 1
