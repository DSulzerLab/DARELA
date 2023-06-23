# DARELA: DA RELease Analysis

DARELA is a Python library that provides computational models written using differential equations to analyze and quantify the kinetics of dopamine (DA) release, reuptake, and diffusion. The models can be used to fit data from in vivo fast-scan cyclic voltammetry (FSCV) experiments.

For more details on how these computational models are derived, please refer to our [paper](https://doi.org/10.1101/2022.05.04.490695).

## Installation

Start by cloning this repo, then run:

```
pip install -e .
```

This will install a development copy of the library, which can be modified as needed.

The library requires `autograd` and `alive-progress`, which are automatically installed as dependencies if your Python setup does not have it.

## Usage

There are three computational models provided with this library: the Simple Uniform Release (SUR) model, Spatiotemporal Uniform Release (STUR) model, and Spatiotemporal Discrete Release (STDR) model.

### Step 1. Model Initialization

To initialize the SUR model, run:

```
from darela import SUR

model = SUR()
model.initialize(bursts, f, NP, I)
model.loss(L)
```

To initialize the STUR model, run:

```
from darela import STUR

model = STUR()
model.initialize(bursts, f, NP, I)
model.geometry(Rl, Rd)
```

To initialize the STDR model, run:

```
from darela import STDR

model = STDR()
model.initialize(bursts, f, NP, I)
model.geometry(Rl, Rd)
model.release_sites(locs)
```

- `bursts`: a list of start times for the bursts in an FSCV experiment
- `f`: stimulus frequency (in Hz)
- `NP`: number of pulses per burst
- `I`: stimulus current (in mA)
- `L`: loss factor of DA
- `Rl`: radius of the cylinder (in μm)
- `Rd`: radius of the dead space (in μm)
- `locs`: a list of discrete release site locations in the striatum

### Step 2: Parameter Initialization

To initialize release and kinetic parameters, load each set of parameters in a dictionary and call the corresponding model functions:

```
release = {
    ...
}

kinetics = {
    ...
}

model.release_params(release)
model.kinetic_params(kinetics)
```

For more information on the list of available parameters, please check the Model Parameters section of this README.

### Step 3: Model Solving

To simulate the model output with the specified experiment, release, and kinetic parameters, run:

```
model.solve(t)
```

- `t`: 1D `np.ndarray` time-series.

### Step 3A: Parameter Estimation

This library comes with a built-in parameter estimation algorithm to find the closest fit parameters to an FSCV trace. To fit the model to the data, run:

```
model.fit(y, t, params)
```
- `y`: 1D `np.ndarray` containing FSCV data points
- `t`: 1D `np.ndarray` containing corresponding time points
- `params`: a list of parameters (release or kinetic) to fit to the data.

## Examples

For detailed examples on how to use this library, please check the `examples` directory.

## Model Parameters

### Release parameters:
- `Vm`: maximum velocity of DAT uptake (μM/s)
- `Km`: affinity (binding) constant of DAT uptake (μM)
- `DAp`: amount of DA release per pulse (μM/mA)
- `kS`: rate transfer of DA from striatum to electrode
- `kE`: rate transfer of DA from electrode back to striatum
- `kads1`: adsorption kinetic of DA
- `kads2`: desorption kinetic of DA
- `kads3`: desorption kinetic 2 of DA

### Kinetic parameters:
- `ktypes`: a list of kinetic types with four possible values:
    - `stf`: short-term facilitation
    - `std`: short-term depression
    - `ltf`: long-term facilitation
    - `ltd`: long-term depression
- `p`: a list of plasticity factors for each kinetic
    - Facilitation: `p` > 0
    - Depression: `p` < 0
- `tau`: a list of time constants for each kinetic
    - Short-Term: 5 < `tau` < 50
    - Long-Term: 600 < `tau` < 1200

## Citing DARELA

If you find this library useful in your research, please cite our paper:

```
N Shashaank, M Somayaji, M Miotto, EV Mosharov, EA Makowicz, DA Knowles, G Ruocco, DL Sulzer. 2023. Computational models of dopamine release measured by fast scan cyclic voltammetry in vivo. PNAS Nexus 2(3). https://doi.org/10.1093/pnasnexus/pgad044 
```
