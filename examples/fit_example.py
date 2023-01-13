from darela import SUR
import autograd.numpy as anp
import matplotlib.pyplot as plt

# For fitting release parameters, initial values
# do not need to be specified (in this case, Vm and DAp)
release = {
    "Km": 0.2,
    "kS": 0.9, 
    "kE": 1.05,
    "kads1": 0.035,
    "kads2": 0.14,
    "kads3": 0.09
}

# For fitting kinetics, None values need to be inserted
# if initial values are not specified
kinetic = {
    "ktypes": ["stf", "std", "ltd"],
    "p": [None, -0.003, -0.0011],
    "tau": [None, 12.5, 900],
}

# Load FSCV trace data and corresponding time series data for fitting
def load_data():
    # Add your code here
    pass

def main():
    # Initialization parameters
    bursts = [0.34]
    f = 50
    NP = 30
    I = 0.4
    L = 0.9

    # Initialize model
    model = SUR()
    model.initialize(bursts, f, NP, I)
    model.loss(L)

    # Load release and kinetic parameters
    model.release_params(release)
    model.kinetic_params(kinetic)

    # Load FSCV trace data and corresponding time series data for fitting
    t_data, y_data = load_data()

    # Fit some of the parameters
    model.fit(y_data, t_data, ['Vm', 'DAp', 'p1', 'tau1'])

    # Solve model and plot solution
    t = anp.linspace(0, 5, 51)
    _, y = model.solve(t)
    plt.plot(t, y)
    plt.plot(t_data, y_data)
    plt.show()

main()
