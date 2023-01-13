from darela import SUR
import autograd.numpy as anp
import matplotlib.pyplot as plt

release = {
    "Vm": 4.8,
    "Km": 0.2,
    "DAp": 0.43,
    "kS": 0.9, 
    "kE": 1.05,
    "kads1": 0.035,
    "kads2": 0.14,
    "kads3": 0.09
}

kinetic = {
    "ktypes": ["stf", "std", "ltd"],
    "p": [0.0105, -0.003, -0.0011],
    "tau": [7.5, 12.5, 900],
}

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

    # Solve model and plot solution
    t = anp.linspace(0, 5, 51)
    _, y = model.solve(t)
    plt.plot(t, y)
    plt.show()

main()
