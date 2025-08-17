from timeit import timeit
import matplotlib.pyplot as plt
from check_pso import kinetics_python
# Define the parameters
params = {
    't_tot': 3600,
    'dt': 0.1,
    'accuracy': 1e-6,
    'Tstart': 2000.0,
    'R_G': 100e-6,
    'dQ': 2.355,
    'D0': 3.3e-5,
    'x_bulk': 0.0004942285667828602,
    'GB_thickness': 8.4e-10,
    'E': -1.2972039599999334,
    'A': 0.27920385353982713,
    'gb_conc': 0.47854028493419043,
    'Tend': 500.0,
    'heat_treatment_type': 'linear',
    'r': -1.0
}



results = kinetics_python(**params)

plt.scatter(results[0] )
plt.savefig("check_pso.pdf")
