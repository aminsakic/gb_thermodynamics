from timeit import timeit
import matplotlib.pyplot as plt
from kinetics_cythonized_nm import kinetics_cythonized_nm
from kinetics_python import kinetics_python
from kinetics_cythonized_pso import kinetics_cythonized_pso

import numpy as np

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

params_pso = {
    't_tot': 3600,
    'dt': 0.1,
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

            
results_python = kinetics_python(**params)
gb_cython_nm, ife_cython_nm = kinetics_cythonized_nm(**params)
gb_cython_pso, ife_cython_pso = kinetics_cythonized_pso(**params_pso)


plt.plot(results_python[0], color="red",label="python")
plt.plot(np.array(gb_cython_nm), color="green",ls="--",label="cython")
plt.plot(np.array(gb_cython_pso), color="blue",ls="-.",label="cython")

plt.legend()
plt.tight_layout()
plt.savefig("comparison_python_cython_pso.pdf")
