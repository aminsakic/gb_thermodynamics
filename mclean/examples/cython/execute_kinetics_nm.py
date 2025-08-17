from timeit import timeit
import matplotlib.pyplot as plt

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


t1 = timeit("kinetics_python(**params)",
            setup="from kinetics_python import kinetics_python; from __main__ import params",
            number=1)

t2 = timeit("kinetics_annotated(**params)",
            setup="from kinetics_annotated import kinetics_annotated; from __main__ import params",
            number=1)

t3 = timeit("kinetics_cythonized(**params)",
            setup="from kinetics_cythonized import kinetics_cythonized; from __main__ import params",
            number=1)

t4 = timeit("kinetics_cythonized_nm(**params)",
            setup="from kinetics_cythonized_nm import kinetics_cythonized_nm; from __main__ import params",
            number=1)
print(f"Python: Time for get kinetics: {t1} seconds for 1 run")
print(f"Annotated: Time for get kinetics: {t2} seconds for 1 run")
print(f"Cythonized: Time for get kinetics: {t3} seconds for 1 run")
print(f"Cythonized and self written Nelder-Mead: Time for get kinetics: {t4} seconds for 1 run")
print(f"Cythonized code with self written Nelder-Mead is {t1/t4} times faster than the originial python implementation using the scipy Nelder-Mead minimizer.")
