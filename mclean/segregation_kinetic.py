import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict
from scipy.optimize import minimize


def diff_coef(D0: float, dQ, T, kB: float = (8.314/(1.602*10**(-19)))/(6.022*10**(23))):
    # D0 ... diffusion prefactor (m2/s)
    # dQ ... activation energy (eV)
    # T .... temperature (K)
    return D0*np.exp(-dQ/(kB*T))


def gb_site_fraction(R_G=100e-6, gb_width=8.4e-10):
    # calculates the GB site fractions in the system
    # grains are assumed to have a square like shape
    # R_G ... grain size (m)
    # gb_width ... width of the GB (m)
    V_G = R_G**3
    V_GB = V_G - (R_G - gb_width)**3
    return V_GB/V_G

# Heat treatment example


def linear_heat_treatment(T0: float, t_tot: int, Tend: float, t: float):
    k = (Tend-T0)/t_tot
    T = k*t + T0
    return T


def kinetic_model(run_time: int, dt: float, accuracy: float, T0: float, Temb: float, R_G: float, diffusion: List[Tuple[float, float]],
                  x_bulk: np.ndarray, GB_thickness: float, E: np.ndarray, A: float, gb_conc: np.ndarray, mi: np.ndarray, lamda: np.ndarray):
    """Links segregation energies calculated from atomistic simulations with non-equilibrium enrichment (kinetics) of solutes at GBs. 
       The code solves multi-component and multi-site systems. The formalism is taken from:
       D.Scheiber, T. Jechtl, J. Svoboda, F.D. Fischer, L. Romaner, On solute depletion zones along grain boundaries during segregation, 
       Acta Materialia (2019), doi: https://doi.org/10.1016/j.actamat.2019.10.040

    Parameters
    ----------
    run_time : int
        Simulation time.
    dt : float
        Time steps in the calculation, e.g. 0.01.
    accuracy : float
        Determines the accuracy for the lamda optimization.
    T0 : float
        Starting temperature.  
    Temb : float
        End temperature.
    R_G : float
        Grain size diameter (m).
    diffusion : List[Tuple[float, float]]
        Information on the activation energy Q_A (eV) and the diffusion prefactor D0 (m2/s) of each component.
    x_bulk : np.ndarray
        Bulk conentrations (-) of solutes.
    GB_thickness : float
        Defines the width (m) of the investigated GB.
    E : np.ndarray
        Segregation energies (eV) of each component to each GB site.
    A : float
        GB area (nm2).
    gb_conc : np.ndarray
        Starting values for the GB concentrations (-) of each solute on each GB site.
    mi : np.ndarray
        Multiplicities of each GB site.
    lamda : np.ndarray
        Initial lamda values for the optimization.

    Returns
    ------- 
    np.ndarray, List, List
        GB concentrations at each time step, times and temperatur profile.   
    """

    # Constants
    # gas constant in eV/K --> Boltzmann constant
    R_g = (8.314/(1.602 * 10 ** (-19)))/(6.022 * 10 ** (23))
    n = len(mi)  # Number of different sites
    n_tot = np.sum(mi)  # total number of sites
    N = len(x_bulk)  # Number of components

    # Results
    Ts = list()
    gb_concs = [np.array([]) for _ in range(N)]
    # ife = list() # interfacial excess (N(cgb-cbulk)/A)
    times = list()

    # fraction of gb sites to whole system
    f = gb_site_fraction(R_G=R_G, gb_width=GB_thickness)
    # fraction of individual sites to whole system
    fi = (f/n_tot)*mi
    # fraction of individual sites compared to all GB sites
    fg = fi / f

    # First we calculate the global concentration of x:
    x_hat_ki = gb_conc  # Optimally, provided by McLean ---> but without any site multiplicities
    x_hat_sum = np.sum(fi * x_hat_ki, axis=1)
    x_global = (1 - f) * x_bulk + np.sum(fi * x_hat_ki, axis=1)
    x_bulk_ki = x_bulk  # only needed for ife

    # initial values for lamda parameters
    lamda = lamda

    # lamda optimization
    def obj_func(lamda, Rg_T, x_hat_sum):
        # storing xk_hat_temp_results ---> individual components and summed up over all sites
        xk_hat_temp_results = np.array([])
        for comp in range(N):

            # storing xki_hat_temp results ---> individual sites
            xki_hat_temp_results = np.array([])
            for site in range(n):
                nom = np.exp(-(E[comp, site] - lamda[comp]) / Rg_T)
                # Clip the exponential values
                nom = np.clip(nom, a_min=None, a_max=1e300)

                sum_exp = 0
                for comp_sum in range(N):
                    denom_exp = np.exp(-(E[comp_sum, site] -
                                       lamda[comp_sum]) / Rg_T)
                    sum_exp += denom_exp

                denom = 1 + sum_exp
                xki_hat_temp = nom / denom
                xki_hat_temp_results = np.append(
                    xki_hat_temp_results, xki_hat_temp)

            # calculating the total occupation of one species
            xk_hat_temp = np.dot(fi, xki_hat_temp_results)
            xk_hat_temp_results = np.append(xk_hat_temp_results, xk_hat_temp)

        # difference between last and actual step
        difference_vector = xk_hat_temp_results - x_hat_sum
        # absolute magnitude = error
        abs_magnitude = np.sqrt(np.sum(difference_vector**2))
        return abs_magnitude

        # Set the optimization options
    options = {'disp': False, 'xatol': accuracy}

    total_iterations = run_time/dt
    t = 0
    with tqdm(total=total_iterations) as pbar:
        while t < int(run_time):

            # Temperature
            T = linear_heat_treatment(T0=T0, t_tot=run_time, Tend=Temb, t=t)
            Ts.append(T)
            Rg_T = R_g*T

            # Diffusion
            # Temperature-dependent diffusion coefficients
            D = np.array([diff_coef(D0=do, dQ=q, T=T) for do, q in diffusion])

            result_value = 1
            rv_i = 0
            while abs(result_value) > accuracy:
                # Optimization using Nelder-Mead algorithm
                result = minimize(obj_func, lamda, args=(
                    Rg_T, x_hat_sum), method="Nelder-Mead", options=options)
                new_lamda = result.x
                lamda = new_lamda
                result_value = result.fun
                rv_i += 1
                if rv_i > 20:  # for checking, 20 is guessed
                    break

            # calculate all site fraction with new lamda
            x_hat_k_new_results = np.array([])
            for comp in range(N):

                # storing xki_hat_temp results ---> individual sites
                x_hat_ki_new_results = np.array([])
                for site in range(n):
                    nom = np.exp(-(E[comp, site] - lamda[comp]) / Rg_T)
                    # Clip the exponential values
                    nom = np.clip(nom, a_min=None, a_max=1e300)

                    sum_exp = 0
                    for comp_sum in range(N):
                        denom_exp = np.exp(-(E[comp_sum,
                                           site] - lamda[comp_sum]) / Rg_T)
                        sum_exp += denom_exp

                    denom = 1 + sum_exp
                    xki_hat_temp = nom / denom
                    x_hat_ki_new_results = np.append(
                        x_hat_ki_new_results, xki_hat_temp)

                # calculating the total occupation of one species
                x_hat_k_new_results = np.append(
                    x_hat_k_new_results, np.dot(x_hat_ki_new_results, fg))

            # update gb_concs for each component
            for comp in range(N):
                gb_concs[comp] = np.append(
                    gb_concs[comp], x_hat_k_new_results[comp])

            # update the concentrations
            dx_hat_sum_results = np.array([])
            # equal for all components -> how many sites in the bulk are occupied by matrix elements
            nom_ln = 1 - np.sum(x_bulk)
            for comp in range(N):
                denom_ln = x_bulk[comp]
                frac_ln = np.log(nom_ln/denom_ln)

                lamda_frac = lamda[comp]/Rg_T

                lhs = 15*x_bulk[comp]*D[comp]/(R_G**2)

                dx_hat_comp = (-frac_ln - lamda_frac) * lhs
                dx_hat_sum_results = np.append(dx_hat_sum_results, dx_hat_comp)

            # if some problems in ln() ---> dx = nan. dt to high
            # if so print warning and stop calculation
            if np.isnan(dx_hat_sum_results).any():
                raise ValueError(
                    "NaN value encountered in rhs. Stopping calculation.")

            x_hat_sum = x_hat_sum + dx_hat_sum_results * dt

            # x_bulk needs to be updated as well
            # keep in mind that x_bulk will always be almost the same as x_global
            # because the bulk area is simply much bigger than the GB area
            x_bulk = (x_global - x_hat_sum)/(1-f)

            # interfacial excess
            # ife.append((x_hat_ki - x_bulk_ki)/A)

            times.append(t)

            # update the progress bar
            pbar.update(1)

            # update the time
            t = t + dt

        return gb_concs, times, Ts
