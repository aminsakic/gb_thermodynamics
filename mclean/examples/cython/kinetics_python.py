
import numpy as np
from scipy.optimize import minimize
from heat_treatment import get_T_and_D

# Calculate fraction of GB sites compared to all sites in the system
def gb_site_fraction(R_G=100e-6, gb_width=8.4e-10):
    V_G = R_G**3
    V_GB = V_G - (R_G - gb_width)**3
    return V_GB/V_G

def kinetics_python(t_tot: int, dt: float, accuracy: float, Tstart: float, R_G: float, D0: float, dQ: float,
                  x_bulk: float, GB_thickness: float, E: float, A: float, gb_conc: float, Tend=None, r=None, heat_treatment_type="linear"):
    
    
    R_g = (8.314 / (1.602 * 10 ** (-19))) / (6.022 * 10 ** (23))  # gas constant in eV/K --> Boltzmann constant

    temperatures, diffusion_coefficients = get_T_and_D(t_tot=t_tot, Tstart=Tstart, D0=D0, dQ=dQ, dt=dt, Tend=Tend, heat_treatment_type=heat_treatment_type, r=r)
    Rg_Ts = temperatures * R_g

    gb_concs = np.empty(shape=(len(temperatures), 1), dtype=np.float32)
    ife = np.empty(shape=(len(temperatures), 1), dtype=np.float32)
    
    # Model specific parameters
    m = 1
    N_tot = 1
    f = gb_site_fraction(R_G=R_G, gb_width=GB_thickness) 
    fi = f * m/N_tot
    fg = fi / f 
    x_hat_ki = gb_conc 
    x_global = (1 - f) * x_bulk + fi * x_hat_ki
    x_hat_sum = fi * x_hat_ki
    x_bulk_start = x_bulk 


    # Starting value for optimization
    lamda = 0.001

    def obj_func(lamda, Rg_T, x_hat_sum):
        
        nom = np.exp( -(E - lamda) / Rg_T)
        nom = np.clip(nom, a_min=None, a_max=1e300)  

        denom = 1 + np.exp( -(E - lamda) / Rg_T)

        xk_i_hat_temp = nom/denom 
        xk_hat_temp = fi * xk_i_hat_temp 

        
        abs_difference = np.abs(xk_hat_temp - x_hat_sum) 
        return abs_difference
    
    
    options = {'disp': False, 'xatol': accuracy}


    for indx in range(len(temperatures)):

        Rg_T = Rg_Ts[indx]
        
        D = diffusion_coefficients[indx]
        
        # Perform minimization
        result_value = 1
        rv_i = 0
        while abs(result_value) > accuracy: 
            # Optimization using Nelder-Mead algorithm
            result = minimize(obj_func, lamda, args=(Rg_T, x_hat_sum), method="Nelder-Mead", options=options)
            new_lamda = result.x
            lamda = new_lamda
            result_value = result.fun
            rv_i += 1
            if rv_i > 10: # for checking, 10 is guessed
                print(result_value)
                break
        
        
        nom = np.exp( -(E - lamda) / Rg_T)
        nom = np.clip(nom, a_min=None, a_max=1e300)  

        denom = 1 + np.exp( -(E - lamda) / Rg_T)

        x_hat_ki = nom/denom
        
        gb_concs[indx] = x_hat_ki * fg
        

        nom_ln = 1 - x_bulk
        denom_ln = x_bulk
        frac_ln = np.log(nom_ln/denom_ln)

        lamda_frac = lamda/Rg_T

        lhs = 15*x_bulk*D/(R_G**2)

        dx_hat = (-frac_ln - lamda_frac) * lhs 

        if np.isnan(dx_hat).any():
                    raise ValueError("NaN value encountered in rhs. Stopping calculation.")

        x_hat_sum = x_hat_sum + dx_hat * dt
    
        
        x_bulk = (x_global - x_hat_sum) / (1-f)

        ife[indx] = (x_hat_sum * fg - x_bulk_start)/A
        
    return gb_concs, ife
