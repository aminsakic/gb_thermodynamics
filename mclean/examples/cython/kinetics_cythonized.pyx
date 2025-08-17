
import numpy as np
cimport numpy as np
from scipy.optimize import minimize
from libc.math cimport exp, sqrt

# Function declarations
cdef np.ndarray[np.float32_t, ndim=1] isothermal(float Tstart, int t_tot, float dt):
    cdef int n = int(t_tot / dt) + 1
    cdef np.ndarray[np.float32_t, ndim=1] t_steps = np.zeros(n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] temperatures = np.zeros(n, dtype=np.float32)
    cdef int i
    for i in range(n):
        t_steps[i] = i * dt
        temperatures[i] = Tstart
    return temperatures


cdef np.ndarray[np.float32_t, ndim=1] linear_heat_treatment(float Tstart, float Tend, int t_tot, float dt):
    cdef int n = int(t_tot / dt) + 1
    cdef np.ndarray[np.float32_t, ndim=1] t_steps = np.zeros(n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] temperatures = np.zeros(n, dtype=np.float32)
    cdef float k = (Tend - Tstart) / t_tot
    cdef int i
    for i in range(n):
        t_steps[i] = i * dt
        temperatures[i] = k * t_steps[i] + Tstart
    return temperatures

cdef np.ndarray[np.float32_t, ndim=1] newton_cooling(float Tstart, float Tend, int t_tot, float dt, float r):
    cdef int n = int(t_tot / dt) + 1
    cdef np.ndarray[np.float32_t, ndim=1] t_steps = np.zeros(n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] temperatures = np.zeros(n, dtype=np.float32)
    cdef int i
    for i in range(n):
        t_steps[i] = i * dt
        temperatures[i] = Tend + (Tstart - Tend) * exp(-r * sqrt(t_steps[i]))
    return temperatures

cdef np.ndarray[np.float32_t, ndim=1] diff_coef(float D0, float dQ, np.ndarray[np.float32_t, ndim=1] T, float kB):
    cdef int n = T.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1] diffusion_coefficients = np.zeros(n, dtype=np.float32)
    cdef int i
    for i in range(n):
        diffusion_coefficients[i] = D0 * exp(-dQ / (kB * T[i]))
    return diffusion_coefficients

cpdef tuple get_T_and_D(int t_tot, float Tstart, float D0, float dQ, float dt, float Tend=-1, str heat_treatment_type="iso", float r=-1):
    cdef np.ndarray[np.float32_t, ndim=1] temperatures, diffusion_coefficients
    if heat_treatment_type == "iso":
        temperatures = isothermal(Tstart=Tstart, t_tot=t_tot, dt=dt)
    elif heat_treatment_type == "linear":
        if Tend == -1:
            raise ValueError("Tend must be provided for linear heat treatment.")
        temperatures = linear_heat_treatment(Tstart=Tstart, Tend=Tend, t_tot=t_tot, dt=dt)
    elif heat_treatment_type == "newton_cooling":
        if Tend == -1 or r == -1:
            raise ValueError("Tend and r must be provided for Newton cooling heat treatment.")
        temperatures = newton_cooling(Tstart=Tstart, Tend=Tend, t_tot=t_tot, dt=dt, r=r)
    else:
        raise ValueError("Invalid heat_treatment_type. Must be 'iso', 'linear', or 'newton_cooling'.")

    diffusion_coefficients = diff_coef(D0=D0, dQ=dQ, T=temperatures, kB=(8.314 / (1.602 * 10**(-19))) / (6.022 * 10**(23)))

    # Convert memoryviewslice to numpy array
    #diffusion_coefficients_np = np.array(diffusion_coefficients)

    return temperatures, diffusion_coefficients


# Calculate fraction of GB sites compared to all sites in the system
cdef double gb_site_fraction(double R_G=100e-6, double gb_width=8.4e-10):
    cdef double V_G = R_G**3
    cdef double V_GB = V_G - (R_G - gb_width)**3
    return V_GB / V_G

cdef double obj_func(double lamda, double Rg_T, double x_hat_sum, double fi, double E):
    cdef double nom = np.exp(-(E - lamda) / Rg_T)
    nom = np.clip(nom, a_min=None, a_max=1e300)
    cdef double denom = 1 + np.exp(-(E - lamda) / Rg_T)
    cdef double xk_i_hat_temp = nom / denom
    cdef double xk_hat_temp = fi * xk_i_hat_temp
    return abs(xk_hat_temp - x_hat_sum)

cpdef tuple kinetics_cythonized(int t_tot, double dt, double accuracy, double Tstart, double R_G, double D0, double dQ,
                            double x_bulk, double GB_thickness, double E, double A, double gb_conc, 
                            double Tend=-1, double r=-1, str heat_treatment_type="linear"):
    cdef double R_g = (8.314 / (1.602 * 10 ** (-19))) / (6.022 * 10 ** (23))
    
    cdef np.ndarray[np.float32_t, ndim=1] temperatures, diffusion_coefficients, gb_concs, ife
    temperatures, diffusion_coefficients = get_T_and_D(t_tot, Tstart, D0, dQ, dt, Tend, heat_treatment_type, r)
    
    cdef int len_temp = len(temperatures)
    gb_concs = np.empty(len_temp, dtype=np.float32)
    ife = np.empty(len_temp, dtype=np.float32)
    
    cdef np.ndarray[np.float32_t, ndim=1] Rg_Ts = temperatures * R_g
    cdef np.ndarray[np.float32_t, ndim=1] diff_coefs = diffusion_coefficients
    
    cdef double m = 1
    cdef double N_tot = 1
    cdef double f = gb_site_fraction(R_G, GB_thickness)
    cdef double fi = f * m / N_tot
    cdef double fg = fi / f
    cdef double x_hat_ki = gb_conc
    cdef double x_global = (1 - f) * x_bulk + fi * x_hat_ki
    cdef double x_hat_sum = fi * x_hat_ki
    cdef double x_bulk_start = x_bulk

    cdef double lamda = 0.001

    cdef int indx
    cdef double Rg_T, D, result_value, new_lamda, nom, denom, nom_ln, denom_ln, frac_ln, lamda_frac, lhs, dx_hat

    for indx in range(len_temp):
        Rg_T = Rg_Ts[indx]
        D = diff_coefs[indx]
        
        result_value = 1
        while abs(result_value) > accuracy:
            result = minimize(obj_func, lamda, args=(Rg_T, x_hat_sum, fi, E), method="Nelder-Mead", options={'disp': False, 'xatol': accuracy})
            new_lamda = result.x[0]
            lamda = new_lamda
            result_value = result.fun
            if result_value < accuracy:
                break
        
        nom = exp(-(E - lamda) / Rg_T)
        nom = max(nom, 1e-300)  # Use max instead of np.clip for Cython
        denom = 1 + exp(-(E - lamda) / Rg_T)
        x_hat_ki = nom / denom
        gb_concs[indx] = x_hat_ki * fg

        nom_ln = 1 - x_bulk
        denom_ln = x_bulk
        frac_ln = np.log(nom_ln / denom_ln)
        lamda_frac = lamda / Rg_T
        lhs = 15 * x_bulk * D / (R_G**2)
        dx_hat = (-frac_ln - lamda_frac) * lhs
        
        if np.isnan(dx_hat).any():
            raise ValueError("NaN value encountered in rhs. Stopping calculation.")
        
        x_hat_sum += dx_hat * dt
        x_bulk = (x_global - x_hat_sum) / (1 - f)
        ife[indx] = (x_hat_sum * fg - x_bulk_start) / A
        
    return gb_concs, ife

