
import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt

cdef tuple nelder_mead_cython(func, double[:] x_start,
                double step, double no_improve_thr,
                int no_improv_break, int max_iter,
                double alpha, double gamma, double rho, double sigma,
                double Rg_T, double x_hat_sum, double fi, double E):
    
    # Initialize variables
    cdef int dim = x_start.shape[0]
    cdef double prev_best = func(x_start, Rg_T, x_hat_sum, fi, E)
    cdef int no_improve = 0
    cdef list res = [[np.copy(x_start), prev_best]]
    
    cdef double[:] best
    cdef double best_score

    # Initialize simplex
    cdef double[:] x
    for i in range(dim):
        x = np.copy(x_start)
        x[i] = x_start[i] + step
        score = func(x, Rg_T, x_hat_sum, fi, E)
        res.append([x, score])

    # Sort
    res.sort(key=lambda x: x[1])
    best = res[0][0]
    best_score = res[0][1]

    cdef int iterations = 0

    while True:
        # Break conditions
        if max_iter and iterations >= max_iter:
            break
        iterations += 1

        # Order
        res.sort(key=lambda x: x[1])
        best = res[0][0]
        best_score = res[0][1]

        # Track improvement
        if best_score < prev_best - no_improve_thr:
            no_improve = 0
            prev_best = best_score
        else:
            no_improve += 1

        if no_improve >= no_improv_break:
            break

        # Centroid
        x0 = np.zeros(dim)
        for tup in res[:-1]:
            x0 += tup[0]
        x0 /= (len(res) - 1)

        # Reflection
        xr = x0 + alpha * (x0 - res[-1][0])
        rscore = func(xr, Rg_T, x_hat_sum, fi, E)
        if res[0][1] <= rscore < res[-2][1]:
            res[-1] = [xr, rscore]
            continue

        # Expansion
        if rscore < res[0][1]:
            xe = x0 + gamma * (x0 - res[-1][0])
            escore = func(xe, Rg_T, x_hat_sum, fi, E)
            if escore < rscore:
                res[-1] = [xe, escore]
                continue
            else:
                res[-1] = [xr, rscore]
                continue

        # Contraction
        xc = x0 + rho * (x0 - res[-1][0])
        cscore = func(xc, Rg_T, x_hat_sum, fi, E)
        if cscore < res[-1][1]:
            res[-1] = [xc, cscore]
            continue

        # Reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma * (tup[0] - x1)
            score = func(redx, Rg_T, x_hat_sum, fi, E)
            nres.append([redx, score])
        res = nres

    return best, best_score


cdef double[:] isothermal(float Tstart, int t_tot, float dt):
    cdef int n = int(t_tot / dt) + 1
    cdef double[:] temperatures = np.zeros(n, dtype=np.float32)
    cdef int i
    for i in range(n):
        temperatures[i] = Tstart
    return temperatures

cdef double[:] linear_heat_treatment(float Tstart, float Tend, int t_tot, float dt):
    cdef int n = int(t_tot / dt) + 1
    cdef double[:] temperatures = np.zeros(n, dtype=np.float64)
    cdef float slope = (Tend - Tstart) / t_tot
    cdef int i
    for i in range(n):
        temperatures[i] = Tstart + slope * (i * dt)
    return temperatures

cdef double[:] newton_cooling(float Tstart, float Tend, int t_tot, float dt, float r):
    cdef int n = int(t_tot / dt) + 1
    cdef double[:] temperatures = np.zeros(n, dtype=np.float32)
    cdef int i
    for i in range(n):
        temperatures[i] = Tend + (Tstart - Tend) * exp(-r * sqrt(i * dt))
    return temperatures


cdef double[:] diff_coef(float D0, float dQ, double[:] T, float kB):
    cdef int n = T.shape[0]
    cdef double[:] diffusion_coefficients = np.zeros(n, dtype=np.float64)
    cdef int i
    for i in range(n):
        diffusion_coefficients[i] = D0 * exp(-dQ / (kB * T[i]))
    return diffusion_coefficients

cpdef tuple get_T_and_D(int t_tot, float Tstart, float D0, float dQ, float dt, float Tend=-1, str heat_treatment_type="iso", float r=-1):
    cdef double[:] temperatures, diffusion_coefficients
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

    return temperatures, diffusion_coefficients


# Calculate fraction of GB sites compared to all sites in the system
cdef double gb_site_fraction(double R_G=100e-6, double gb_width=8.4e-10):
    cdef double V_G = R_G**3
    cdef double V_GB = V_G - (R_G - gb_width)**3
    return V_GB / V_G

cdef double obj_func(double[:] lamda, double Rg_T, double x_hat_sum, double fi, double E):
    cdef double nom = exp(-(E - lamda[0]) / Rg_T)
    nom = min(nom, 1e300)  # Use min instead of np.clip for Cython
    cdef double denom = 1 + exp(-(E - lamda[0]) / Rg_T)
    cdef double xk_i_hat_temp = nom / denom
    cdef double xk_hat_temp = fi * xk_i_hat_temp
    return abs(xk_hat_temp - x_hat_sum)


cdef double wrapped_obj_func(double[:] lamda, double Rg_T, double x_hat_sum, double fi, double E):
    return obj_func(lamda, Rg_T, x_hat_sum, fi, E)


cpdef tuple kinetics_cythonized_nm(int t_tot, double dt, double accuracy, double Tstart, double R_G, double D0, double dQ,
                                   double x_bulk, double GB_thickness, double E, double A, double gb_conc,
                                   double Tend=-1, double r=-1, str heat_treatment_type="linear"):
    # Minimizer defaults
    cdef double step = 0.1
    cdef double no_improve_thr = 10e-6
    cdef int no_improv_break = 10
    cdef int max_iter = 0
    cdef double alpha = 1.
    cdef double gamma = 2.
    cdef double rho = -0.5
    cdef double sigma = 0.5

    cdef double R_g = (8.314 / (1.602 * 10 ** (-19))) / (6.022 * 10 ** (23))

    cdef double[:] temperatures, diffusion_coefficients, gb_concs, ife
    temperatures, diffusion_coefficients = get_T_and_D(t_tot, Tstart, D0, dQ, dt, Tend, heat_treatment_type, r)

    cdef int len_temp = temperatures.shape[0]
    gb_concs = np.empty(len_temp, dtype=np.float64)
    ife = np.empty(len_temp, dtype=np.float64)

    cdef double[:] Rg_Ts = np.array(temperatures) * R_g
    cdef double[:] diff_coefs = diffusion_coefficients

    cdef double m = 1
    cdef double N_tot = 1
    cdef double f = gb_site_fraction(R_G, GB_thickness)
    cdef double fi = f * m / N_tot
    cdef double fg = fi / f
    cdef double x_hat_ki = gb_conc
    cdef double x_global = (1 - f) * x_bulk + fi * x_hat_ki
    cdef double x_hat_sum = fi * x_hat_ki
    cdef double x_bulk_start = x_bulk

    cdef double[:] lamda = np.array([0.001], dtype=np.float64)
    cdef double[:] new_lamda

    cdef int indx
    cdef double Rg_T, D, result_value, nom, denom, nom_ln, denom_ln, frac_ln, lamda_frac, lhs, dx_hat

    for indx in range(len_temp):
        Rg_T = Rg_Ts[indx]
        D = diff_coefs[indx]

        # Use wrapped_obj_func instead of lambda
        result = nelder_mead_cython(wrapped_obj_func, lamda,
                                    step, no_improve_thr, no_improv_break, max_iter, alpha, gamma, rho, sigma, Rg_T, x_hat_sum, fi, E)
        new_lamda = result[0]
        lamda = new_lamda
        result_value = result[-1]  # This value is not used, consider removing it if unnecessary

        nom = exp(-(E - lamda[0]) / Rg_T)
        nom = max(nom, 1e-300)  # Use max instead of np.clip for Cython
        denom = 1 + exp(-(E - lamda[0]) / Rg_T)
        x_hat_ki = nom / denom
        gb_concs[indx] = x_hat_ki * fg

        nom_ln = 1 - x_bulk
        denom_ln = x_bulk
        frac_ln = np.log(nom_ln / denom_ln)
        lamda_frac = lamda[0] / Rg_T
        lhs = 15 * x_bulk * D / (R_G**2)
        dx_hat = (-frac_ln - lamda_frac) * lhs

        if np.isnan(dx_hat).any():
            raise ValueError("NaN value encountered in dx_hat. Stopping calculation.")

        x_hat_sum += dx_hat * dt
        x_bulk = (x_global - x_hat_sum) / (1 - f)
        ife[indx] = (x_hat_sum * fg - x_bulk_start) / A

    return gb_concs, ife

