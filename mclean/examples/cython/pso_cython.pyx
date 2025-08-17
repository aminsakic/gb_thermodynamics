import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt

class Particle:
    def __init__(self, list bounds):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.array([np.random.uniform(-abs(high - low), abs(high - low)) for low, high in bounds])
        self.best_position = self.position.copy()
        self.best_score = float('inf')

    def update_velocity(self, np.ndarray[np.double_t, ndim=1] global_best_position, double w, double c1, double c2):
        cdef double r1, r2
        r1, r2 = np.random.random(2)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, list bounds):
        self.position += self.velocity
        cdef int i
        for i in range(len(bounds)):
            self.position[i] = max(min(self.position[i], bounds[i][1]), bounds[i][0])

cdef tuple pso_cython(func, list bounds, double Rg_T, double x_hat_sum, double fi, double E, int num_particles=30, int max_iter=100, double w=0.5, double c1=1.0, double c2=2.0, double tol=1e-6, int stagnation_iter=10):
    cdef list particles = [Particle(bounds) for _ in range(num_particles)]
    cdef np.ndarray[np.double_t, ndim=1] global_best_position = particles[0].position.copy()
    cdef double global_best_score = float('inf')
    cdef int no_improvement_count = 0

    cdef int iter
    for iter in range(max_iter):
        for particle in particles:
            score = func(particle.position, Rg_T, x_hat_sum, fi, E)
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()
            if score < global_best_score:
                global_best_score = score
                global_best_position = particle.position.copy()
                no_improvement_count = 0  # Reset the stagnation counter
            else:
                no_improvement_count += 1

        if no_improvement_count >= stagnation_iter:
            break

        if global_best_score < tol:
            break

        for particle in particles:
            particle.update_velocity(global_best_position, w, c1, c2)
            particle.update_position(bounds)

    return global_best_position, global_best_score

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

cdef double gb_site_fraction(double R_G=100e-6, double gb_width=8.4e-10):
    cdef double V_G = R_G**3
    cdef double V_GB = V_G - (R_G - gb_width)**3
    return V_GB / V_G

cdef double obj_func(double[:] lamda, double Rg_T, double x_hat_sum, double fi, double E):
    cdef double nom = exp(-(E - lamda[0]) / Rg_T)
    nom = min(nom, 1e300)  
    cdef double denom = 1 + exp(-(E - lamda[0]) / Rg_T)
    cdef double xk_i_hat_temp = nom / denom
    cdef double xk_hat_temp = fi * xk_i_hat_temp
    return abs(xk_hat_temp - x_hat_sum)

cdef double wrapped_obj_func(double[:] lamda, double Rg_T, double x_hat_sum, double fi, double E):
    return obj_func(lamda, Rg_T, x_hat_sum, fi, E)

cpdef tuple kinetics_cythonized_pso(int t_tot, double dt, double Tstart, double R_G, double D0, double dQ,
                                   double x_bulk, double GB_thickness, double E, double A, double gb_conc,
                                   double Tend=-1, double r=-1, str heat_treatment_type="linear", int num_particles=30, 
                                   int max_iter=100, float w=0.5, float c1=1.0, float c2=2.0, float tol=1e-6, int stagnation_iter=10):
    
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
        result = pso_cython(wrapped_obj_func, [(-3, 3)], Rg_T, x_hat_sum, fi, E, num_particles, max_iter, w, c1, c2, tol, stagnation_iter)
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
