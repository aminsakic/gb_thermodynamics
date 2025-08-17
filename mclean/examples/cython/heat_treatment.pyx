import numpy as np
cimport numpy as np  # Cython-specific import for type declarations
from libc.math cimport exp, sqrt

# Function declarations
cdef float[:] isothermal(float Tstart, int t_tot, float dt):
    cdef int n = int(t_tot / dt) + 1
    cdef float[:] t_steps = np.zeros(n, dtype=np.float32)
    cdef float[:] temperatures = np.zeros(n, dtype=np.float32)
    cdef int i
    for i in range(n):
        t_steps[i] = i * dt
        temperatures[i] = Tstart
    return temperatures

cdef float[:] linear_heat_treatment(float Tstart, float Tend, int t_tot, float dt):
    cdef int n = int(t_tot / dt) + 1
    cdef float[:] t_steps = np.zeros(n, dtype=np.float32)
    cdef float[:] temperatures = np.zeros(n, dtype=np.float32)
    cdef float k = (Tend - Tstart) / t_tot
    cdef int i
    for i in range(n):
        t_steps[i] = i * dt
        temperatures[i] = k * t_steps[i] + Tstart
    return temperatures

cdef float[:] newton_cooling(float Tstart, float Tend, int t_tot, float dt, float r):
    cdef int n = int(t_tot / dt) + 1
    cdef float[:] t_steps = np.zeros(n, dtype=np.float32)
    cdef float[:] temperatures = np.zeros(n, dtype=np.float32)
    cdef int i
    for i in range(n):
        t_steps[i] = i * dt
        temperatures[i] = Tend + (Tstart - Tend) * exp(-r * sqrt(t_steps[i]))
    return temperatures

cdef float[:] diff_coef(float D0, float dQ, float[:] T, float kB):
    cdef int n = T.shape[0]
    cdef float[:] diffusion_coefficients = np.zeros(n, dtype=np.float32)
    cdef int i
    for i in range(n):
        diffusion_coefficients[i] = D0 * exp(-dQ / (kB * T[i]))
    return diffusion_coefficients

# Main function
cpdef tuple get_T_and_D(int t_tot, float Tstart, float D0, float dQ, float dt, float Tend=-1, str heat_treatment_type="iso", float r=-1):
    cdef float[:] temperatures
    cdef float[:] diffusion_coefficients
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
