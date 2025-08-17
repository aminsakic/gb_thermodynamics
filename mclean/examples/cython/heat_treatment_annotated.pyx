import numpy as np

def isothermal(Tstart: float, t_tot: int, dt: float) -> np.ndarray:
    t_steps = np.arange(0, t_tot + dt, dt)
    return np.full_like(t_steps, Tstart, dtype=np.float32)
    
def linear_heat_treatment(Tstart: float, Tend: float, t_tot: int, dt: float) -> np.ndarray:
    t_steps = np.arange(0, t_tot + dt, dt)
    k = (Tend - Tstart) / t_tot
    temperatures = k * t_steps + Tstart
    return temperatures

def newton_cooling(Tstart: float, Tend: float, t_tot: int, dt: float, r: float) -> np.ndarray:
    t_steps = np.arange(0, t_tot + dt, dt)
    temperatures = Tend + (Tstart - Tend) * np.exp(-r * t_steps**0.5)
    return temperatures

def diff_coef(D0: float, dQ: float, T: np.ndarray, kB: float = (8.314 / (1.602 * 10**(-19))) / (6.022 * 10**(23))) -> np.ndarray:
    diffusion_coefficients = D0 * np.exp(-dQ / (kB * T))
    return diffusion_coefficients

#######################################################################
def get_T_and_D(t_tot: int, Tstart: float, D0: float, dQ: float, Tend: float = None, heat_treatment_type: str = "iso", dt: float = None, r: float = None) -> Tuple[np.ndarray, np.ndarray]:
    if heat_treatment_type == "iso":
        temperatures = isothermal(Tstart=Tstart, t_tot=t_tot, dt=dt)
    elif heat_treatment_type == "linear":
        temperatures = linear_heat_treatment(Tstart=Tstart, Tend=Tend, t_tot=t_tot, dt=dt)
    elif heat_treatment_type == "newton_cooling":
        temperatures = newton_cooling(Tstart=Tstart, Tend=Tend, t_tot=t_tot, dt=dt, r=r)
    else:
        raise ValueError("Invalid heat_treatment_type. Must be 'iso', 'linear', or 'newton_cooling'.")

    diffusion_coefficients = diff_coef(D0=D0, dQ=dQ, T=temperatures)
    return temperatures, diffusion_coefficients
