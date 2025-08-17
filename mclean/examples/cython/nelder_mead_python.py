
import numpy as np

def nelder_mead_python(func, x_start,
                step=0.1, no_improve_thr=10e-6,
                no_improv_break=10, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    """
    Nelder-Mead algorithm.

    Parameters:
    -----------
    func : callable
        The objective function to be minimized.
    x_start : numpy array
        Initial guess for the minimizer.
    step : float
        Look-around radius to initialize the simplex.
    no_improve_thr : float
        Threshold for improvement (default: 10e-6).
    no_improv_break : int
        Number of iterations with no improvement to break (default: 10).
    max_iter : int
        Maximum number of iterations (default: 0 for infinite).
    alpha : float
        Reflection coefficient (default: 1.0).
    gamma : float
        Expansion coefficient (default: 2.0).
    rho : float
        Contraction coefficient (default: -0.5).
    sigma : float
        Shrink coefficient (default: 0.5).

    Returns:
    --------
    best : numpy array
        Best parameter array found.
    best_score : float
        Score of the best parameter array.
    """
    
    # Initialize variables
    dim = len(x_start)
    prev_best = func(x_start)
    no_improve = 0
    res = [[x_start, prev_best]]

    # Initialize simplex
    for i in range(dim):
        x = np.copy(x_start)
        x[i] = x[i] + step
        score = func(x)
        res.append([x, score])

    # Sort
    res.sort(key=lambda x: x[1])
    best = res[0][0]
    best_score = res[0][1]

    iterations = 0

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
        rscore = func(xr)
        if res[0][1] <= rscore < res[-2][1]:
            res[-1] = [xr, rscore]
            continue

        # Expansion
        if rscore < res[0][1]:
            xe = x0 + gamma * (x0 - res[-1][0])
            escore = func(xe)
            if escore < rscore:
                res[-1] = [xe, escore]
                continue
            else:
                res[-1] = [xr, rscore]
                continue

        # Contraction
        xc = x0 + rho * (x0 - res[-1][0])
        cscore = func(xc)
        if cscore < res[-1][1]:
            res[-1] = [xc, cscore]
            continue

        # Reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma * (tup[0] - x1)
            score = func(redx)
            nres.append([redx, score])
        res = nres

    return best, best_score
