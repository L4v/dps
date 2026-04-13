from typing import Tuple
import numpy as np


def gaussian_plume(x, mean=0.1, sigma=0.05):
    return np.exp(-((x - mean) ** 2) / (2 * sigma**2))


def evolve(func, xs, ts, D, f0):
    q = f0(xs)
    dx = xs[1] - xs[0]
    dt = ts[1] - ts[0]

    results = [q.copy()]

    for _ in ts[:-1]:
        Tprim = func(q, dt, dx, D)
        results.append(Tprim.copy())
        q = np.copy(Tprim)

    return np.array(results)


# Neumann BC's where dq/dx = 0
def diffusion_neumann(q, dt, dx, D):
    q_new = q.copy()

    q_new[1:-1] = q[1:-1] + dt * D / (dx**2) * (q[:-2] - 2 * q[1:-1] + q[2:])

    q_new[0] = q_new[1]
    q_new[-1] = q_new[-2]

    return q_new


def generate_ground_truth(
    Lx: int,
    Nx: int,
    Lt: int,
    D: float,
    cfl_coeff=0.4,
    fn=diffusion_neumann,
    f0=gaussian_plume,
) -> Tuple:
    dx = Lx / Nx
    xs = np.arange(dx, Lx, dx)
    dt = cfl_coeff * (dx**2) / (2 * D)

    ts = np.arange(0, Lt, dt)
    qs = evolve(fn, xs, ts, D, f0)
    return xs, ts, qs
