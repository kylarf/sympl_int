import numpy as np
import numba as nb
from numpy.typing import NDArray

from typing import Callable

type ConservativeForce = Callable[[NDArray[np.float64]], NDArray[np.float64]]
type FloatNDArray = NDArray[np.float64]


@nb.njit
def euler_step(
    acceleration: ConservativeForce,
    position: FloatNDArray,
    velocity: FloatNDArray,
    dt: np.float64,
) -> tuple[FloatNDArray, FloatNDArray]:
    q_next = position + dt * velocity
    v_next = velocity + dt * acceleration(position)
    return q_next, v_next


@nb.njit
def euler(
    acceleration: ConservativeForce,
    q0: FloatNDArray,
    v0: FloatNDArray,
    tspan: tuple[np.float64, np.float64],
    dt: np.float64,
    subsample: int = 1,
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
    t0, tf = tspan
    n_steps = int(np.floor((tf - t0) / dt)) + 1
    timesteps = t0 + dt * np.arange(0, n_steps, subsample, dtype=np.int64).astype(np.float64)
    n_save = timesteps.size

    n_dims = q0.size
    if n_dims != v0.size:
        raise ValueError(
            "Position and velocity must have equal number of components, "
            f"got {n_dims} and {v0.size} respectively."
        )

    q_t = np.zeros((n_save, n_dims), dtype=np.float64)
    v_t = np.zeros((n_save, n_dims), dtype=np.float64)
    q_t[0] = q0
    v_t[0] = v0

    q = q0.copy()
    v = v0.copy()

    i_save = 1
    for i in range(1, n_steps):
        q, v = euler_step(acceleration, q, v, dt)
        if not (i % subsample):
            q_t[i_save] = q
            v_t[i_save] = v
            i_save += 1

    return timesteps, q_t, v_t
