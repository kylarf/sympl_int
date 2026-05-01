import numpy as np
import numba as nb
import scipy as sp
from numpy.typing import NDArray, ArrayLike

from typing import Callable, Optional

type ODEFunc = Callable[[np.floating, NDArray[np.floating]], NDArray[np.floating]]


@nb.njit
def verlet_integrate(
    a: ODEFunc,
    y0: NDArray[np.floating],
    tspan: NDArray[np.floating],
    dt: Optional[np.floating] = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    if tspan.size < 2:
        raise ValueError("tspan must have at least 2 elements.")
    elif tspan.size == 2:
        if dt is not None:
            t = np.arange(tspan[0], tspan[1]+dt, dt, dtype=tspan.dtype)
        else:
            t = tspan
    else:
        if dt is not None:
            raise ValueError("Cannot specify both evaluation times and dt.")
        t = tspan

    # preallocate solution array
    n_states = y0.size
    if n_states % 2:
        raise ValueError(
            "Number of solution states (set by initial condition vector) must "
            "be even (i.e. equal number of positions and velocities)."
        )
    N = n_states // 2
    y = np.zeros((t.size, n_states), dtype=y0.dtype)

    y[0, :] = y0
    for i in range(t.size - 1):
        # get timestep
        delta_t = t[i + 1] - t[i]
        # extract current states
        x_i = y[i, :N]
        v_i = y[i, N:]
        # compute acceleration
        a_i = a(t[i], x_i)
        # kick
        v_i_half = v_i + 0.5 * a_i * delta_t
        # drift
        x_i_plus_1 = x_i + v_i_half * delta_t
        # kick
        a_i_plus_1 = a(t[i+1], x_i_plus_1)
        v_i_plus_1 = v_i_half + 0.5 * a_i_plus_1 * delta_t
        # store states
        y[i + 1, :N] = x_i_plus_1
        y[i + 1, N:] = v_i_plus_1

    return t, y
