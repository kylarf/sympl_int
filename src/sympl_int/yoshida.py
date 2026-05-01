import pathlib

import numpy as np
import numba as nb

from sympl_int.utils import ConservativeForce, FloatNDArray


@nb.njit
def full_coeff_array(base_coeffs: FloatNDArray) -> FloatNDArray:
    # Equation 5.18, Yoshida 1990
    w0 = 1.0 - 2.0 * np.sum(base_coeffs)
    return np.concatenate(
        (base_coeffs, np.array((w0,), dtype=np.float64), base_coeffs[::-1])
    )


def _load_coeffs_from_file(filepath: pathlib.Path) -> np.recarray:
    solutions = np.genfromtxt(
        filepath, delimiter=",", names=True, dtype=None, encoding="utf-8"
    ).copy()
    n_base_coeff = solutions.shape[0]
    if solutions.dtype.names is not None:
        for solution in solutions.dtype.names:
            full_coeffs = full_coeff_array(solutions[solution][:n_base_coeff])
            if full_coeffs.shape[0] != n_base_coeff:
                solutions.resize(full_coeffs.shape[0])
            solutions[solution] = full_coeffs
    solution_view = solutions.view(np.recarray)
    solution_view.flags.writeable = False
    return solution_view


_module_dir = pathlib.Path(__file__).resolve().parent
YOSHIDA8 = _load_coeffs_from_file(_module_dir / "yoshida8.csv")
YOSHIDA6 = _load_coeffs_from_file(_module_dir / "yoshida6.csv")


@nb.njit
def yoshida_step(
    acceleration: ConservativeForce,
    position: FloatNDArray,
    velocity: FloatNDArray,
    dt: np.float64,
    coeffs: FloatNDArray,
) -> tuple[FloatNDArray, FloatNDArray]:
    """
    NOTE: mutates original input states. Caller must make a copy if explicitly
    wanting to avoid this behavior
    """
    q = position
    v = velocity

    v += 0.5 * coeffs[0] * dt * acceleration(q)

    for i in range(coeffs.size - 1):
        q += coeffs[i] * dt * v
        v += 0.5 * (coeffs[i] + coeffs[i + 1]) * dt * acceleration(q)

    q += coeffs[-1] * dt * v
    v += 0.5 * coeffs[-1] * dt * acceleration(q)
    return q, v


@nb.njit
def _yoshida(
    acceleration: ConservativeForce,
    q0: FloatNDArray,
    v0: FloatNDArray,
    tspan: tuple[np.float64, np.float64],
    dt: np.float64,
    coeffs: FloatNDArray,
    subsample: int = 1,
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
    t0, tf = tspan
    n_steps = int(np.floor((tf - t0) / dt)) + 1
    timesteps = t0 + dt * np.arange(0, n_steps, subsample, dtype=np.int64).astype(
        np.float64
    )
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
        yoshida_step(acceleration, q, v, dt, coeffs)
        if not (i % subsample):
            q_t[i_save] = q
            v_t[i_save] = v
            i_save += 1

    return timesteps, q_t, v_t


def yoshida(
    acceleration: ConservativeForce,
    q0: FloatNDArray,
    v0: FloatNDArray,
    tspan: tuple[np.float64, np.float64],
    dt: np.float64,
    coeffs: FloatNDArray,
    subsample: int = 1,
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
    return _yoshida(acceleration, q0, v0, tspan, dt, coeffs, subsample)


def yoshida8(
    acceleration: ConservativeForce,
    q0: FloatNDArray,
    v0: FloatNDArray,
    tspan: tuple[np.float64, np.float64],
    dt: np.float64,
    subsample: int = 1,
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
    # Solution E, Yoshida 1990
    return yoshida(acceleration, q0, v0, tspan, dt, YOSHIDA8.A, subsample)


def yoshida6(
    acceleration: ConservativeForce,
    q0: FloatNDArray,
    v0: FloatNDArray,
    tspan: tuple[np.float64, np.float64],
    dt: np.float64,
    subsample: int = 1,
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
    # Solution A, Yoshida 1990
    return yoshida(acceleration, q0, v0, tspan, dt, YOSHIDA6.A, subsample)


def verlet(
    acceleration: ConservativeForce,
    q0: FloatNDArray,
    v0: FloatNDArray,
    tspan: tuple[np.float64, np.float64],
    dt: np.float64,
    subsample: int = 1,
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
    base_coeffs = np.array([], dtype=np.float64)
    return yoshida(
        acceleration, q0, v0, tspan, dt, full_coeff_array(base_coeffs), subsample
    )
