import numba as nb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from sympl_int import yoshida6, yoshida8, rk4, verlet
from sympl_int.yoshida import yoshida, YOSHIDA8, YOSHIDA6
from sympl_int.utils import OrbitalElements

from solution_comparisons import get_yoshida_integrator


type FloatNDArray = NDArray[np.float64]


METHODS8 = tuple(
    (f"yoshida8{variant}", get_yoshida_integrator(YOSHIDA8, variant))
    for variant in ("A", "B", "C", "D", "E")
)

METHODS6 = tuple(
    (f"yoshida6{variant}", get_yoshida_integrator(YOSHIDA6, variant))
    for variant in ("A", "B", "C")
)



def two_body_acceleration(mu: float):
    @nb.njit
    def acceleration(q: FloatNDArray) -> FloatNDArray:
        r = np.linalg.norm(q)
        return -mu * q / r**3
    return acceleration


def kepler_initial_state(
    mu: float = 1.0,
    a: float = 1.0,
    e: float = 0.9,
) -> tuple[FloatNDArray, FloatNDArray, float]:
    oe = OrbitalElements(
        mu=np.float64(mu),
        semimajor_axis=np.array(a, dtype=np.float64),
        eccentricity=np.array(e, dtype=np.float64),
        inclination=np.array(0.0, dtype=np.float64),
        lon_ascending_node=np.array(0.0, dtype=np.float64),
        arg_periapsis=np.array(0.0, dtype=np.float64),
        true_anomaly=np.array(0.0, dtype=np.float64),  # periapsis
        angular_momentum=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )
    r0, v0 = oe.to_rv()
    return r0.astype(np.float64), v0.astype(np.float64), float(oe.period)


def reference_orbit(
    mu: float = 1.0,
    a: float = 1.0,
    e: float = 0.9,
    n_points: int = 2000,
) -> FloatNDArray:
    nu = np.linspace(0.0, 2.0 * np.pi, n_points, dtype=np.float64)
    oe = OrbitalElements(
        mu=np.float64(mu),
        semimajor_axis=np.full_like(nu, a, dtype=np.float64),
        eccentricity=np.full_like(nu, e, dtype=np.float64),
        inclination=np.zeros_like(nu, dtype=np.float64),
        lon_ascending_node=np.zeros_like(nu, dtype=np.float64),
        arg_periapsis=np.zeros_like(nu, dtype=np.float64),
        true_anomaly=nu,
        angular_momentum=np.zeros((nu.size, 3), dtype=np.float64),
    )
    r_ref, _ = oe.to_rv()
    return r_ref.astype(np.float64)


def choose_subsample(steps_per_orbit: int, target_saved_per_orbit: int = 1000) -> int:
    return max(1, steps_per_orbit // target_saved_per_orbit)


def propagate_orbit(
    method,
    acceleration,
    q0: FloatNDArray,
    v0: FloatNDArray,
    period: float,
    n_orbits: int,
    steps_per_orbit: int,
) -> FloatNDArray:
    dt = period / steps_per_orbit
    subsample = choose_subsample(steps_per_orbit, target_saved_per_orbit=1000)
    tspan = (np.float64(0.0), np.float64(n_orbits * period))

    _, r, _ = method(
        acceleration,
        q0,
        v0,
        tspan,
        np.float64(dt),
        int(subsample),
    )

    return r


def plot_method_overplot(
    method_name: str,
    r_ref: FloatNDArray,
    r_2000: FloatNDArray,
    r_4000: FloatNDArray,
    r_8000: FloatNDArray,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(r_2000[:, 0], r_2000[:, 1], label="T/2000")
    ax.plot(r_4000[:, 0], r_4000[:, 1], label="T/4000")
    ax.plot(r_8000[:, 0], r_8000[:, 1], label="T/8000")
    ax.plot(r_ref[:, 0], r_ref[:, 1], linestyle=":", color="k", label="reference")

    ax.plot([0], [0], marker=".", linestyle="None")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{method_name}: 5000 orbits")
    ax.legend(loc=1)

    fig.tight_layout()
    fig.savefig(f"{method_name}_orbit_overplot.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    mu = 1.0
    a = 1.0
    e = 0.9
    n_orbits = 5000

    q0, v0, period = kepler_initial_state(mu=mu, a=a, e=e)
    acceleration = two_body_acceleration(mu)
    r_ref = reference_orbit(mu=mu, a=a, e=e)

    for method_name, method in METHODS6:
        print(f"{method_name}: T/2000")
        r_2000 = propagate_orbit(
            method, acceleration, q0, v0, period, n_orbits, 2000
        )

        print(f"{method_name}: T/4000")
        r_4000 = propagate_orbit(
            method, acceleration, q0, v0, period, n_orbits, 4000
        )

        print(f"{method_name}: T/8000")
        r_8000 = propagate_orbit(
            method, acceleration, q0, v0, period, n_orbits, 8000
        )

        plot_method_overplot(method_name, r_ref, r_2000, r_4000, r_8000)

    for method_name, method in METHODS8:
        print(f"{method_name}: T/2000")
        r_2000 = propagate_orbit(
            method, acceleration, q0, v0, period, n_orbits, 2000
        )

        print(f"{method_name}: T/4000")
        r_4000 = propagate_orbit(
            method, acceleration, q0, v0, period, n_orbits, 4000
        )

        print(f"{method_name}: T/8000")
        r_8000 = propagate_orbit(
            method, acceleration, q0, v0, period, n_orbits, 8000
        )

        plot_method_overplot(method_name, r_ref, r_2000, r_4000, r_8000)


if __name__ == "__main__":
    main()
