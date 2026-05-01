import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from sympl_int import yoshida6, yoshida8, rk4, verlet
from sympl_int.utils import OrbitalElements


type FloatNDArray = NDArray[np.float64]
type Integrator = Callable[
    [Callable[[FloatNDArray], FloatNDArray], FloatNDArray, FloatNDArray, tuple[np.float64, np.float64], np.float64, int],
    tuple[FloatNDArray, FloatNDArray, FloatNDArray],
]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    func: Integrator
    force_evals_per_step: int


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("Verlet", verlet, 1),
    MethodSpec("RK4", rk4, 4),
    MethodSpec("Yoshida6", yoshida6, 7),
    MethodSpec("Yoshida8", yoshida8, 15),
)


def two_body_acceleration(mu: float) -> Callable[[FloatNDArray], FloatNDArray]:
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
    nu = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=True, dtype=np.float64)
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


def specific_energy(mu: float, r: FloatNDArray, v: FloatNDArray) -> FloatNDArray:
    norm_r = np.linalg.norm(r, axis=-1)
    norm_v = np.linalg.norm(v, axis=-1)
    return 0.5 * norm_v**2 - mu / norm_r


def specific_angmom(r: FloatNDArray, v: FloatNDArray) -> FloatNDArray:
    return np.cross(r, v)


def eccentricity_vector(mu: float, r: FloatNDArray, v: FloatNDArray) -> FloatNDArray:
    h_vec = np.cross(r, v)
    norm_r = np.linalg.norm(r, axis=-1)
    return np.cross(v, h_vec) / mu - r / norm_r[..., None]


def unwrap_planar_apsides(e_vec: FloatNDArray) -> FloatNDArray:
    angle = np.arctan2(e_vec[..., 1], e_vec[..., 0])
    return np.unwrap(angle)


@dataclass
class RunResult:
    method: str
    dt: float
    steps_per_orbit: float
    force_evals_per_orbit: float
    n_orbits: int
    subsample: int
    t: FloatNDArray
    r: FloatNDArray
    v: FloatNDArray

    max_abs_energy_error: float
    final_apsis_drift: float
    max_abs_apsis_drift: float

    max_abs_h_error: float
    max_abs_a_error: float
    max_abs_e_error: float


def analyze_run(
    method: MethodSpec,
    mu: float,
    period: float,
    n_orbits: int,
    dt: float,
    subsample: int,
    t: FloatNDArray,
    r: FloatNDArray,
    v: FloatNDArray,
) -> RunResult:
    energy = specific_energy(mu, r, v)
    energy_error = energy - energy[0]

    h_vec = specific_angmom(r, v)
    h_mag = np.linalg.norm(h_vec, axis=-1)
    h_error = h_mag - h_mag[0]

    oe = OrbitalElements.from_rv(r, v, np.float64(mu))
    e_vec = eccentricity_vector(mu, r, v)
    apsis = unwrap_planar_apsides(e_vec)
    apsis_drift = apsis - apsis[0]

    dt_integrator = dt
    steps_per_orbit = period / dt_integrator
    force_evals_per_orbit = steps_per_orbit * method.force_evals_per_step

    return RunResult(
        method=method.name,
        dt=dt_integrator,
        steps_per_orbit=steps_per_orbit,
        force_evals_per_orbit=force_evals_per_orbit,
        n_orbits=n_orbits,
        subsample=subsample,
        t=t,
        r=r,
        v=v,
        max_abs_energy_error=float(np.max(np.abs(energy_error))),
        final_apsis_drift=float(apsis_drift[-1]),
        max_abs_apsis_drift=float(np.max(np.abs(apsis_drift))),
        max_abs_h_error=float(np.max(np.abs(h_error))),
        max_abs_a_error=float(np.max(np.abs(oe.semimajor_axis - oe.semimajor_axis[0]))),
        max_abs_e_error=float(np.max(np.abs(oe.eccentricity - oe.eccentricity[0]))),
    )


def choose_subsample(steps_per_orbit: int, target_saved_per_orbit: int = 1000) -> int:
    return max(1, steps_per_orbit // target_saved_per_orbit)


def run_case(
    method: MethodSpec,
    acceleration: Callable[[FloatNDArray], FloatNDArray],
    q0: FloatNDArray,
    v0: FloatNDArray,
    mu: float,
    period: float,
    n_orbits: int,
    dt: float,
    subsample: int,
) -> RunResult:
    tspan = (np.float64(0.0), np.float64(n_orbits * period))
    t, r, v = method.func(
        acceleration,
        q0,
        v0,
        tspan,
        np.float64(dt),
        int(subsample),
    )
    return analyze_run(method, mu, period, n_orbits, dt, subsample, t, r, v)


def sweep_fixed_steps_per_orbit(
    methods: tuple[MethodSpec, ...],
    acceleration: Callable[[FloatNDArray], FloatNDArray],
    q0: FloatNDArray,
    v0: FloatNDArray,
    mu: float,
    period: float,
    n_orbits: int,
    steps_per_orbit_list: list[int],
    target_saved_per_orbit: int = 1000,
) -> list[RunResult]:
    results: list[RunResult] = []
    for steps_per_orbit in steps_per_orbit_list:
        dt = period / steps_per_orbit
        subsample = choose_subsample(steps_per_orbit, target_saved_per_orbit)
        for method in methods:
            print(f"Running {method.name} with {steps_per_orbit=}, {subsample=}")
            results.append(
                run_case(
                    method,
                    acceleration,
                    q0,
                    v0,
                    mu,
                    period,
                    n_orbits,
                    dt,
                    subsample,
                )
            )
    return results


def sweep_fixed_force_budget(
    methods: tuple[MethodSpec, ...],
    acceleration: Callable[[FloatNDArray], FloatNDArray],
    q0: FloatNDArray,
    v0: FloatNDArray,
    mu: float,
    period: float,
    n_orbits: int,
    force_evals_per_orbit_list: list[int],
    target_saved_per_orbit: int = 1000,
    min_steps_per_orbit: int = 20,
) -> list[RunResult]:
    results: list[RunResult] = []
    for force_budget in force_evals_per_orbit_list:
        for method in methods:
            steps_per_orbit = force_budget // method.force_evals_per_step
            if steps_per_orbit < min_steps_per_orbit:
                continue
            dt = period / steps_per_orbit
            subsample = choose_subsample(steps_per_orbit, target_saved_per_orbit)
            print(f"Running {method.name} with {steps_per_orbit=}, {subsample=}")
            results.append(
                run_case(
                    method,
                    acceleration,
                    q0,
                    v0,
                    mu,
                    period,
                    n_orbits,
                    dt,
                    subsample,
                )
            )
    return results


def extract_metric_series(
    results: list[RunResult],
    x_attr: str,
    y_attr: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for result in results:
        grouped.setdefault(result.method, []).append(
            (getattr(result, x_attr), getattr(result, y_attr))
        )

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method, pairs in grouped.items():
        pairs_sorted = sorted(pairs, key=lambda p: p[0])
        out[method] = (
            np.array([p[0] for p in pairs_sorted], dtype=np.float64),
            np.array([p[1] for p in pairs_sorted], dtype=np.float64),
        )
    return out


def plot_headline_summary(
    results: list[RunResult],
    x_attr: str,
    x_label: str,
    title_prefix: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    energy_series = extract_metric_series(results, x_attr, "max_abs_energy_error")
    apsis_series = extract_metric_series(results, x_attr, "final_apsis_drift")

    for method, (x, y) in energy_series.items():
        axes[0].plot(x, np.abs(y), marker="o", label=method)
    axes[0].set_ylabel("max |ΔE|")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[0].set_title(title_prefix)

    for method, (x, y) in apsis_series.items():
        axes[1].plot(x, np.abs(y), marker="o", label=method)
    axes[1].set_ylabel("|final Δϖ| [rad]")
    axes[1].set_xlabel(x_label)
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()


def plot_secondary_summary(
    results: list[RunResult],
    x_attr: str,
    x_label: str,
    title_prefix: str,
) -> None:
    metrics = (
        ("max_abs_h_error", "max |Δ|h||"),
        ("max_abs_a_error", "max |Δa|"),
        ("max_abs_e_error", "max |Δe|"),
        ("max_abs_apsis_drift", "max |Δϖ| [rad]"),
    )

    fig, axes = plt.subplots(len(metrics), 1, figsize=(7, 12), sharex=True)
    for ax, (y_attr, y_label) in zip(axes, metrics):
        series = extract_metric_series(results, x_attr, y_attr)
        for method, (x, y) in series.items():
            ax.plot(x, np.abs(y), marker="o", label=method)
        ax.set_ylabel(y_label)
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_title(title_prefix)
    axes[-1].set_xlabel(x_label)
    axes[0].legend()
    fig.tight_layout()


def pick_result(
    results: list[RunResult],
    method_name: str,
    target_x: float,
    x_attr: str,
) -> RunResult:
    candidates = [r for r in results if r.method == method_name]
    return min(candidates, key=lambda r: abs(getattr(r, x_attr) - target_x))


def plot_time_history(result: RunResult, mu: float, period: float) -> None:
    energy = specific_energy(mu, result.r, result.v)
    energy_err = energy - energy[0]

    e_vec = eccentricity_vector(mu, result.r, result.v)
    apsis = unwrap_planar_apsides(e_vec)
    apsis_drift = apsis - apsis[0]

    orbit_count = result.t / period

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(orbit_count, energy_err)
    axes[0].set_ylabel("ΔE")
    axes[0].set_title(
        f"{result.method}: dt={result.dt:.3e}, "
        f"steps/orbit={result.steps_per_orbit:.1f}, "
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(orbit_count, apsis_drift)
    axes[1].set_ylabel("Δϖ [rad]")
    axes[1].set_xlabel("orbit count")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()


def plot_orbit(result: RunResult, r_ref: FloatNDArray) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(r_ref[:, 0], r_ref[:, 1], linestyle="--", alpha=0.6, label="reference")
    ax.plot(result.r[:, 0], result.r[:, 1], label=result.method)
    ax.plot(result.r[0, 0], result.r[0, 1], marker="o", linestyle="None", label="start")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"{result.method}: dt={result.dt:.3e}, "
        f"steps/orbit={result.steps_per_orbit:.1f}, "
        f"force evals/orbit={result.force_evals_per_orbit:.1f}"
    )
    ax.legend()
    fig.tight_layout()


def plot_orbit_gallery(
    results: list[RunResult],
    x_attr: str,
    title_prefix: str,
    r_ref: FloatNDArray,
) -> None:
    results_sorted = sorted(results, key=lambda r: (r.method, getattr(r, x_attr)))
    n = len(results_sorted)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, result in zip(axes, results_sorted):
        ax.plot(r_ref[:, 0], r_ref[:, 1], linestyle="--", alpha=0.5)
        ax.plot(result.r[:, 0], result.r[:, 1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        x_val = getattr(result, x_attr)
        ax.set_title(f"{result.method}\n{x_attr}={x_val:.1f}", fontsize=9)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title_prefix)
    fig.tight_layout()


def plot_orbit_comparison(
    results: list[RunResult],
    x_attr: str,
    target_x: float,
    r_ref: FloatNDArray,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(r_ref[:, 0], r_ref[:, 1], linestyle="--", alpha=0.6, label="reference")

    for method_name in sorted({r.method for r in results}):
        result = pick_result(results, method_name, target_x, x_attr)
        ax.plot(result.r[:, 0], result.r[:, 1], label=method_name)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"Orbit comparison near {x_attr}={target_x}")
    fig.tight_layout()


def format_steps_per_orbit_label(result: RunResult) -> str:
    return f"{result.steps_per_orbit:.0f} steps/orbit"


def format_force_budget_label(result: RunResult) -> str:
    return f"{result.force_evals_per_orbit:.0f} fe/orbit"


def plot_method_error_overplots(
    results: list[RunResult],
    method_name: str,
    group_attr: str,
    group_labeler: Callable[[RunResult], str],
    mu: float,
    period: float,
    title_prefix: str,
) -> None:
    method_results = [r for r in results if r.method == method_name]
    method_results = sorted(method_results, key=lambda r: getattr(r, group_attr))

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for result in method_results:
        energy = specific_energy(mu, result.r, result.v)
        energy_err = energy - energy[0]

        e_vec = eccentricity_vector(mu, result.r, result.v)
        apsis = unwrap_planar_apsides(e_vec)
        apsis_err = apsis - apsis[0]

        orbit_count = result.t / period
        label = group_labeler(result)

        axes[0].plot(orbit_count, energy_err, label=label)
        axes[1].plot(orbit_count, apsis_err, label=label)

    axes[0].set_ylabel("ΔE")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"{title_prefix}: {method_name}")

    axes[1].set_ylabel("Δϖ [rad]")
    axes[1].set_xlabel("orbit count")
    axes[1].grid(True, alpha=0.3)

    axes[0].legend()
    fig.tight_layout()


def main() -> None:
    mu = 1.0
    a = 1.0
    e = 0.9
    n_orbits = 5000

    q0, v0, period = kepler_initial_state(mu=mu, a=a, e=e)
    acceleration = two_body_acceleration(mu)
    r_ref = reference_orbit(mu=mu, a=a, e=e)

    # Fixed-step-size study
    steps_per_orbit_list = [1000, 2000, 4000, 8000, 16000, 32000]
    fixed_dt_results = sweep_fixed_steps_per_orbit(
        METHODS,
        acceleration,
        q0,
        v0,
        mu,
        period,
        n_orbits,
        steps_per_orbit_list,
        target_saved_per_orbit=1000,
    )

    # Fixed-force-budget study
    # force_evals_per_orbit_list = [1000, 2000, 4000, 8000, 16000, 32000]
    # fixed_budget_results = sweep_fixed_force_budget(
    #     METHODS,
    #     acceleration,
    #     q0,
    #     v0,
    #     mu,
    #     period,
    #     n_orbits,
    #     force_evals_per_orbit_list,
    #     target_saved_per_orbit=200,
    #     min_steps_per_orbit=20,
    # )

    plot_headline_summary(
        fixed_dt_results,
        x_attr="steps_per_orbit",
        x_label="steps per orbit",
        title_prefix=f"Headline metrics: fixed macro-step size, e={e}, {n_orbits} orbits",
    )

    # plot_headline_summary(
    #     fixed_budget_results,
    #     x_attr="force_evals_per_orbit",
    #     x_label="force evaluations per orbit",
    #     title_prefix=f"Headline metrics: fixed force budget, e={e}, {n_orbits} orbits",
    # )

    plot_secondary_summary(
        fixed_dt_results,
        x_attr="steps_per_orbit",
        x_label="steps per orbit",
        title_prefix=f"Secondary metrics: fixed macro-step size, e={e}, {n_orbits} orbits",
    )

    # plot_secondary_summary(
    #     fixed_budget_results,
    #     x_attr="force_evals_per_orbit",
    #     x_label="force evaluations per orbit",
    #     title_prefix=f"Secondary metrics: fixed force budget, e={e}, {n_orbits} orbits",
    # )

    # plot_orbit_gallery(
    #     fixed_dt_results,
    #     x_attr="steps_per_orbit",
    #     title_prefix=f"Orbit gallery: fixed macro-step size, e={e}, {n_orbits} orbits",
    #     r_ref=r_ref,
    # )

    # plot_orbit_gallery(
    #     fixed_budget_results,
    #     x_attr="force_evals_per_orbit",
    #     title_prefix=f"Orbit gallery: fixed force budget, e={e}, {n_orbits} orbits",
    #     r_ref=r_ref,
    # )

    plot_orbit_comparison(
        fixed_dt_results,
        x_attr="steps_per_orbit",
        target_x=4000.0,
        r_ref=r_ref,
    )

    # plot_orbit_comparison(
    #     fixed_budget_results,
    #     x_attr="force_evals_per_orbit",
    #     target_x=8000.0,
    #     r_ref=r_ref,
    # )

    target_steps_per_orbit = 8000.0
    for method_name in ("Verlet", "RK4", "Yoshida6", "Yoshida8"):
        plot_time_history(
            pick_result(
                fixed_dt_results,
                method_name,
                target_steps_per_orbit,
                "steps_per_orbit",
            ),
            mu,
            period,
        )

    for method_name in ("Verlet", "RK4", "Yoshida6", "Yoshida8"):
        plot_method_error_overplots(
            fixed_dt_results,
            method_name=method_name,
            group_attr="steps_per_orbit",
            group_labeler=format_steps_per_orbit_label,
            mu=mu,
            period=period,
            title_prefix="Error overplots, fixed macro-step size",
        )

    plt.show()


if __name__ == "__main__":
    main()
