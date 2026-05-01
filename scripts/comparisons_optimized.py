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
        true_anomaly=np.array(0.0, dtype=np.float64),
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


def sparsify_xy(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int = 100000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample a time series for plotting while preserving local extrema.
    Returns up to ~max_points points, typically in min/max pairs.
    """
    n = x.size
    if n <= max_points:
        return x, y

    n_bins = max(1, max_points // 2)
    edges = np.linspace(0, n, n_bins + 1, dtype=np.int64)

    x_out = np.empty(2 * n_bins, dtype=x.dtype)
    y_out = np.empty(2 * n_bins, dtype=y.dtype)
    k = 0

    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if hi <= lo:
            continue

        x_bin = x[lo:hi]
        y_bin = y[lo:hi]

        j_min = np.argmin(y_bin)
        j_max = np.argmax(y_bin)

        if j_min == j_max:
            x_out[k] = x_bin[j_min]
            y_out[k] = y_bin[j_min]
            k += 1
        elif j_min < j_max:
            x_out[k] = x_bin[j_min]
            y_out[k] = y_bin[j_min]
            k += 1
            x_out[k] = x_bin[j_max]
            y_out[k] = y_bin[j_max]
            k += 1
        else:
            x_out[k] = x_bin[j_max]
            y_out[k] = y_bin[j_max]
            k += 1
            x_out[k] = x_bin[j_min]
            y_out[k] = y_bin[j_min]
            k += 1

    return x_out[:k], y_out[:k]


def sparsify_orbit(
    r: np.ndarray,
    max_points: int = 20000,
) -> np.ndarray:
    n = r.shape[0]
    if n <= max_points:
        return r
    idx = np.linspace(0, n - 1, max_points, dtype=np.int64)
    return r[idx]


@dataclass
class RunResult:
    method: str
    dt: float
    steps_per_orbit: float
    force_evals_per_orbit: float
    n_orbits: int
    subsample: int

    orbit_count: NDArray[np.float32]
    energy_error: NDArray[np.float32]
    apsis_drift: NDArray[np.float32]

    max_abs_energy_error: float
    final_apsis_drift: float
    max_abs_apsis_drift: float

    max_abs_h_error: float | None = None
    max_abs_a_error: float | None = None
    max_abs_e_error: float | None = None

    r_plot: NDArray[np.float32] | None = None


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
    *,
    compute_secondary: bool = False,
    store_orbit: bool = False,
) -> RunResult:
    energy = specific_energy(mu, r, v)
    energy_error = energy - energy[0]

    e_vec = eccentricity_vector(mu, r, v)
    apsis = unwrap_planar_apsides(e_vec)
    apsis_drift = apsis - apsis[0]

    orbit_count = t / period

    max_abs_h_error = None
    max_abs_a_error = None
    max_abs_e_error = None

    if compute_secondary:
        h_vec = specific_angmom(r, v)
        h_mag = np.linalg.norm(h_vec, axis=-1)
        h_error = h_mag - h_mag[0]

        oe = OrbitalElements.from_rv(r, v, np.float64(mu))
        max_abs_h_error = float(np.max(np.abs(h_error)))
        max_abs_a_error = float(np.max(np.abs(oe.semimajor_axis - oe.semimajor_axis[0])))
        max_abs_e_error = float(np.max(np.abs(oe.eccentricity - oe.eccentricity[0])))

    return RunResult(
        method=method.name,
        dt=dt,
        steps_per_orbit=period / dt,
        force_evals_per_orbit=(period / dt) * method.force_evals_per_step,
        n_orbits=n_orbits,
        subsample=subsample,
        orbit_count=orbit_count.astype(np.float32),
        energy_error=energy_error.astype(np.float32),
        apsis_drift=apsis_drift.astype(np.float32),
        max_abs_energy_error=float(np.max(np.abs(energy_error))),
        final_apsis_drift=float(apsis_drift[-1]),
        max_abs_apsis_drift=float(np.max(np.abs(apsis_drift))),
        max_abs_h_error=max_abs_h_error,
        max_abs_a_error=max_abs_a_error,
        max_abs_e_error=max_abs_e_error,
        r_plot=sparsify_orbit(r).astype(np.float32) if store_orbit else None,
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
    *,
    compute_secondary: bool = False,
    store_orbit: bool = False,
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
    result = analyze_run(
        method,
        mu,
        period,
        n_orbits,
        dt,
        subsample,
        t,
        r,
        v,
        compute_secondary=compute_secondary,
        store_orbit=store_orbit,
    )
    del t, r, v
    return result


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
    *,
    compute_secondary: bool = False,
    orbit_store_selector: Callable[[MethodSpec, int], bool] | None = None,
) -> list[RunResult]:
    results: list[RunResult] = []
    for steps_per_orbit in steps_per_orbit_list:
        dt = period / steps_per_orbit
        subsample = choose_subsample(steps_per_orbit, target_saved_per_orbit)
        for method in methods:
            print(f"Running {method.name} with {steps_per_orbit=}, {subsample=}")
            store_orbit = orbit_store_selector(method, steps_per_orbit) if orbit_store_selector else False
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
                    compute_secondary=compute_secondary,
                    store_orbit=store_orbit,
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
        y = getattr(result, y_attr)
        if y is None:
            continue
        grouped.setdefault(result.method, []).append((getattr(result, x_attr), y))

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method, pairs in grouped.items():
        pairs_sorted = sorted(pairs, key=lambda p: p[0])
        out[method] = (
            np.array([p[0] for p in pairs_sorted], dtype=np.float64),
            np.array([p[1] for p in pairs_sorted], dtype=np.float64),
        )
    return out


def plot_summary(
    results: list[RunResult],
    x_attr: str,
    x_label: str,
    title_prefix: str,
):
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    energy_series = extract_metric_series(results, x_attr, "max_abs_energy_error")
    apsis_series = extract_metric_series(results, x_attr, "final_apsis_drift")

    for method, (x, y) in energy_series.items():
        axes[0].semilogy(x, np.abs(y), marker="o", label=method)
    axes[0].set_ylabel("max |ΔE|")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title(title_prefix)

    for method, (x, y) in apsis_series.items():
        axes[1].semilogy(x, np.abs(y), marker="o", label=method)
    axes[1].set_ylabel("|final Δϖ| [rad]")
    axes[1].set_xlabel(x_label)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_secondary_summary(
    results: list[RunResult],
    x_attr: str,
    x_label: str,
    title_prefix: str,
):
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
            ax.semilogy(x, np.abs(y), marker="o", label=method)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(title_prefix)
    axes[-1].set_xlabel(x_label)
    axes[0].legend()
    fig.tight_layout()
    return fig


def pick_result(
    results: list[RunResult],
    method_name: str,
    target_x: float,
    x_attr: str,
) -> RunResult:
    candidates = [r for r in results if r.method == method_name]
    return min(candidates, key=lambda r: abs(getattr(r, x_attr) - target_x))


def plot_time_history(result: RunResult):
    x_e, y_e = sparsify_xy(result.orbit_count, result.energy_error)
    x_w, y_w = sparsify_xy(result.orbit_count, result.apsis_drift)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(x_e, y_e)
    axes[0].set_ylabel("ΔE")
    axes[0].set_title(
        f"{result.method}: dt={result.dt:.3e}, "
        f"steps/orbit={result.steps_per_orbit:.1f}"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_w, y_w)
    axes[1].set_ylabel("Δϖ [rad]")
    axes[1].set_xlabel("orbit count")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_orbit_tiled_by_method(
    results: list[RunResult],
    x_attr: str,
    target_x_values: tuple[float, ...],
    r_ref: FloatNDArray,
):
    method_names = sorted({r.method for r in results})

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, method_name in zip(axes, method_names):
        ax.plot(r_ref[:, 0], r_ref[:, 1], linestyle="--", alpha=0.6, label="reference")

        for target_x in target_x_values:
            result = pick_result(results, method_name, target_x, x_attr)
            if result.r_plot is None:
                continue

            ax.plot(
                result.r_plot[:, 0],
                result.r_plot[:, 1],
                label=f"{x_attr}={getattr(result, x_attr):.0f}",
            )

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_title(method_name)
        ax.legend()

    for ax in axes[len(method_names):]:
        ax.axis("off")

    fig.supxlabel("x")
    fig.supylabel("y")
    fig.tight_layout()
    return fig


def format_steps_per_orbit_label(result: RunResult) -> str:
    return f"{result.steps_per_orbit:.0f} steps/orbit"


def plot_method_error_overplots(
    results: list[RunResult],
    method_name: str,
    group_attr: str,
    group_labeler: Callable[[RunResult], str],
    title_prefix: str,
    runs_slice: slice = slice(None, None, None),
):
    method_results = [r for r in results if r.method == method_name]
    method_results = sorted(method_results, key=lambda r: getattr(r, group_attr))

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for result in method_results[runs_slice]:
        x_e, y_e = sparsify_xy(result.orbit_count, result.energy_error, max_points=100000)
        x_w, y_w = sparsify_xy(result.orbit_count, result.apsis_drift, max_points=100000)
        label = group_labeler(result)

        axes[0].plot(x_e, np.abs(y_e), label=label)
        axes[1].plot(x_w, np.abs(y_w), label=label)

    axes[0].set_ylabel("|ΔE|")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"{method_name}")

    axes[1].set_ylabel("|Δϖ| [rad]")
    axes[1].set_xlabel("orbit count")
    axes[1].grid(True, alpha=0.3)

    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    return fig


def save_and_close(fig, path: str, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    mu = 1.0
    a = 1.0
    e = 0.9
    n_orbits = 5000

    q0, v0, period = kepler_initial_state(mu=mu, a=a, e=e)
    acceleration = two_body_acceleration(mu)
    r_ref = reference_orbit(mu=mu, a=a, e=e)

    def orbit_store_selector(method: MethodSpec, steps_per_orbit: int) -> bool:
        return steps_per_orbit in (4000, 8000)

    steps_per_orbit_list = [1000, 2000, 3000, 4000, 6000, 8000]# , 16000, 32000]
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
        compute_secondary=False,   # turn on only if you really need it
        orbit_store_selector=orbit_store_selector,
    )

    save_and_close(
        plot_summary(
            fixed_dt_results,
            x_attr="steps_per_orbit",
            x_label="steps per orbit",
            title_prefix="",
        ),
        "summary_fixed_dt.svg",
    )

    save_and_close(
        plot_orbit_tiled_by_method(
            fixed_dt_results,
            x_attr="steps_per_orbit",
            target_x_values=(4000.0, 8000.0),
            r_ref=r_ref,
        ),
        "orbit_tiled_fixed_dt.svg",
    )

    target_steps_per_orbit = 8000.0
    for method_name in ("Verlet", "RK4", "Yoshida6", "Yoshida8"):
        save_and_close(
            plot_time_history(
                pick_result(
                    fixed_dt_results,
                    method_name,
                    target_steps_per_orbit,
                    "steps_per_orbit",
                )
            ),
            f"time_history_{method_name.lower()}.svg",
        )

    for method_name in ("Verlet", "RK4", "Yoshida6", "Yoshida8"):
        save_and_close(
            plot_method_error_overplots(
                fixed_dt_results,
                method_name=method_name,
                group_attr="steps_per_orbit",
                group_labeler=format_steps_per_orbit_label,
                title_prefix="",
            ),
            f"overplot_{method_name.lower()}.svg",
        )

    # more zoomed-in look at final 3 stepsizes
    for method_name in ("Verlet", "RK4", "Yoshida6", "Yoshida8"):
        save_and_close(
            plot_method_error_overplots(
                fixed_dt_results,
                method_name=method_name,
                group_attr="steps_per_orbit",
                group_labeler=format_steps_per_orbit_label,
                title_prefix="",
                runs_slice=slice(-3, None)
            ),
            f"overplot_{method_name.lower()}_finest.svg",
        )


if __name__ == "__main__":
    main()
