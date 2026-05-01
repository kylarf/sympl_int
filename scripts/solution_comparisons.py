import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from sympl_int import yoshida6, yoshida8
from sympl_int.yoshida import YOSHIDA6, YOSHIDA8, yoshida
from sympl_int.utils import OrbitalElements, ConservativeForce, FloatNDArray

from comparisons_optimized import (
    reference_orbit,
    kepler_initial_state,
    Integrator,
    two_body_acceleration,
    MethodSpec,
    sweep_fixed_steps_per_orbit,
    save_and_close,
    plot_summary,
    plot_orbit_tiled_by_method,
    plot_time_history,
    pick_result,
    plot_method_error_overplots,
    format_steps_per_orbit_label,
    RunResult
)


def sweep_steps_and_methods_plot(
    prefix: str,
    method_set: tuple[MethodSpec, ...],
    timesteps: list[int],
    a,
    e,
    mu,
    n_orbits,
    save_stepsizes: tuple[int, ...],
):
    def orbit_store_selector(method: MethodSpec, steps_per_orbit: int) -> bool:
        return steps_per_orbit in save_stepsizes

    q0, v0, period = kepler_initial_state(mu=mu, a=a, e=e)
    acceleration = two_body_acceleration(mu)
    r_ref = reference_orbit(mu=mu, a=a, e=e)

    fixed_dt_results = sweep_fixed_steps_per_orbit(
        method_set,
        acceleration,
        q0,
        v0,
        mu,
        period,
        n_orbits,
        timesteps,
        target_saved_per_orbit=1000,
        compute_secondary=False,
        orbit_store_selector=orbit_store_selector,
    )

    save_and_close(
        plot_summary(
            fixed_dt_results,
            x_attr="steps_per_orbit",
            x_label="steps per orbit",
            title_prefix="",
        ),
        f"{prefix}_summary_fixed_dt.svg",
    )

    save_and_close(
        plot_orbit_tiled_by_method(
            fixed_dt_results,
            x_attr="steps_per_orbit",
            target_x_values=(4000.0, 8000.0),
            r_ref=r_ref,
        ),
        f"{prefix}orbit_tiled_fixed_dt.svg",
    )

    target_steps_per_orbit = 8000.0
    for method in method_set:
        method_name = method.name
        save_and_close(
            plot_time_history(
                pick_result(
                    fixed_dt_results,
                    method_name,
                    target_steps_per_orbit,
                    "steps_per_orbit",
                )
            ),
            f"{prefix}_time_history_{method_name.lower()}.svg",
        )

    for method in method_set:
        method_name = method.name
        save_and_close(
            plot_method_error_overplots(
                fixed_dt_results,
                method_name=method_name,
                group_attr="steps_per_orbit",
                group_labeler=format_steps_per_orbit_label,
                title_prefix=method_name,
            ),
            f"{prefix}_overplot_{method_name.lower()}.svg",
        )

    # more zoomed-in look at final 3 stepsizes
    for method in method_set:
        method_name = method.name
        save_and_close(
            plot_method_error_overplots(
                fixed_dt_results,
                method_name=method_name,
                group_attr="steps_per_orbit",
                group_labeler=format_steps_per_orbit_label,
                title_prefix="method_name",
                runs_slice=slice(-3, None),
            ),
            f"{prefix}_overplot_{method_name.lower()}_finest.svg",
        )

    return fixed_dt_results


def get_yoshida_integrator(
    solution_set: np.recarray, variant_letter: str
) -> Integrator:
    def integrator(
        acceleration: ConservativeForce,
        q0: FloatNDArray,
        v0: FloatNDArray,
        tspan: tuple[np.float64, np.float64],
        dt: np.float64,
        subsample: int = 1,
    ) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
        return yoshida(
            acceleration, q0, v0, tspan, dt, solution_set[variant_letter], subsample
        )

    return integrator


def get_coeff_stats(method_coeffs: np.recarray, display=False):
    stats = {}
    header = "solution, max_abs, min, sum_abs"
    print(header)
    if method_coeffs.dtype.names is not None:
        for variant in method_coeffs.dtype.names:
            coeffs = method_coeffs[variant]
            max_abs = np.max(np.abs(coeffs))
            min_coeff = np.min(coeffs)
            mean_abs = np.sum(np.abs(coeffs)) / coeffs.size
            print(f"{variant}, {max_abs}, {min_coeff}, {mean_abs}")
            stats[variant] = {
                "max_abs": max_abs,
                "min": min_coeff,
                "mean_abs": mean_abs,
            }
    return stats


def plot_coeff_stats(coeff_stats, run_results: list[RunResult]):
    max_abs_coeff = []
    min_coeff = []
    mean_abs_coeff = []
    for _, stats in coeff_stats.items():
        max_abs_coeff.append(stats["max_abs"])
        min_coeff.append(stats["min"])
        mean_abs_coeff.append(stats["mean_abs"])

    energy_fig, energy_axes = plt.subplots(1, 3, sharey=True)
    apsis_fig, apsis_axes = plt.subplots(1, 3, sharey=True)

    steps_per_orbit = sorted({round(result.steps_per_orbit) for result in run_results})

    for stepsize in steps_per_orbit:
        steps_label = f"{stepsize} steps/orbit"
        energy_error = [
            result.max_abs_energy_error
            for result in run_results
            if round(result.steps_per_orbit) == stepsize
        ]
        energy_axes[0].scatter(max_abs_coeff, energy_error, label=steps_label)
        energy_axes[0].set_yscale('log')
        energy_axes[1].scatter(min_coeff, energy_error, label=steps_label)
        energy_axes[1].set_yscale('log')
        energy_axes[2].scatter(mean_abs_coeff, energy_error, label=steps_label)
        energy_axes[2].set_yscale('log')

        apsis_drift = [
            result.max_abs_apsis_drift
            for result in run_results
            if round(result.steps_per_orbit) == stepsize
        ]
        apsis_axes[0].scatter(max_abs_coeff, apsis_drift, label=steps_label)
        apsis_axes[0].set_yscale('log')
        apsis_axes[1].scatter(min_coeff, apsis_drift, label=steps_label)
        apsis_axes[1].set_yscale('log')
        apsis_axes[2].scatter(mean_abs_coeff, apsis_drift, label=steps_label)
        apsis_axes[2].set_yscale('log')

    energy_axes[2].legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
    energy_fig.subplots_adjust(right=0.75)

    energy_axes[0].set_xlabel("max |c_i|")
    energy_axes[0].set_ylabel("max |∆E|")
    energy_axes[1].set_xlabel("min c_i")
    energy_axes[2].set_xlabel("mean(|c_i|)")

    apsis_axes[2].legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
    apsis_fig.subplots_adjust(right=0.75)

    apsis_axes[0].set_xlabel("max |c_i|")
    apsis_axes[0].set_ylabel("max |∆ϖ| [rad]")
    apsis_axes[1].set_xlabel("min c_i")
    apsis_axes[2].set_xlabel("mean(|c_i|)")

    return energy_fig, apsis_fig


def main() -> None:
    mu = 1.0
    a = 1.0
    e = 0.9
    n_orbits = 1500

    methods_8: tuple[MethodSpec, ...] = tuple(
        MethodSpec(f"Yoshida8{variant}", get_yoshida_integrator(YOSHIDA8, variant), 15)
        for variant in ("A", "B", "C", "D", "E")
    )
    methods_6: tuple[MethodSpec, ...] = tuple(
        MethodSpec(f"Yoshida6{variant}", get_yoshida_integrator(YOSHIDA6, variant), 7)
        for variant in ("A", "B", "C")
    )

    yoshida8_stats = get_coeff_stats(YOSHIDA8)
    yoshida6_stats = get_coeff_stats(YOSHIDA6)

    steps_per_orbit_list = [1000, 2000, 3000, 4000, 6000, 8000]  # , 16000, 32000]

    results_8 = sweep_steps_and_methods_plot(
        "yoshida8", methods_8, steps_per_orbit_list, a, e, mu, n_orbits, (4000, 8000)
    )

    results_6 = sweep_steps_and_methods_plot(
        "yoshida6", methods_6, steps_per_orbit_list, a, e, mu, n_orbits, (4000, 8000)
    )

    energy_6, apsis_6 = plot_coeff_stats(yoshida6_stats, results_6)
    save_and_close(energy_6, "yoshida6_energy_vs_coeffs.svg")
    save_and_close(apsis_6, "yoshida6_apsis_vs_coeffs.svg")
    energy_8, apsis_8 = plot_coeff_stats(yoshida8_stats, results_8)
    save_and_close(energy_8, "yoshida8_energy_vs_coeffs.svg")
    save_and_close(apsis_8, "yoshida8_apsis_vs_coeffs.svg")


if __name__ == "__main__":
    main()
