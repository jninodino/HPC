#!/usr/bin/env python3
"""Run and analyse the shear-wave viscosity experiment reproducibly."""

import argparse
import csv
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analytical_viscosity(omega: float) -> float:
    return (1.0 / 3.0) * (1.0 / omega - 0.5)


def omega_tag(omega: float) -> str:
    return f"{omega:.6g}".replace("-", "m").replace(".", "p")


def run_simulation(executable: Path, omega: float, steps: int, width: int,
                   height: int, epsilon: float, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable.resolve()),
        str(omega),
        str(steps),
        str(width),
        str(height),
        str(epsilon),
        str(output_file),
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def fit_viscosity(filename: Path, omega: float, height: int,
                  fit_start: int, min_fraction: float) -> dict:
    data = np.genfromtxt(filename, delimiter=",", names=True)
    t = np.asarray(data["step"], dtype=float)
    amplitude = np.abs(np.asarray(data["amplitude"], dtype=float))

    if amplitude.size < 3 or amplitude[0] <= 0.0:
        raise RuntimeError(f"Invalid amplitude history in {filename}")

    threshold = max(amplitude[0] * min_fraction, 1e-14)
    mask = (t >= fit_start) & np.isfinite(amplitude) & (amplitude > threshold)

    if np.count_nonzero(mask) < 20:
        raise RuntimeError(
            f"Too few usable fit points for omega={omega}. "
            f"Try more steps or a smaller --min-fraction."
        )

    slope, intercept = np.polyfit(t[mask], np.log(amplitude[mask]), 1)
    if slope >= 0.0:
        raise RuntimeError(
            f"Non-decaying fitted amplitude for omega={omega}: slope={slope}"
        )

    wave_number = 2.0 * np.pi / height
    nu_numeric = -slope / wave_number**2
    nu_analytic = analytical_viscosity(omega)
    relative_error = abs(nu_numeric - nu_analytic) / abs(nu_analytic)

    return {
        "omega": omega,
        "nu_analytic": nu_analytic,
        "nu_numeric": nu_numeric,
        "relative_error": relative_error,
        "slope": slope,
        "intercept": intercept,
        "points_used": int(np.count_nonzero(mask)),
    }


def write_summary(results: list[dict], filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "omega",
        "nu_analytic",
        "nu_numeric",
        "relative_error",
        "slope",
        "points_used",
    ]
    with filename.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def make_plot(results: list[dict], filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)

    omegas = np.array([result["omega"] for result in results])
    numeric = np.array([result["nu_numeric"] for result in results])

    omega_ref = np.linspace(max(0.05, omegas.min() * 0.9),
                            min(1.99, omegas.max() * 1.05), 400)
    nu_ref = analytical_viscosity(omega_ref)

    plt.figure(figsize=(7.0, 4.8))
    plt.plot(omega_ref, nu_ref,
             label=r"Analytical: $\nu=\frac{1}{3}(\frac{1}{\omega}-\frac{1}{2})$")
    plt.scatter(omegas, numeric, marker="x", s=55,
                label="Numerical shear-wave decay")
    plt.xlabel(r"Relaxation parameter $\omega$")
    plt.ylabel(r"Kinematic viscosity $\nu$")
    plt.title("Shear-wave viscosity validation")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run several shear-wave decays and compare measured "
                    "viscosity with the BGK analytical relation."
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=Path("build-cpu/executables/shearWaveDecay"),
        help="Path to the shearWaveDecay executable.",
    )
    parser.add_argument(
        "--omegas",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
        help="Relaxation parameters to evaluate.",
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--fit-start",
        type=int,
        default=10,
        help="Ignore the first N steps when fitting log(amplitude).",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=1e-5,
        help="Ignore amplitudes below this fraction of the initial amplitude.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/shearWaveDecay")
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/shearWaveDecay/viscosity_summary.csv"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("plots/milestone4/shearWaveDecay.png"),
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Only analyse existing per-omega CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 < args.min_fraction < 1.0:
        raise SystemExit("--min-fraction must satisfy 0 < value < 1")

    if not args.no_run and not args.exe.exists():
        raise SystemExit(
            f"Executable not found: {args.exe}\n"
            "Build first with: cmake --build build-cpu -j --target shearWaveDecay"
        )

    results = []
    for omega in args.omegas:
        if not 0.0 < omega < 2.0:
            raise SystemExit(f"omega must satisfy 0 < omega < 2, got {omega}")

        output_file = args.data_dir / f"omega_{omega_tag(omega)}.csv"
        if not args.no_run:
            run_simulation(
                args.exe, omega, args.steps, args.width, args.height,
                args.epsilon, output_file
            )

        result = fit_viscosity(
            output_file, omega, args.height, args.fit_start, args.min_fraction
        )
        results.append(result)

    write_summary(results, args.summary)
    make_plot(results, args.plot)

    print("\nViscosity comparison")
    print("omega    analytic       numeric        rel. error")
    for result in results:
        print(
            f"{result['omega']:>4.1f}   "
            f"{result['nu_analytic']:>12.6g}  "
            f"{result['nu_numeric']:>12.6g}  "
            f"{100.0 * result['relative_error']:>9.3f}%"
        )

    print(f"\nSummary: {args.summary}")
    print(f"Plot:    {args.plot}")


if __name__ == "__main__":
    main()