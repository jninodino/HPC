import numpy as np
import matplotlib.pyplot as plt

# Settings
csv_file = "data/shearWaveDecay/omega_1p0.csv"
Ly = 100

# Load measured amplitudes
data = np.loadtxt(csv_file, delimiter=",", skiprows=1)

t = data[:, 0]
A = data[:, 1]

# Spatial coordinate
y = np.linspace(0.0, 1.0, Ly)

# Choose a few representative time steps
steps_to_plot = [0, 100, 250, 500]

plt.figure(figsize=(8, 5))

for target_step in steps_to_plot:
    idx = np.argmin(np.abs(t - target_step))

    amplitude = A[idx]

    u = amplitude * np.sin(2.0 * np.pi * y)

    plt.plot(
        y,
        u,
        linewidth=2,
        label=fr"$t={int(t[idx])}$"
    )

plt.axhline(0.0, linewidth=0.8)

plt.xlabel(r"$y/L_y$")
plt.ylabel(r"$u_x(y,t)$")
plt.title("Shear-wave decay")

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "plots/milestone4/shearWaveDecay_profiles.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()