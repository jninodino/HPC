import numpy as np
import matplotlib.pyplot as plt
import os

filename = "viscosity_meas.bin"

with open(filename, "rb") as f:
    steps = np.frombuffer(f.read(4), dtype=np.int32)
    raw = np.frombuffer(f.read(), dtype=np.float64)

data = raw.reshape(steps)

x_ax = np.arange(steps[0])

# Fit ln(a_sim) = ln(epsilon) - nu * k^2 * t  =>  straight line
log_a = np.log(data)
coeffs = np.polyfit(x_ax, log_a, 1)   # coeffs[0]=slope, coeffs[1]=intercept
slope = coeffs[0]

# Extract viscosity: slope = -nu * k^2,  k = 2*pi / N_y
N_y = 100  # your grid height
k = 2 * np.pi / N_y
nu_measured = -slope / k**2

print(f"Measured kinematic viscosity: nu = {nu_measured:.6f}")

for w in [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]:
    v = (1/3) * ((1/w) - 0.5)
    print(f"w: {w}, v: {v}")

# Plot
fit_line = np.polyval(coeffs, x_ax)

os.makedirs("plots", exist_ok=True)
plt.plot(x_ax, log_a, label="ln(a_sim)")
plt.plot(x_ax, fit_line, "--", label=f"fit: slope={slope:.5f}, ν={nu_measured:.4f}")
plt.title("Shear-wave decay: ln(amplitude) vs time")
plt.xlabel("timestep")
plt.ylabel("ln(û_x)")
plt.legend()
plt.savefig("plots/milestone4_fit.png")
plt.show()

