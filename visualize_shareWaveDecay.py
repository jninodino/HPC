import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

omega_dic = {0.2: "02", 0.4: "04", 0.6: "06", 0.8: "08", 1.: "10", 1.2: "12",
             1.4: "14", 1.6: "16", 1.8: "18"}

def analyze_slope(omega: float):
    # Load velocity field
    filename_velocity = f"data/shareWaveDecay_velocity_{omega_dic[omega]}.bin"

    with open(filename_velocity, "rb") as f:
        # Header written by C++: [steps, width, height]
        steps, width, height, dim = np.frombuffer(f.read(16), dtype=np.int32)
        raw = np.frombuffer(f.read(), dtype=np.float64)

    slice_size = width * height * dim
    steps = raw.size // slice_size
    data = raw.reshape(steps, width, height, dim)

    # cut off ghost layers
    data_velocity = data[:, 1:width-1, :]

    print(data_velocity.shape)

    x = 0
    a_sim = np.zeros((data_velocity.shape[0]))
    for t in range(data_velocity.shape[0]):
        for j in range(height):
            a_sim[t] +=  data_velocity[t, x, j, 0] * np.sin(2 * np.pi / height * j)
            a_sim[t] = 2 / height * a_sim[t]

    a_sim_log = np.log(a_sim)

    t = np.arange(steps)

    slope, intercept = np.polyfit(t, a_sim_log, 1)

    return height, slope

omegas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
vs_sim = []
for omega in omegas:
    height, m = analyze_slope(omega)
    v_sim = -m / (2 * np.pi / height) ** 2
    vs_sim += [v_sim]

omega_ref = np.linspace(0, 2, 100)
vs_ref = (1 / 3) * ((1 / omega_ref) - 0.5)
plt.plot(omega_ref, vs_ref, label="reference", color="b")
plt.scatter(omegas, vs_sim, label="simulated", marker='x', color="r" \
"")
plt.title("Shear-wave decay: ln(amplitude) vs time")
plt.xlabel("omega")
plt.ylabel("v")
plt.legend()
plt.savefig("plots/milestone4/shareWaveDecay.png")
plt.show()
