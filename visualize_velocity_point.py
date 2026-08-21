import numpy as np
import matplotlib.pyplot as plt

filename = "observed_velocity.bin"

with open(filename, "rb") as f:
    steps = np.frombuffer(f.read(4), dtype=np.int32)
    raw = np.frombuffer(f.read(), dtype=np.float64)

data = raw.reshape(steps)

x_ax = np.arange(steps[0])

plt.plot(x_ax, data)
plt.title("u_x at point Lx/4 and Ly/4")
plt.xlabel("timestep (delta_t=1)")
plt.ylabel("u_x (1)")
plt.savefig("plots/milestone4_short.png")
plt.show()