import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

filename = "observed_point.bin"

with open(filename, "rb") as f:
    # Header written by C++: [steps]
    steps, dim = np.frombuffer(f.read(8), dtype=np.int32)
    raw = np.frombuffer(f.read(), dtype=np.float64)

print(f"steps: {steps}")
data = raw.reshape(steps, dim)


# data = np.clip(data, -100, 100)

x_ax = np.arange(steps)

fig, ax = plt.subplots(3, 3)
for i in range(9):
    a = i // 3
    b = i % 3 
    ax[a][b].plot(x_ax, data[:, i])
    ax[a][b].set_title(f"i = {i}")
plt.show()

p = np.zeros(steps)
for s in range(steps):
    p[s] += np.sum(data[s])
plt.plot(x_ax, p)
plt.show()
