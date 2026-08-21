import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load velocity field
filename_velocity = f"data/lidDrivenCavity_velocity.bin"

with open(filename_velocity, "rb") as f:
    # Header written by C++: [steps, width, height]
    steps, width, height, dim = np.frombuffer(f.read(16), dtype=np.int32)
    raw = np.frombuffer(f.read(), dtype=np.float64)

slice_size = width * height * dim
steps = raw.size // slice_size
data = raw.reshape(steps, width, height, dim)

# cut off ghost layers
data_velocity = data[:, 1:width-1, :]

# Generate x
x = np.linspace(0, 1, width-2)
y = np.linspace(0, 1, height)

timestep = -1
vx = data_velocity[timestep, :, :, 0]
vy = data_velocity[timestep, :, :, 1]
speed = np.sqrt(vx**2 + vy**2)

# streamplot expects U, V with shape (ny, nx)
U = vx.T
V = vy.T

fig, ax = plt.subplots(figsize=(6, 5.5))

strm = ax.streamplot(x, y, U, V, color=speed.T, cmap="viridis",
                     linewidth=1, arrowsize=1.2, density=1.5)

# Arrow
ax.annotate(
    "", xy=(0.9, 1.08), xytext=(0.1, 1.08),
    xycoords="data",
    arrowprops=dict(arrowstyle="->", color="red", lw=2),
)
ax.text(0.5, 1.14, r"$u_{\mathrm{lid}}$", color="red",
        ha="center", va="bottom", fontsize=12)


cbar = fig.colorbar(strm.lines, ax=ax, pad=0.02)
cbar.set_label("Velocity magnitude |u| (lattice units)")
ax.set_xlabel("x/L")
ax.set_ylabel("y/L")


plt.tight_layout()
plt.savefig(r"plots/milestone5/lidDrivenCavity.png")