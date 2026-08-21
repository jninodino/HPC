import numpy as np
import matplotlib.pyplot as plt

filename = "data/streaming_velocity.bin"

with open(filename, "rb") as f:
    # Header written by C++: [steps, width, height, dim=2]
    steps, width, height, dim = np.frombuffer(f.read(16), dtype=np.int32)
    raw = np.frombuffer(f.read(), dtype=np.float64)

slice_size = width * height * dim
available_steps = raw.size // slice_size
data = raw.reshape(available_steps, width, height, dim)

# Use the last step
vx = data[-1, :, :, 0]  # shape (width, height)
vy = data[-1, :, :, 1]

speed = np.sqrt(vx**2 + vy**2)

x = np.linspace(0, 1, width)
y = np.linspace(0, 1, height)

# streamplot expects U, V with shape (ny, nx)
U = vx.T
V = vy.T

fig, ax = plt.subplots()

strm = ax.streamplot(
    x, y, U, V,
    color=speed.T,
    cmap="viridis",
    linewidth=1,
    arrowsize=1.2,
    density=1.5,
)

fig.colorbar(strm.lines, ax=ax, label="Velocity magnitude |u| (lattice units)")

ax.set_xlabel("x/L")
ax.set_ylabel("y/L")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Lid arrow above the top boundary
ax.annotate(
    "",
    xy=(0.72, 1.06), xytext=(0.28, 1.06),
    xycoords="axes fraction", textcoords="axes fraction",
    arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5),
    annotation_clip=False,
)
ax.text(0.5, 1.09, r"$u_\mathrm{lid}$", ha="center", va="bottom",
        transform=ax.transAxes, color="red", fontsize=11)

plt.tight_layout()
plt.show()