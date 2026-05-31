import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

filename = "output.bin"

with open(filename, "rb") as f:
    # Header written by C++: [steps, width, height]
    steps, width, height = np.frombuffer(f.read(12), dtype=np.int32)
    raw = np.frombuffer(f.read(), dtype=np.int32)

expected_size = steps * width * height
slice_size = width * height
if raw.size % slice_size != 0:
    raise ValueError(
        f"Data size {raw.size} is not divisible by one slice size {slice_size}"
    )

available_steps = raw.size // slice_size
if raw.size != expected_size:
    print(
        f"Warning: header says steps={steps}, but file currently contains {available_steps} step(s)."
    )

data = raw.reshape(available_steps, width, height)
steps = available_steps

fig, ax = plt.subplots()
im = ax.imshow(data[0].T, origin="lower", animated=True)
ax.set_title(f"Density map - step 0/{steps - 1}")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(im, ax=ax, label="density")


def update(frame):
    im.set_array(data[frame].T)
    ax.set_title(f"Density map - step {frame}/{steps - 1}")
    return (im,)


ani = FuncAnimation(fig, update, frames=steps, interval=150, blit=True, repeat=True)

plt.tight_layout()
plt.show()
