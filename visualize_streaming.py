import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Load density map
filename_density = "data/streaming_density.bin"

with open(filename_density, "rb") as f:
    # Header written by C++: [steps, width, height]
    steps, width, height = np.frombuffer(f.read(12), dtype=np.int32)
    raw = np.frombuffer(f.read(), dtype=np.float64)

slice_size = width * height
steps = raw.size // slice_size
data = raw.reshape(steps, width, height)

# cut off ghost layers
data_density = data[:, 1:width-1, :]


# Show density map
plot_density = True
if plot_density:
    fig, ax = plt.subplots()
    im = ax.imshow(data_density[0].T, origin="lower", animated=True, cmap="Blues")
    ax.set_title(f"Density map - step 0/{steps - 1}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="density")


    def update(frame):
        im.set_array(data_density[frame].T)
        ax.set_title(f"Density map - time step {frame}/{steps - 1}")
        return (im, ax.title)

    ani = FuncAnimation(fig, update, frames=steps, interval=200, blit=False, repeat=True)

    plt.tight_layout()
    # plt.show()
    ani.save("plots/milestone2/streaming_y.gif", writer="pillow", fps=5)


