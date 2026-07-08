import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def get_step(filename):
    return int(re.search(r"output_(\d+)\.csv", filename).group(1))


files = sorted(glob.glob("output_*.csv"), key=get_step)

if not files:
    raise FileNotFoundError("No output_*.csv files found")

fig, ax = plt.subplots(figsize=(6, 5))

first = pd.read_csv(files[0])
rho0 = first.pivot(index="y", columns="x", values="rho").values

im = ax.imshow(
    rho0,
    origin="lower",
    aspect="equal",
    cmap="viridis",
    vmin=0.98,
    vmax=1.10,
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Density rho")

ax.set_xlabel("x")
ax.set_ylabel("y")
title = ax.set_title(f"Density wave, step {get_step(files[0])}")


def update(i):
    df = pd.read_csv(files[i])
    rho = df.pivot(index="y", columns="x", values="rho").values

    im.set_array(rho)
    title.set_text(f"Density wave, step {get_step(files[i])}")
    return im, title


ani = FuncAnimation(
    fig,
    update,
    frames=len(files),
    interval=250,
    blit=False,
    repeat=True,
)

plt.tight_layout()
ani.save("density_wave.gif", writer="pillow", fps=4)
plt.show()