import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

nx, ny = 15, 10
step = range(31)

fig, ax = plt.subplots()

def update(step):
    ax.clear()
    df = pd.read_csv(f"output_{step}.csv")

    X = df["x"].values.reshape(nx, ny)
    Y = df["y"].values.reshape(nx, ny)
    U = df["ux"].values.reshape(nx, ny)
    V = df["uy"].values.reshape(nx, ny)

    ax.quiver(X, Y, U, V)
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_title(f"Velocity field, step{step}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

ani = FuncAnimation(fig, update, frames=steps, interval=300)
ani.save("velocity_animation.gif", writer="pillow")