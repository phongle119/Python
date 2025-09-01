import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Use the ipympl backend for interactive plots in VS Code/Jupyter
plt.close('all')
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')

# plt.rcParams["animation.html"] = "jshtml"
plt.ioff()
def init():
    # data, labels = generate_data([2,200])
    # ax = plot_data(data,labels)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1.1, 1.1)
    return line,

def update(frame):
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x + frame * 0.1)
    line.set_data(x, y)
    return line,

# Assign the animation object to a variable to prevent garbage collection
ani = animation.FuncAnimation(fig, update, frames=50, init_func=init, blit=True, interval=20, repeat=False)
plt.show() # Display the animation
