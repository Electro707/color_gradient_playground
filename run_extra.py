import run
from matplotlib import pyplot as plt

def plot_interpolation():
    fig, ax = plt.subplots(figsize=(3*3, 2*3))

    r = run.interpolate((0, 1.0), 1000, 'lin')
    ax.plot(r, label='Linear')
    r = run.interpolate((0, 1.0), 1000, 'smoothstep', 1)
    ax.plot(r, label='Smoothstep 1')
    r = run.interpolate((0, 1.0), 1000, 'smoothstep', 2)
    ax.plot(r, label='Smoothstep 2')
    r = run.interpolate((0, 1.0), 1000, 'smoothstep', 10)
    ax.plot(r, label='Smoothstep 10')
    r = run.interpolate((0, 1.0), 1000, 'quad')
    ax.plot(r, label='Quad')
    r = run.interpolate((0, 1.0), 1000, 'sin')
    ax.plot(r, label='Sin')

    ax.legend()
    ax.set_title("Interpolation Method")

    fig.tight_layout()
    plt.show()
    # plt.savefig("interp_methods.png")

def plot_interpolation_loop():
    fig, ax = plt.subplots(figsize=(3*3, 2*3))

    r = run.interpolate((0, 1.0), 1000, 'loop_lin')
    ax.plot(r, label='Linear')
    r = run.interpolate((0, 1.0), 1000, 'loop_smoothstep', 1)
    ax.plot(r, label='Smoothstep 1')
    r = run.interpolate((0, 1.0), 1000, 'loop_smoothstep', 2)
    ax.plot(r, label='Smoothstep 2')
    r = run.interpolate((0, 1.0), 1000, 'loop_smoothstep', 10)
    ax.plot(r, label='Smoothstep 10')
    r = run.interpolate((0, 1.0), 1000, 'loop_sin')
    ax.plot(r, label='Sin')

    ax.legend()
    ax.set_title("Interpolation Method, Looping")

    fig.tight_layout()
    # plt.show()
    plt.savefig("interp_methods_loop.png")

if __name__ == "__main__":
    # plot_interpolation()
    plot_interpolation_loop()
