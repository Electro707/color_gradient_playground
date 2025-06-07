import numpy as np
from scipy.special import comb
from matplotlib import pyplot as plt

def interpolate(val):
    _type = 'lin'
    _len = 100

    if _type == 'lin':
        interp = np.linspace(0, 1, _len)
    elif _type == 'smoothstep':
        interp = smoothstep(np.arange(_len), 0, _len, args[0])
    else:
        raise UserWarning(f"Invalid type {_type:s}")

    delta = val[1] - val[0]
    if 1 > abs(delta) > 0.5:
        if delta < 0:
            interp *= 1-abs(delta)
        else:
            interp *= delta-1
    else:
        interp *= delta
    interp += val[0]

    print(delta)

    interp = np.where((interp > 1), interp - 1, interp)
    interp = np.where((interp < 0), interp + 1, interp)

    return interp

def test_exp():
    _len = 100

    x = np.arange(_len)
    return x**2/(_len**2)


if __name__ == "__main__":
    fig, ax = plt.subplots()

    #ax.plot(interpolate((0.9, 0.5)))
    #ax.plot(interpolate((1, 0)))
    #ax.plot(interpolate((0.9, 0.1)))
    #ax.plot(interpolate((0, 0.6667)))
    #ax.plot(interpolate((0.0627451, 0.03921569)))
    #ax.plot(interpolate((0.18039216, 0.67843137)))
    #ax.plot(interpolate((0.92941176, 0.08235294)))

    ax.plot(test_exp())
    ax.set_ylim(0, 1)
    plt.show()

    #z = np.linspace(0, 1, 100)
    #a = smoothstep(z, N=10)
    #plt.plot(z, a)
    #plt.show()
