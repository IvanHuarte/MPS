import numpy as np


def TFMOhmicModel(Nk, wc, g, **kwargs):
    # g es alpha

    delta = kwargs["delta"]
    alpha = g

    dk = wc / Nk
    k = np.linspace(0, wc, Nk)
    wk = abs(k)

    gk = np.sqrt(alpha * delta * dk / 2) * np.sqrt(wk)

    return (k, wk, gk, alpha)


def OhmicModel(Nk, wc, g, **kwargs):
    """For 1 atom and v_group = 1"""

    k = np.linspace(-10 * wc, 10 * wc, Nk)
    wk = np.abs(k)

    gk = g * np.exp(-wk / (2 * wc)) * np.sqrt(wk / 2)
    alpha = g**2 / np.pi

    return (k, wk, gk, alpha)


def SubSubOhmicModel(Nk, wc, g, **kwargs):

    w0 = kwargs["w0"]

    k = np.linspace(-np.pi, np.pi, Nk)
    wk = w0 * wc / np.sqrt(w0**2 + 2 * wc**2 * (1 - np.cos(k)))

    gk = np.array([g / np.sqrt(Nk)])
    gk = np.broadcast_to(gk, (Nk,))

    alpha = 2 * g**2

    return (k, wk, gk, alpha)

def SubSubOhmic32Model(Nk, wc, g, **kwargs):

    w0 = kwargs["w0"]

    k = np.linspace(-np.pi, np.pi, Nk)
    wk = w0 * wc / np.sqrt(w0**2 + 2 * wc**2 * (1 - np.cos(k)))

    gk = g * (wk/w0)**(3/2) / np.sqrt(Nk)
    gk = np.broadcast_to(gk, (Nk,))

    alpha = 2 * g**2

    return (k, wk, gk, alpha)
