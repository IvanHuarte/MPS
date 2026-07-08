import numpy as np


def TFMOhmicModel(Nk, wc, w0, g):
    s = 1  # ohmico
    dk = 2 * np.pi / Nk

    k = np.linspace(-wc, wc, Nk)
    wk = abs(k)

    alpha = 2 * g**2 / np.pi
    alpha = 2 * g**2 / (dk * wc ** (1 - s))
    gk = g * np.sqrt(wk)

    return (k, wk, gk, alpha)


def OhmicModel(Nk, wc, w0, g):
    """For 1 atom and v_group = 1"""

    k = np.linspace(-10 * wc, 10 * wc, Nk)
    wk = np.abs(k)

    gk = g * np.exp(-wk / (2 * wc)) * np.sqrt(wk / 2)
    alpha = g**2 / np.pi

    return (k, wk, gk, alpha)


def SubSubOhmicModel(Nk, wc, w0, g):

    k = np.linspace(-np.pi, np.pi, Nk)
    wk = w0 * wc / np.sqrt(w0**2 + 2 * wc**2 * (1 - np.cos(k)))

    gk = np.array([g / np.sqrt(Nk)])
    gk = np.broadcast_to(gk, (Nk,))

    alpha = 2 * g**2

    return (k, wk, gk, alpha)
