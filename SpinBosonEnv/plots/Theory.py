import numpy as np
import scipy


def sz_subsubohmic(gvals, w0, wc, delta):
    print("Theoretical expression for Sz uses 'g' as coupling strength")

    def wk_pos(k, ω0, ωc):
        # Subsubohmic
        wk = ω0 * ωc / np.sqrt(ω0**2 + 2 * ωc**2 * (1 - np.cos(k)))

        return wk

    def I3_numeric(ω0, ωc, Nk=200000):
        k = np.linspace(0, np.pi, Nk)
        wk = wk_pos(k, ω0, ωc)
        return np.trapezoid(1 / wk**3, k)

    def deltar_lambert(delta, g0, ω0, ωc, I3):
        # if g0 == 0:
        #     return delta

        A0 = 2 * g0**2 * (1 / ωc**2 + 2 / ω0**2)
        B = (4 * g0**2 / np.pi) * I3
        z = -B * delta * np.exp(-A0)

        return np.real(-(1 / B) * scipy.special.lambertw(z))

    B = I3_numeric(w0, wc)
    W0 = deltar_lambert(delta, gvals, w0, wc, B)
    Szteo = -0.5 * (1 / delta) * W0

    return Szteo


def sz_ohmic(alphavals, wc, delta):
    print("Theoretical expression for Sz uses 'alpha' as coupling strength")

    delta_r = delta * (delta / wc) ** (alphavals / (1 - alphavals))
    Szteo = -0.5 * delta_r / delta

    return Szteo
