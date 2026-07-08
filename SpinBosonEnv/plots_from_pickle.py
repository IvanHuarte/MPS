import os
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import scipy

plt.rcParams["text.usetex"] = True

def plot_from_pickle(pickle_path, plot_field=True):

    sim_label = Path(pickle_path).name.split(".pkl")[0].split("Main_results_")[1]
    new_folder = "Figures_" + sim_label
    folder = str(Path(pickle_path).parent)
    write_folder = folder + "/" + new_folder + "/"
    os.makedirs(write_folder, exist_ok=True)

    with open(pickle_path, "rb") as f:
        data = pkl.load(f)
    # Saved as: w0 | wc | Nk | g | E_gr | Sx | Sy | Sz | S_bond | alpha | delta

    # OCUPPATION VS SITES

    w0 = int(data[:, 0][0])
    wc = int(data[:, 1][0])
    Nk = int(data[:, 2][0])
    delta = float(data[:, 10][0])
    gvals = data[:, 3][1::5]

    nprime = wc / w0
    wmin = w0 * wc / np.sqrt(w0**2 + 4 * wc**2)

    if plot_field:

        fig1, ax1 = plt.subplots(figsize=[10, 8])
        ax1.set_title(
            r"$Ground\;state\;photons \quad (n^\star=%.2f)$" % (nprime), fontsize=18
        )
        ax1.set_xlabel(r"$n/n^\star$", fontsize=18)
        ax1.set_ylabel(r"$\langle a_n^\dagger a_n \rangle_{GS}$", fontsize=18)
        ax1.set_ylim(1e-20, 1e1)
        ax1.set_xlim(1 / nprime - 0.001, Nk // 2 / nprime)

        ax1.text(
            1,
            0.1,
            r"$n^\star = \frac{\omega_c}{\omega_0}$",
            transform=transforms.blended_transform_factory(
                ax1.transData, ax1.transAxes
            ),
            fontsize=12,
        )
        ax1.text(
            0.4,
            0.9,
            (
                rf"$\Delta = {delta:.3f}$"
                "\n"
                rf"$\omega_c = {wc:.3f}$"
                "\n"
                rf"$\omega_{{\min}} = {wmin:.3f}$"
            ),
            transform=transforms.blended_transform_factory(
                ax1.transAxes, ax1.transAxes
            ),
            fontsize=12,
        )
        cmap = plt.colormaps["inferno"](np.linspace(0, 1, len(gvals)))

        for i, g in enumerate(gvals):
            Nx = np.loadtxt(
                folder + "/GroundState_wc_%.4f_g_%.4f_delta_%.4f_NoccX.txt" % (wc, g, delta),
                dtype=complex,
            ).real
            Nxhalf = Nx[Nk // 2 :]
            n_values = np.arange(len(Nxhalf))

            # TEORIC
            Nteo = (
                g**2
                / (np.pi**2 * w0**2 * nprime**2)
                * (scipy.special.k1(n_values / nprime) / n_values) ** 2
            )
            ax1.plot(np.arange(0, len(n_values)) / nprime, Nteo, color=cmap[i])

            # MPS RESULTS
            ax1.plot(
                np.arange(len(Nxhalf)) / nprime,
                Nxhalf,
                color=cmap[i],
                marker="o",
                ls="",
                ms=3,
                label=r"$g=%.2f$" % g,
            )

        ax1.vlines(1, 1e-20, 1e1, color="grey", ls="--")
        ax1.legend()
        ax1.set_yscale("log")
        ax1.set_xscale("log")
        ax1.grid()
        fig1.savefig(
            write_folder + f"Nx_RealSpace_occupation_{sim_label}.pdf",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig1)

    # ENERGY
    gvals = data[:, 3]
    E_gr = data[:, 4]

    fig2, ax2 = plt.subplots(figsize=[10, 8])
    ax2.set_title(r"$Energy\;vs.\;g \quad (n^\star=%.2f)$" % (nprime), fontsize=18)
    ax2.set_xlabel(r"$g$", fontsize=18)
    ax2.set_ylabel(r"$\langle H \rangle_{GS}$", fontsize=18)
    ax2.set_ylim(min(E_gr) + min(E_gr) / 10, max(E_gr) - max(E_gr))
    ax2.grid()
    ax2.plot(gvals, E_gr)

    # ENTROPY
    w0 = data[:, 0][0]
    wc = data[:, 1][0]
    Nk = data[:, 2][0]
    gvals = data[:, 3]
    Sbond = data[:, 8]

    fig3, ax3 = plt.subplots(figsize=[10, 8])
    ax3.set_title(
        r"$Cavity-Atom\;Entropy  \quad (n^\star=%.2f)$" % (nprime), fontsize=18
    )
    ax3.set_xlabel(r"$g$", fontsize=18)
    ax3.set_ylabel(r"$\langle S_0 \rangle_{GS}$", fontsize=18)
    # ax3.set_ylim(min(E_gr)+min(E_gr)/10, max(E_gr)-max(E_gr))
    ax3.grid()
    ax3.plot(gvals, Sbond)

    # SPIN OBSERVABLES
    gvals = data[:, 3]
    alphavals = gvals**2 / np.pi

    Sz = data[:, 5]
    # Sx = data[:, 6]
    # Sy = data[:, 7]

    fig4, ax4 = plt.subplots(1, 1, figsize=[10, 8])
    ax4.set_title(r"$n^\star=%.2f$" % (nprime), fontsize=15)
    # ax4.set_ylim(-0.51, 0.01)

    # ax4[0].set_ylabel(r"$\langle S_x \rangle_{GS}$", fontsize=18)
    # ax4[1].set_ylabel(r"$\langle S_y \rangle_{GS}$", fontsize=18)
    # ax4[0].set_xticklabels([])
    # ax4[1].set_xticklabels([])
    ax4.set_ylabel(r"$\langle S_z \rangle_{GS}$", fontsize=18)
    ax4.set_xlabel(r"$\alpha $", fontsize=18)

    # ax4[0].plot(gvals, Sx, color="r", label=r"MPS")
    # ax4[1].plot(gvals, Sy, color="cyan", label=r"MPS")
    ax4.plot(alphavals, Sz, color="purple", ls="", marker="o", label=r"MPS")

    # if wc != 100:

    #     load_sz = np.loadtxt(folder + f"/g0s_vs_magnetization_wc_{int(wc):d}.txt")
    #     gs, sz = load_sz[:, 0], load_sz[:, 1]
    # load_sz = np.loadtxt(folder + f"/g0s_vs_magnetization_wc_{int(wc):d}.txt")
    # gs, sz = load_sz[:, 0], load_sz[:, 1]

    # ax4.plot(
    #     gs,
    #     sz,
    #     color="purple",
    #     label=r"$Polaron$",
    # )

    #     ax4.plot(
    #         gs,
    #         sz,
    #         color="purple",
    #         label=r"$Polaron$",
    #     )

    # SUBSUBOHMIC
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

    # OHMIC
    alphavals = gvals**2 / np.pi
    delta_r = delta * (delta / wc) ** (alphavals / (1 - alphavals))

    Szteo = -0.5 * delta_r / delta

    ax4.plot(alphavals[:-1], Szteo[:-1], color="k", ls="--", label=r"$Polaron_{teo}$")
    ax4.legend()

    # ax4[0].grid()
    # ax4[1].grid()
    # ax4[2].grid()
    ax4.grid()

    plt.tight_layout()

    if plot_field:
        fig1.savefig(
            write_folder + f"Nx_RealSpace_occupation_{sim_label}.pdf",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig1)

    fig2.savefig(write_folder + f"Energy_{sim_label}.pdf", dpi=600, bbox_inches="tight")
    fig3.savefig(
        write_folder + f"Entropy_{sim_label}.pdf", dpi=600, bbox_inches="tight"
    )
    fig4.savefig(
        write_folder + f"SpinOps_{sim_label}.pdf", dpi=600, bbox_inches="tight"
    )

    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
