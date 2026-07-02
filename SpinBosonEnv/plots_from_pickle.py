import os
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np


def plot_from_pickle(pickle_path):

    sim_label = Path(pickle_path).name.split(".pkl")[0].split("Main_results_")[1]
    new_folder = "Figures_" + sim_label
    folder = str(Path(pickle_path).parent)
    write_folder = folder + "/" + new_folder + "/"
    os.makedirs(write_folder, exist_ok=True)

    with open(pickle_path, "rb") as f:
        data = pkl.load(f)
    # Saved as: w0 | wc | Nk | g | E_gr | Sx | Sy | Sz | S_bond |

    # OCUPPATION VS SITES

    w0 = int(data[:, 0][0])
    wc = int(data[:, 1][0])
    Nk = int(data[:, 2][0])
    gvals = data[:, 3]

    nprime = wc / w0

    fig1, ax1 = plt.subplots(figsize=[12, 8])
    ax1.set_title(
        r"$Occupation\;decayment \quad (n^\star=%.2f)$" % (nprime), fontsize=18
    )
    ax1.set_xlabel(r"$n$", fontsize=18)
    ax1.set_ylabel(r"$\langle a_n^\dagger a_n \rangle_{GS}$", fontsize=18)
    ax1.set_ylim(1e-18, 1e1)
    ax1.set_xlim(9e-1, Nk // 2 + 2)

    ax1.vlines(nprime, 1e-18, 1e1, color="grey", ls="--")
    ax1.text(
        nprime + 1,
        0.1,
        r"$n^\star = \frac{\omega_c}{\omega_0}$",
        transform=transforms.blended_transform_factory(ax1.transData, ax1.transAxes),
        fontsize=12,
    )
    cmap = plt.colormaps["viridis"](np.linspace(0, 1, len(gvals)))

    for i, g in enumerate(gvals):
        Nx = np.loadtxt(
            folder + "/GroundState_wc_%.4f_g_%.4f_NoccX.txt" % (wc, g),
            dtype=complex,
        ).real
        Nxhalf = Nx[Nk // 2 :]
        n_values = np.arange(len(Nxhalf))

        # TEORIC
        # n < nprime
        Nteo_1 = g**2 / (np.pi**2 * w0**2 * nprime**2) * (1 / n_values**4)
        # n > nprime
        Nteo_2 = (
            g**2
            / (2 * np.pi * w0**2 * nprime**2)
            * (np.exp(-2 * n_values / nprime) / n_values**3)
        )
        Nteo = np.concatenate([Nteo_1[1 : int(nprime)], Nteo_2[int(nprime) :]])
        ax1.plot(range(1, len(n_values)), Nteo, color=cmap[i])

        # MPS RESULTS
        ax1.plot(
            range(len(Nxhalf)),
            Nxhalf,
            color=cmap[i],
            marker="o",
            ls="",
            ms=3,
            label=r"$g=%.2f$" % g,
        )

    ax1.legend()
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.grid()

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
    Sz = data[:, 5]
    Sx = data[:, 6]
    Sy = data[:, 7]

    fig4, ax4 = plt.subplots(3, 1, figsize=[6, 12])
    ax4[0].set_title(r"$n^\star=%.2f$" % (nprime), fontsize=15)
    ax4[2].set_ylim(-0.51, 0.01)

    ax4[0].set_ylabel(r"$\langle S_x \rangle_{GS}$", fontsize=18)
    ax4[1].set_ylabel(r"$\langle S_y \rangle_{GS}$", fontsize=18)
    ax4[2].set_ylabel(r"$\langle S_z \rangle_{GS}$", fontsize=18)
    ax4[0].set_xticklabels([])
    ax4[1].set_xticklabels([])
    ax4[2].set_xlabel(r"$g$", fontsize=18)

    ax4[0].plot(gvals, Sx, color="r")
    ax4[1].plot(gvals, Sy, color="cyan")
    ax4[2].plot(gvals, Sz, color="purple")

    ax4[0].grid()
    ax4[1].grid()
    ax4[2].grid()

    plt.tight_layout()

    fig1.savefig(
        write_folder + f"Nx_RealSpace_occupation_{sim_label}.pdf",
        dpi=600,
        bbox_inches="tight",
    )
    fig2.savefig(write_folder + f"Energy_{sim_label}.pdf", dpi=600, bbox_inches="tight")
    fig3.savefig(
        write_folder + f"Entropy_{sim_label}.pdf", dpi=600, bbox_inches="tight"
    )
    fig4.savefig(
        write_folder + f"SpinOps_{sim_label}.pdf", dpi=600, bbox_inches="tight"
    )

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
