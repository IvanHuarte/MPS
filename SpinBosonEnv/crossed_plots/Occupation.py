import json
import pickle as pkl
import numpy as np
import scipy

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from SpinBosonEnv.toolbox import join_modes, join_static_modes


def w_min(ohmic_model, w0, wc):

    if ohmic_model == "SubSubOhmic":
        return w0 * wc / np.sqrt(w0**2 + 4 * wc**2)


def crossed_occupationVSsites(
    artifact_paths, 
    plot_setup, 
    write_folder, 
    static_args, 
    mode, 
    output_format="png"
):
    cmap = plot_setup["cmap"]   
    x_lims = plot_setup["x_lims"]
    y_lims = plot_setup["y_lims"]
    coupling_indexes = plot_setup["coupling_indexes"]
    coupling_indexes = np.array(coupling_indexes)
    plot_theo = plot_setup["plot_theo"]
    use_alpha = plot_setup["use_alpha"]


    sim_label = ""
    for args in static_args:
        sim_label += f"_{args[0]}_{args[1]}"
    sim_label += f"_runIn_{mode}"

    fig, ax = plt.subplots(figsize=[10, 8])

    modes_list = mode.split("-")

    for i, (modes_values_list, artifact_path) in enumerate(artifact_paths.items()):

        modes_values_list = [modes_values_list] if not isinstance(modes_values_list, (list, tuple)) else modes_values_list

        label = join_modes(modes_list, modes_values_list)

        with open(artifact_path[0], "r") as f:
            artifact = json.load(f)

        results_path = artifact["results"]["main_results"]
        # Load Main results .pkl
        with open(results_path, "rb") as f:
            data = pkl.load(f)

        #########################################################

        w0 = int(data[:, 0][0])
        delta = float(data[:, 1][0])
        wc = int(data[:, 2][0])
        Nk = int(data[:, 3][0])

        gvals = data[:, 4][coupling_indexes]
        alphavals = data[:, 5][coupling_indexes]
        indexes = np.arange(data.shape[0])[coupling_indexes]

    x_data = alphavals if use_alpha else gvals
    curve_label = r"\alpha" if use_alpha else r"g"

    nprime = wc / w0
    ohmic_model = artifact["SIM"]["SB_parmas"]["ohmic_model"]
    wmin = w_min(ohmic_model, w0=w0, wc=wc)

    cmap = plt.colormaps["inferno"](np.linspace(0, 1, len(gvals)))
    y_min = 1

    field_paths = artifact["results"]["field"]["x"]
    field_paths = [field_paths[i] for i in coupling_indexes]

    for i, (coupling, field_path) in enumerate(zip(x_data, field_paths)):

        Nx = np.loadtxt(
            field_path,
            dtype=complex,
        ).real

        Nxhalf = Nx[Nk // 2 :]
        n_values = np.arange(len(Nxhalf))
        if min(Nxhalf) < y_min:
            y_min = min(Nxhalf)

        # TEORIC
        if plot_setup["plot_theo"]:
            Nteo = (
                coupling**2
                / (np.pi**2 * w0**2 * nprime**2)
                * (scipy.special.k1(n_values / nprime) / n_values) ** 2
            )
            ax.plot(np.arange(0, len(n_values)) / nprime, Nteo, color=cmap[i])

        # MPS RESULTS
        ax.plot(
            np.arange(len(Nxhalf)) / nprime,
            Nxhalf,
            color=cmap[i],
            marker="o",
            ls="",
            ms=3,
            label=rf"${curve_label}={coupling}$",
        )

    ax.set_title(
        r"$Ground\;state\;photons \quad (n^\star=%.2f)$" % (nprime), fontsize=18
    )
    ax.set_xlabel(r"$n/n^\star$", fontsize=18)
    ax.set_ylabel(r"$\langle a_n^\dagger a_n \rangle_{GS}$", fontsize=18)
    ax.set_xlim(1 / nprime - 0.001, Nk // 2 / nprime)

    ax.text(
        1,
        0.1,
        r"$n^\star = \frac{\omega_c}{\omega_0}$",
        transform=transforms.blended_transform_factory(ax.transData, ax.transAxes),
        fontsize=12,
    )
    ax.text(
        0.4,
        0.9,
        (
            rf"$\Delta = {delta:.3f}$"
            "\n"
            rf"$\omega_c = {wc:.3f}$"
            "\n"
            rf"$\omega_{{\min}} = {wmin:.3f}$"
        ),
        transform=transforms.blended_transform_factory(ax.transAxes, ax.transAxes),
        fontsize=12,
    )
    ax.set_ylim(y_min, 1e1)
    ax.vlines(1, y_min, 1e1, color="grey", ls="--")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid()
    fig.savefig(
        write_folder + f"Nx_RealSpace_occupation_{sim_label}.{output_format}",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)
