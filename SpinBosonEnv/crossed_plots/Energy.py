import json
import pickle as pkl
import numpy as np

import matplotlib.pyplot as plt

from SpinBosonEnv.toolbox import join_modes, join_static_modes



def crossed_energyVScoupling(
    artifact_paths, 
    plot_setup, 
    write_folder, 
    static_args, 
    mode, 
    output_format="png",

):

    cmap = plot_setup["cmap"]
    x_lims = plot_setup["x_lims"]
    y_lims = plot_setup["y_lims"]

    sim_label = ""
    for args in static_args:
        sim_label += f"_{args[0]}_{args[1]}"
    sim_label += f"_runIn_{mode}"

    fig, ax = plt.subplots(figsize=[10, 8])

    modes_list = mode.split("-")

    cmap = plt.colormaps[cmap](np.linspace(0.1, 0.9, len(artifact_paths)))

    y_min = np.inf
    y_max = -np.inf

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

        # ENERGY
        gvals = data[:, 4]
        alphavals = data[:, 5]
        E_gr = data[:, 6]

        y_min = min(E_gr) if y_min > min(E_gr) else y_min
        y_max = max(E_gr) if y_max < max(E_gr) else y_max

        x_data = alphavals if plot_setup["use_alpha"] else gvals
        coup_label = r"\alpha" if plot_setup["use_alpha"] else r"g"

        ax.plot(x_data, E_gr, color=cmap[i], ls="", marker='o', ms=3, label=rf"${label}$")

    static_args_text = join_static_modes(static_args)
    ax.text(
        0.7, 
        0.9, 
        rf"{static_args_text}",
        transform=ax.transAxes, 
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black",
            alpha=0.8,
        ),
        multialignment="center"
    )

    ax.set_title(rf"$Energy\;vs.\;{coup_label}$", fontsize=18)
    ax.set_xlabel(rf"${coup_label}$", fontsize=18)
    ax.set_ylabel(r"$\langle H \rangle$", fontsize=18)
    ax.set_ylim(y_min - np.sign(y_min) *  y_min / 10 - 0.1, y_max - np.sign(y_min) * y_max / 10 + 0.1)
    if x_lims is not None:
        ax.set_xlim(*x_lims)
    if y_lims is not None:
        ax.set_ylim(*y_lims)
    ax.legend()
    ax.grid()
    

    fig.savefig(
        write_folder + f"EnergyVS{coup_label}{sim_label}.{output_format}",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)
