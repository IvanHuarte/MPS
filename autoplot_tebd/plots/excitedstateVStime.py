import json
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np


def excitedstateVStime(
    artifact_paths, plot_setup, write_folder, static_args, mode, output_format="png"
):

    color = plot_setup["color"]
    x_lims = plot_setup["x_lims"]
    y_lims = plot_setup["y_lims"]

    n_plots = len(artifact_paths)

    sim_label = ""
    for args in static_args:
        sim_label += f"_{args[0]}_{args[1]}"
    sim_label += (
        f"_{mode}_{list(artifact_paths.keys())[0]}"
        if n_plots == 1
        else f"_runIn_{mode}"
    )

    y_min = -0.05
    y_max = 1.05

    print(f"Static: {static_args}")
    print(f"Current mode: {mode}")
    print(artifact_paths.keys())

    t = []
    Pe = []
    mode_values = []

    for mode_value, artifact_path in artifact_paths.items():
        mode_values.append(mode_value)
        with open(artifact_path[0], "r") as f:
            artifact = json.load(f)

        observables_pkl_path = artifact["results"]["observables"]

        with open(observables_pkl_path, "rb") as fp:
            data = pkl.load(fp)

        t.append(np.array(data["t"]))
        Sz = np.array(data["Sz"])
        Pe.append(Sz + 0.5)

    fig, ax = plt.subplots()
    ax.set_title("Dynamics")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\langle\psi_{e,0} | \psi(t)$")

    for i in range(n_plots):
        times = t[i]
        exc_prob = Pe[i]
        ax.plot(times, exc_prob, color=color, label=f"{mode} = {mode_values[i]}")

    ax.set_ylim(-0.05, 1.05)

    ax.grid()
    ax.legend()
    plt.tight_layout()

    fig.savefig(
        write_folder + f"ExcitedVStime_{sim_label}.{output_format}",
        dpi=600,
        bbox_inches="tight",
    )
