import json
import pickle as pkl

import matplotlib.pyplot as plt


def energyVScoupling(
    artifact_paths, plot_setup, write_folder, static_args, mode, output_format="png"
):
    sim_label = ""
    for args in static_args:
        sim_label += f"_{args[0]}_{args[1]}"
    sim_label += f"_runIn_{mode}"

    fig, ax = plt.subplots(figsize=[10, 8])

    for artifact_path in artifact_paths:

        with open(artifact_path, "wb") as f:
            artifact = json.load(artifact_path, f)

        results_path = artifact["results"]["main_results"]
        # Load Main results .pkl
        with open(results_path, "rb") as f:
            data = pkl.load(f)

        # ENERGY
        gvals = data[:, 4]
        alphavals = data[:, 5]
        E_gr = data[:, 6]

        x_data = alphavals if plot_setup["use_alpha"] else gvals
        coup_label = r"$\alpha$" if plot_setup["use_alpha"] else r"$g$"

        ax.set_title(rf"$Energy\;vs.\;{coup_label} \quad (n^\star=%.2f)$", fontsize=18)
        ax.set_xlabel(rf"${coup_label}$", fontsize=18)
        ax.set_ylabel(r"$\langle H \rangle_{\GS}$", fontsize=18)
        ax.set_ylim(min(E_gr) + min(E_gr) / 10, max(E_gr) - max(E_gr))
        ax.grid()
        ax.plot(x_data, E_gr, color="green")

    fig.savefig(
        write_folder + f"Energy_{sim_label}.{output_format}",
        dpi=600,
        bbox_inches="tight",
    )
