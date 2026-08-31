import os

from .plots.excitedstateVStime import excitedstateVStime
from .utils import classify, main_artifacts_filtering


def plot_artifact_batch(artifact_paths, plot_config, static_args, dynamic_args, folder):

    new_folder = "Figures"
    for args in static_args:
        new_folder += f"_{args[0]}_{args[1]}"
    write_folder = folder + "/" + new_folder + "/"
    os.makedirs(write_folder, exist_ok=True)

    # Quality control
    if plot_config["MACROS"]["filter_mode"] is not None:
        artifact_paths = main_artifacts_filtering(artifact_paths, plot_config["MACROS"])

    # Energy
    if plot_config["energyVSirrep"]["on"]:
        print("Energy-vs-Irrep")

        excitedstateVStime(
            artifact_paths,
            plot_config["energyVSirrep"],
            write_folder,
            static_args,
            dynamic_args,
            output_format=plot_config["MACROS"]["output_format"],
        )

    print("\n")


def recursive_plots(grouped_paths, modes, plot_config, folder, static_through_plots=[]):

    mode, remaining_modes = modes.split("|", 1) if "|" in modes else (modes, "")

    for k, v in grouped_paths.items():

        if not "|" in remaining_modes:  # isinstance(list(v.values())[0], list):
            plot_artifact_batch(
                v,
                plot_config,
                static_through_plots + [(mode, k)],
                remaining_modes,
                folder,
            )

        elif isinstance(v, dict):
            if remaining_modes.endswith("|") and remaining_modes.count("|") == 1:
                remaining_mode = remaining_modes[:-1]
                for kk, vv in v.items():

                    plot_artifact_batch(
                        {kk: vv},
                        plot_config,
                        static_through_plots + [(mode, k)],
                        remaining_mode,
                        folder,
                    )
            else:
                recursive_plots(
                    v,
                    modes=remaining_modes,
                    plot_config=plot_config,
                    folder=folder,
                    static_through_plots=static_through_plots + [(mode, k)],
                )


def crossed_artifact_plots(artifact_paths, plot_config, folder):

    mode = plot_config["mode"]

    grouped_paths = classify(artifact_paths, mode)
    from .utils import print_tree

    print_tree(grouped_paths, values=False)

    # print_tree(grouped_paths, values=False)
    recursive_plots(grouped_paths, mode, plot_config, folder)
