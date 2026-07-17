import json
import os
import pickle as pkl
from pathlib import Path

from SpinBosonEnv.plots.Energy import energyVScoupling
from SpinBosonEnv.plots.Entropy import entropyVScoupling
from SpinBosonEnv.plots.Occupation import occupationVSsites
from SpinBosonEnv.plots.Spin import sxVScoupling, syVScoupling, szVScoupling
from SpinBosonEnv.crossed_plots.Energy import crossed_energyVScoupling

from SpinBosonEnv.toolbox import classify, print_tree


def plot_from_artifact(artifact_path, plot_config):

    with open(artifact_path, "wb") as f:
        artifact = json.load(artifact_path, f)

    sim_label = artifact["labels"]
    new_folder = "Figures_" + sim_label
    folder = str(Path(artifact_path).parent)
    write_folder = folder + "/" + new_folder + "/"
    os.makedirs(write_folder, exist_ok=True)

    results_path = artifact["results"]["main_results"]
    # Load Main results .pkl
    with open(results_path, "rb") as f:
        data = pkl.load(f)

    # Saved as: w0 | wc | Nk | g | alpha | delta | E_gr | Sz | Sx | Sy | S_bond |

    # OCUPPATION VS SITES
    if plot_config["occupationVSsites"]["on"]:
        occupationVSsites(
            data,
            artifact,
            plot_config["occupationVSsites"],
            write_folder,
            **plot_config["MACROS"],
        )

    # ENERGY VS. COUPLING
    if plot_config["energyVScoupling"]["on"]:
        energyVScoupling(
            data,
            artifact,
            plot_config["energyVScoupling"],
            write_folder,
            **plot_config["MACROS"],
        )

    # ENTROPY VS. COUPLING
    if plot_config["entropyVScoupling"]["on"]:
        entropyVScoupling(
            data,
            artifact,
            plot_config["entropyVScoupling"],
            write_folder,
            **plot_config["MACROS"],
        )

    # SPIN OBSERVABLES VS. COUPLING
    if plot_config["spinobsVScoupling"]["sx"]:
        sxVScoupling(
            data,
            artifact,
            plot_config["spinobsVScoupling"],
            write_folder,
            **plot_config["MACROS"],
        )
    if plot_config["spinobsVScoupling"]["sy"]:
        syVScoupling(
            data,
            artifact,
            plot_config["spinobsVScoupling"],
            write_folder,
            **plot_config["MACROS"],
        )
    if plot_config["spinobsVScoupling"]["sz"]:
        szVScoupling(
            data,
            artifact,
            plot_config["spinobsVScoupling"],
            write_folder,
            **plot_config["MACROS"],
        )


def plot_artifact_batch(artifact_paths, plot_config, static_args, dynamic_args, folder):

    new_folder = "Figures_"
    for args in static_args:
        new_folder += f"_{args[0]}_{args[1]}"
    write_folder = folder + "/" + new_folder + "/"
    os.makedirs(write_folder, exist_ok=True)

    # Energy
    if plot_config["crossed_energyVScoupling"]["on"]:
        crossed_energyVScoupling(
            artifact_paths, 
            plot_config["crossed_energyVScoupling"], 
            write_folder, 
            static_args, 
            dynamic_args,
            **plot_config["MACROS"]
        )

    # Occupation
    if plot_config["crossed_occupationVSsites"]["on"]:
        crossed_energyVScoupling(
            artifact_paths, 
            plot_config["crossed_occupationVSsites"], 
            write_folder, 
            static_args, 
            dynamic_args,
            **plot_config["MACROS"]
        )



def recursive_plots(grouped_paths, modes, plot_config, folder, static_through_plots=[]):

    mode, remaining_modes = modes.split("/", 1) # if "/" in modes else (modes, "")

    for k, v in grouped_paths.items():

        if not "/" in remaining_modes: #isinstance(list(v.values())[0], list):
            plot_artifact_batch(
                grouped_paths[k], plot_config, static_through_plots + [(mode, k)], remaining_modes, folder
            )

        elif isinstance(v, dict):
            recursive_plots(
                grouped_paths[k],
                modes=remaining_modes,
                plot_config=plot_config,
                folder=folder,
                static_through_plots=static_through_plots + [(mode, k)],
            )


def crossed_artifact_plots(artifact_paths, plot_config, folder):

    mode = plot_config["mode"]

    grouped_paths = classify(artifact_paths, mode)

    print_tree(grouped_paths, values=False)

    recursive_plots(grouped_paths, mode, plot_config, folder)
