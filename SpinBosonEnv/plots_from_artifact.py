import json
import os
import pickle as pkl
from pathlib import Path

# plt.rcParams["text.usetex"] = True


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
    # ENERGY VS. COUPLING

    # OCUPPATION VS SITES

    # wmin = w0 * wc / np.sqrt(w0**2 + 4 * wc**2)
