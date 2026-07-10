import os
import pickle as pkl
from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import scipy

# plt.rcParams["text.usetex"] = True


def plot_from_artifact(artifact_path, plot_config):

    with open(artifact_path, "wb") as f:
        artifact = json.load(artifact_path, f)

    sim_label = Path(artifact_path).name.split(".pkl")[0].split("Main_results_")[1]
    new_folder = "Figures_" + sim_label
    folder = str(Path(artifact_path).parent)
    write_folder = folder + "/" + new_folder + "/"
    os.makedirs(write_folder, exist_ok=True)

    with open(artifact_path, "rb") as f:
        data = pkl.load(f)
    # Saved as: w0 | wc | Nk | g | E_gr | Sx | Sy | Sz | S_bond | alpha | delta

    # OCUPPATION VS SITES

    w0 = int(data[:, 0][0])
    wc = int(data[:, 1][0])
    Nk = int(data[:, 2][0])
    print(data.shape)
    delta = 0.1  # float(data[:, 10][0])
    gvals = data[:, 3][1::5]

    nprime = wc / w0
    wmin = w0 * wc / np.sqrt(w0**2 + 4 * wc**2)

   
