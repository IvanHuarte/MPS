#!/home/ihuarte/Escritorio/Ivan/MPS/.venv/bin/python

import argparse
import glob
import json
from pathlib import Path

from SpinBosonEnv.plots_from_artifact import crossed_artifact_plots, plot_from_artifact

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--directory",
    required=True,
    help="Directory of MPS artifacts (.json)",
)

args = parser.parse_args()
directory = args.directory

project_folder = str((Path(__file__).parent)) + "/"
with open(project_folder + "/config_plots.json", "r") as f:
    plot_config = json.load(f)

files = glob.glob(f"{directory}/**/*results*.json", recursive=True)

# Singles
if plot_config["single"]["on"]:
    for artifact_path in files:
        print(f"\n\nFile:{artifact_path}")
        plot_from_artifact(artifact_path, plot_config["single"])

# Crossed plots
if plot_config["crossed"]["on"]:
    crossed_artifact_plots(files, plot_config["crossed"], directory)
