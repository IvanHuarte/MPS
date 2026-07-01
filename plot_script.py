#!/home/ihuarte/Escritorio/Ivan/MPS/.venv/bin/python

import argparse
import glob

from SpinBosonEnv.plots_from_pickle import plot_from_pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--directory",
    required=True,
    help="Directory of MPS results (pkl)",
)

args = parser.parse_args()
directory = args.directory

files = glob.glob(f"{directory}/**/*.pkl", recursive=True)

for file in files:
    print(f"\n\nFile:{file}")

    plot_from_pickle(file)
