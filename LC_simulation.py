#!/home/ihuarte/Escritorio/Ivan/MPS/.venv/bin/python

import argparse
import copy
import json
import os
import pickle
import platform
import time
import uuid
from datetime import date

import numpy as np
import tenpy
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
from tenpy.algorithms.tebd import TEBDEngine

from SpinBosonEnv.CavityArrayAtom import CavityArrayAtom
from SpinBosonEnv.GeneralSpinBosonEnv import GeneralSpinBosonEnv
from SpinBosonEnv.measurement import (
    measurement_TEBD_field,
    measurement_TEBD_observables,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    action="append",
    required=False,
    help="Parse configuration files in order. Simulation/CM/NN",
)

args = parser.parse_args()

if args.config is None:
    args.config = ["/home/ihuarte/Escritorio/Ivan/MPS/config.json"]


with open("/home/ihuarte/Escritorio/Ivan/MPS/config.json", "r") as f:
    config = json.load(f)

sim_setup = copy.deepcopy(config["sim_setup"])
SB_params = copy.deepcopy(config["SB_params"])
model_params = copy.deepcopy(config["model_params"])
DMRG_options = copy.deepcopy(config["DMRG_options"])
TEBD_options = copy.deepcopy(config["TEBD_options"])

for k, v in SB_params.items():
    SB_params[k] = np.pi if v == "pi" else v

write_folder = config["write_folder"]
# Create simulation folder
sim_uuid = str(uuid.uuid4())[:8]
write_folder = write_folder + f"{SB_params["ohmic_model"]}/UUID_{sim_uuid}/"
os.makedirs(write_folder, exist_ok=True)


# Run in parameters
w0 = SB_params["w0"]
wc = SB_params["wc"]
Nk = SB_params["Nk"]

w_min = w0 * wc / np.sqrt(w0**2 + 4 * wc**2)

g_list = [0.1, 0.3, 0.5, 0.7, 0.9]
delta_list = [0.2, 1.0, 2.1]

calc_obs_steps = 2 ** np.arange(11)

for delta in delta_list:

    SB_params["delta"] = delta

    for g in g_list:
        SB_params["g"] = g

        sim_artifact = {}
        sim_artifact["SIM"] = config
        field_paths = {"x": []} if sim_setup["field"] else None
        sim_label = f"{SB_params["ohmic_model"]}_w0_{w0:.4f}_wc_{wc:.4f}_Nk_{Nk:.4f}_delta_{delta:.4f}_g_{g:.4f}"

        print("\n************************ TEBD SIMULATION *************************")
        print(
            f"w0: {SB_params["w0"]:.2f}  wc: {wc:.2f}  Nk: {SB_params["Nk"]} g: {g:.2f}"
        )
        print(f"delta: {SB_params["delta"]:.2f} Boson_dim: {model_params["N_max"]}")
        print("********************************************************************")

        # Spin Boson model init
        env_SB = GeneralSpinBosonEnv(SB_params)
        print(f"\nAlpha: {env_SB.alpha}")
        print(f"Hk: {env_SB.Hk.shape}")
        print(f"Hmap: {env_SB.Hmap.shape}")

        # Tenpy init
        model_params["L"] = len(env_SB.wlist)
        model_params["w"] = env_SB.wlist
        model_params["delta"] = SB_params["delta"]
        model_params["g"] = env_SB.g0
        model_params["J"] = env_SB.Jlist

        tini_dmrg = time.time()

        """ INITIAL STATE PREPARATION """

        caa = CavityArrayAtom(model_params, DMRG_options=None)
        spin_up = [[1, 0]]
        spin_down = [[0, 1]]
        # initial_state = caa.InitialState(config=config_up, GS=False)
        dmrg_init_state = caa.InitialState(config=spin_down, GS=False)
        E, dmrg_ground_state = TwoSiteDMRGEngine(
            dmrg_init_state, caa, DMRG_options
        ).run()
        dmrg_ground_state.apply_local_op(0, "Sx")
        tebd_initial_state = dmrg_ground_state.copy()
        tend_dmrg = time.time()
        dt_dmrg = tend_dmrg - tini_dmrg
        print(f"DMRG time: {dt_dmrg}")

        """ TIME EVOLUTION BLOCK DECIMATION """
        eng_TEBD = TEBDEngine(tebd_initial_state, caa, TEBD_options)

        print(eng_TEBD.options)
        Tmax = TEBD_options["Tmax"]
        dt = TEBD_options["dt"]
        n_time_steps = int(Tmax / dt + 1)

        # Observables saved as: t | truncation_error | E | S_bond | parity | Sx | Sy | Sz |
        # Field saved as: t | Nx

        observables = ["t", "trunc_error", "E", "S_bond", "Sx", "Sy", "Sz"]
        fields = ["t", "Nx"]

        observables_save = dict([(k, []) for k in observables])
        fields_save = dict([(k, []) for k in fields])

        psi = tebd_initial_state.copy()

        # t = 0 measurement
        observables_save = measurement_TEBD_observables(eng_TEBD, observables_save)
        fields_save = measurement_TEBD_field(eng_TEBD, fields_save, env_SB.basis)

        step = 0
        while eng_TEBD.evolved_time < Tmax:
            print("t: ", eng_TEBD.evolved_time)

            step += 1

            tini_tebd = time.time()
            eng_TEBD.run()
            tend_tebd = time.time()
            dt_tebd = tend_tebd - tini_tebd
            print(f"tebd step time: {dt_tebd}")

            # observables_save = measurement_TEBD_observables(eng_TEBD, observables_save)
            # print("Sz: ", observables_save["Sz"][-1])
            # print("trunc_error: ", observables_save["trunc_error"][-1])

            if step in calc_obs_steps:
                observables_save = measurement_TEBD_observables(
                    eng_TEBD, observables_save
                )
                print("Sz: ", observables_save["Sz"][-1])
                print("Calculating population in bosonic field")
                fields_save = measurement_TEBD_field(
                    eng_TEBD, fields_save, env_SB.basis
                )

        # Set MAIN PARAMS to simulation artifact
        sim_artifact["SIM"]["SB_params"]["w0"] = SB_params["w0"]
        sim_artifact["SIM"]["SB_params"]["delta"] = SB_params["delta"]
        sim_artifact["SIM"]["SB_params"]["wc"] = SB_params["wc"]
        sim_artifact["SIM"]["SB_params"]["Nk"] = SB_params["Nk"]
        sim_artifact["SIM"]["SB_params"]["g"] = SB_params["g"]
        sim_artifact["SIM"]["SB_params"]["alpha"] = env_SB.alpha

        # Save OBSERVABLEs values over time
        observables_path = write_folder + "Observables_over_time_" + sim_label + ".pkl"
        with open(
            observables_path,
            "wb",
        ) as f:
            pickle.dump(observables_save, f)

        # Save FIELD occupation over time
        Nx_path = write_folder + "Field_over_time_" + sim_label + "_NoccX.pkl"
        with open(
            Nx_path,
            "wb",
        ) as f:
            pickle.dump(fields_save, f)
        field_paths["x"].append(Nx_path)

        # Save artifact
        sim_artifact["results"] = {
            "observables": observables_path,
            "field": field_paths,
        }
        sim_artifact["labels"] = {"sim_label": sim_label}
        # Write metadata in main setup artifact
        metadata = {
            "uuid": sim_uuid,
            "date": date.today().strftime("%x"),
            "architecture": platform.architecture(),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": tenpy.__version__,
            },
            "device": {
                "processor": platform.processor(),
                "machine": platform.machine(),
                "platform": platform.platform(),
            },
        }

        sim_artifact["metadata"] = metadata
        artifact_path = (
            write_folder + "Simulation_results_" + sim_label + f"_UUID_{sim_uuid}.json"
        )
        with open(
            artifact_path,
            "w",
        ) as f:
            json.dump(sim_artifact, f, separators=(",", ":"), sort_keys=True, indent=4)
