#!/home/ihuarte/Escritorio/Ivan/MPS/.venv/bin/python
import argparse
import copy
import json
import os
import pickle
import platform
import uuid
from datetime import date

import numpy as np
import tenpy
from tenpy.algorithms.tebd import TEBDEngine

from SpinBosonEnv.CavityArrayAtom import CavityArrayAtom
from SpinBosonEnv.GeneralSpinBosonEnv import GeneralSpinBosonEnv

from SpinBosonEnv.measurement import measurement_TEBD_observables, measurement_TEBD_field

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
TEBD_options = copy.deepcopy(config["TEBD_options"])

for k, v in SB_params.items():
    SB_params[k] = np.pi if v == "pi" else v

write_folder = config["write_folder"]
# Create simulation folder
sim_uuid = str(uuid.uuid4())[:8]
write_folder = write_folder + f"{SB_params["ohmic_model"]}/{sim_uuid}/"
os.makedirs(write_folder, exist_ok=True)


# Run in parameters
w0 = SB_params["w0"]
wc = SB_params["wc"]
Nk = SB_params["Nk"]

w_min = w0 * wc / np.sqrt(w0**2 + 4 * wc**2)
# delta_list = [0.15, 0.3, 0.45, 0.48, 0.49, w_min, 0.51, 0.55, 0.7, 0.8, 0.9]
# delta_list = [45, 48, 49, 49.5, wc, 50.5, 51, 55]
# g_list = np.concatenate([np.arange(0.01, 2.05, 0.05)], axis=-1)

g_list = [0.1, 0.5, 0.9]
delta_list = [0.2, 1.0, 2.1]


for delta in delta_list:

    SB_params["delta"] = delta

    for g in g_list:
        SB_params["g"] = g

        sim_artifact = {}
        sim_artifact["SIM"] = config
        field_paths = {"x": []} if sim_setup["field"] else None
        sim_label = f"{SB_params["ohmic_model"]}_w0_{w0:.4f}_wc_{wc:.4f}_Nk_{Nk:.4f}_delta_{delta:.4f}_g_{g:.4f}"

        print("\n************************ TEBD SIMULATION *************************")
        print(f"w0: {SB_params["w0"]:.2f}  wc: {wc:.2f}  Nk: {SB_params["Nk"]} g: {g:.2f}")
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

        caa = CavityArrayAtom(model_params, DMRG_options=None)

        """ TIME EVOLUTION BLOCK DECIMATION """
        config_up = [[0, 1]]
        initial_state = caa.InitialState(config=config_up, GS=False)

        eng_TEBD = TEBDEngine(initial_state, caa, TEBD_options)

        Tmax = TEBD_options['Tmax']
        dt = TEBD_options['dt']
        n_time_steps = int(Tmax / dt + 1)

        # Observables saved as: t | truncation_error | E | S_bond | parity | Sx | Sy | Sz |
        # Field saved as: t | Nx

        observables =  ["t", "trunc_error", "E", "S_bond", "parity", "Sx", "Sy", "Sz"]
        fields = ["t", "Nx"]

        observables_save = dict([(k, []) for k in observables])
        fields_save = dict([(k, []) for k in fields])

        psi = initial_state.copy()

        # t = 0 measurement
        observables_save = measurement_TEBD_observables(eng_TEBD, observables_save)
        fields_save = measurement_TEBD_field(eng_TEBD, fields_save, env_SB.basis)

        step = 0
        while eng_TEBD.evolved_time < Tmax:
            step += 1
            eng_TEBD.run()

            observables_save = measurement_TEBD_observables(eng_TEBD, observables_save)

            print('t: ',eng_TEBD.evolved_time)
            print("Sz: ", observables_save["Sz"][-1])
            print('trunc_error: ',observables_save["trunc_error"][-1])

            if sim_setup["field"] and step % sim_setup["field_each"] == 0:
                print("Calculating population in bosonic field")
                fields_save = measurement_TEBD_field(eng_TEBD, fields_save, env_SB.basis)           

        
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
        sim_artifact["results"] = {"observables": observables_path, "field": field_paths}
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
