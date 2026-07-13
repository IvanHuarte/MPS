#!/home/ihuarte/Escritorio/Ivan/MPS/.venv/bin/python
import argparse
import json
import os
import pickle
import platform
import uuid
from datetime import date

import numpy as np
import tenpy
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine

from SpinBosonEnv.Basis import Map2Xcontraction
from SpinBosonEnv.CavityArrayAtom import CavityArrayAtom
from SpinBosonEnv.GeneralSpinBosonEnv import GeneralSpinBosonEnv

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

sim_setup = config["sim_setup"]
SB_params = config["SB_params"]
model_params = config["model_params"]
DMRG_options = config["DMRG_options"]

for k, v in SB_params.items():
    SB_params[k] = np.pi if v == "pi" else v


# Create simulation folder
sim_uuid = str(uuid.uuid4())[:8]
write_folder = (
    f"/home/ihuarte/Escritorio/Ivan/MPS/Results/{SB_params["ohmic_model"]}/{sim_uuid}/"
)
os.makedirs(write_folder, exist_ok=True)


# Run in parameters
w0 = SB_params["w0"]
wc = SB_params["wc"]
Nk = SB_params["Nk"]

w_min = w0 * wc / np.sqrt(w0**2 + 4 * wc**2)
delta_list = [0.15]  # , 0.45] , 0.48, 0.49, w_min, 0.5, 0.51, 0.55, 0.7, 0.8, 0.9]
# delta_list = [45, 48, 49, 49.5, wc, 50.5, 51, 55]
g_list = np.concatenate([np.arange(0.01, 2.05, 0.05)], axis=-1)

for delta in delta_list:

    SB_params["delta"] = delta

    main_results = np.zeros((len(g_list), 11))
    sim_artifact = {}
    sim_artifact["SIM"] = config

    if sim_setup["field"]:
        field_paths = {"map": [], "x": []}

    sim_label = f"{SB_params["ohmic_model"]}_w0_{w0:.4f}_wc_{wc:.4f}_Nk_{Nk:.4f}_delta_{delta:.4f}"

    for i, g in enumerate(g_list):
        SB_params["g"] = g

        print("**********************************************************")
        print(f"w0: {SB_params["w0"]}  wc: {wc}  Nk: {SB_params["Nk"]} g: {g}")
        print(f"delta: {SB_params["delta"]} Boson_dim: {model_params["N_max"]}")
        print("**********************************************************")

        # Spin Boson model init
        env_SB = GeneralSpinBosonEnv(SB_params)
        print(f"Alpha: {env_SB.alpha}")
        print(f"Hk: {env_SB.Hk.shape}")
        print(f"Hmap: {env_SB.Hmap.shape}")

        # Tenpy init
        model_params["L"] = len(env_SB.wlist)
        model_params["w"] = env_SB.wlist
        model_params["delta"] = SB_params["delta"]
        model_params["g"] = env_SB.g0
        model_params["J"] = env_SB.Jlist

        caa = CavityArrayAtom(model_params, DMRG_options)

        """ GROUND STATE AND ENERGY """
        initial_state = caa.InitialState(config=[[0, 1]], GS=False)
        if model_params["conserve"] == "parity":
            P0 = caa.calc_mps_parity(initial_state)
            print(f"psi0 parity: {P0}")

        eng = TwoSiteDMRGEngine(initial_state, caa, config["DMRG_options"])
        E_gr, psi_gr = eng.run()

        if model_params["conserve"] == "parity":
            P1 = caa.calc_mps_parity(initial_state)
            print(f"psi ground parity: {P1}")
        """ Save results """
        # Saved as: w0 | wc | Nk | g | alpha | delta | E_gr | Sz | Sx | Sy | S_bond |

        Sz = psi_gr.expectation_value("Sz", caa.atpos_idx)
        Sx = psi_gr.expectation_value("Sx", caa.atpos_idx)
        Sy = psi_gr.expectation_value("Sy", caa.atpos_idx)
        Sbond = psi_gr.entanglement_entropy(n=1)[0]

        print("E_gr:%.2f " % E_gr)
        print("Sz: ", Sz)
        print("Sx", Sx)
        print("Sy", Sy)
        print("Entanglement_S:", Sbond, "\n\n")

        main_results[i][0] = SB_params["w0"]
        main_results[i][10] = SB_params["delta"]
        main_results[i][1] = wc
        main_results[i][2] = Nk
        main_results[i][3] = g
        main_results[i][9] = env_SB.alpha
        main_results[i][4] = E_gr
        main_results[i][5] = Sz[0]
        main_results[i][6] = Sx[0]
        main_results[i][7] = Sy[0]
        main_results[i][8] = Sbond

        if sim_setup["field"]:
            print("Calculating population in bosonic field")

            N = psi_gr.expectation_value(caa.OperatorChain("N"), caa.bs_idx)
            C = psi_gr.correlation_function(
                caa.OperatorChain("Bd"), caa.OperatorChain("B"), caa.bs_idx, caa.bs_idx
            )
            Nx = Map2Xcontraction(C, env_SB.basis)
            # print(f"N: ({type(N)})\n {N}")
            # print(f"Nx: ({type(Nx)})\n {Nx}")

            Nmap_path = write_folder + "Field_" + sim_label + f"_g_{g}" + "_NoccMap.txt"
            Nx_path = write_folder + "Field_" + sim_label + f"_g_{g}" + "_NoccX.txt"

            np.savetxt(
                Nmap_path,
                N,
            )
            np.savetxt(
                Nx_path,
                Nx,
            )
            field_paths["map"].append(Nmap_path)
            field_paths["x"].append(Nx_path)

    # Save results
    main_results_path = write_folder + "Main_results_" + sim_label + ".pkl"

    with open(
        main_results_path,
        "wb",
    ) as f:
        pickle.dump(main_results, f)

    # Save artifact
    sim_artifact["results"] = {"main_results": main_results_path, "field": field_paths}
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
