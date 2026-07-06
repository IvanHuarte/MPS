#!/home/ihuarte/Escritorio/Ivan/MPS/.venv/bin/python
import argparse
import json
import os
import pickle
import shutil
import uuid

import numpy as np
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

SB_params = config["SB_params"]
model_params = config["model_params"]
DMRG_options = config["DMRG_options"]

for k, v in SB_params.items():
    SB_params[k] = np.pi if v == "pi" else v


# Create simulation folder
sim_uuid = str(uuid.uuid4())[:8]
write_folder = f"/home/ihuarte/Escritorio/Ivan/MPS/Results/{SB_params["ohmic_model"]}/{sim_uuid}/"
os.makedirs(write_folder, exist_ok=True)
shutil.copy("/home/ihuarte/Escritorio/Ivan/MPS/config.json", write_folder)


# Run in parameters

# wc_list = [0.5, 1.0, 5.0, 10.0, 50.0, 100.0]  # , 500.0, 1000.0]
# Nk_list = [101, 101, 101, 101, 301, 501]  # , 2501, 5001]
w0 = SB_params["w0"]
wc = SB_params["wc"]
Nk = SB_params["Nk"] 

w_min = w0 * wc / np.sqrt(w0**2 + 4 * wc**2)

lamda_list = [0.3, 0.45, 0.48, 0.49, w_min, 0.5, 0.51, 0.55, 0.7, 0.8, 0.9]
lamda_list = [45, 48, 49, 49.5, wc, 50.5, 51, 55]

g_list = np.arange(0.0, 2.05, 0.05)

for lamda in lamda_list:
    SB_params["wc"] = wc
    SB_params["Nk"] = Nk

    main_results = np.zeros((len(g_list), 10))

    for i, g in enumerate(g_list):
        SB_params["g"] = g

        print("**********************************************************")
        print(f"w0: {SB_params["w0"]}  wc: {wc}  Nk: {SB_params["Nk"]} g: {g}")
        print(f"delta: {SB_params["delta"]} Boson_dim: {model_params["N_max"]}")
        print("**********************************************************")

        # Spin Boson model init
        env_SB = GeneralSpinBosonEnv(SB_params)
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

        initial_state = caa.InitialState(config=[], GS=False)
        eng = TwoSiteDMRGEngine(initial_state, caa, config["DMRG_options"])
        E_gr, psi_gr = eng.run()

        """ Save results """
        # Save as: w0 | wc | Nk | g | E_gr | Sx | Sy | Sz | S_bond | alpha

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
        main_results[i][1] = wc
        main_results[i][2] = Nk
        main_results[i][3] = g
        main_results[i][4] = E_gr
        main_results[i][5] = Sz[0]
        main_results[i][6] = Sx[0]
        main_results[i][7] = Sy[0]
        main_results[i][8] = Sbond
        main_results[i][9] = env_SB.alpha

        # N = psi_gr.expectation_value(caa.OperatorChain("N"), caa.bs_idx)
        # C = psi_gr.correlation_function(
        #     caa.OperatorChain("Bd"), caa.OperatorChain("B"), caa.bs_idx, caa.bs_idx
        # )
        # Nx = Map2Xcontraction(C, env_SB.basis)

        # np.savetxt(write_folder + "GroundState_wc_%.4f_g_%.4f_delta_%.4f_NoccMap.txt" % (wc, g, SB_params["delta"]), N)
        # np.savetxt(write_folder + "GroundState_wc_%.4f_g_%.4f_delta_%.4f_NoccX.txt" % (wc, g, SB_params["delta"]), Nx)

    with open(write_folder + "Main_results_wc_%.4f_Nk_%.4f_delta_%.4f.pkl" % (wc, Nk, SB_params["delta"]), "wb") as f:
        pickle.dump(main_results, f)
