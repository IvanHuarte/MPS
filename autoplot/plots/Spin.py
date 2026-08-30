import matplotlib.pyplot as plt

from autoplot.plots.Theory import sz_ohmic, sz_subsubohmic


def calc_sz_teo(x_data, ohmic_model, w0, wc, delta):

    if ohmic_model == "SubSubOhmic":
        return sz_subsubohmic(x_data, w0, wc, delta)

    if ohmic_model == "Ohmic":
        return sz_ohmic(x_data, wc, delta)

    else:
        raise ValueError(f"No such ohmic model: {ohmic_model}")


def sxVScoupling(data, artifact, plot_setup, write_folder, output_format="png"):

    sim_label = artifact["labels"]

    gvals = data[:, 3]
    alphavals = data[:, 4]
    Sx = data[:, 8]

    x_data = alphavals if plot_setup["use_alpha"] else gvals
    curve_label = r"$\alpha$" if plot_setup["use_alpha"] else r"$g$"

    fig, ax = plt.subplots(1, 1, figsize=[10, 8])
    ax.set_ylabel(r"$\langle S_x \rangle_{GS}$", fontsize=18)
    ax.set_xlabel(rf"{curve_label}")

    ax.plot(x_data, Sx, color="r", label=r"MPS")

    fig.savefig(
        write_folder + f"Energy_{sim_label}.{output_format}",
        dpi=600,
        bbox_inches="tight",
    )


def syVScoupling(data, artifact, plot_setup, write_folder, output_format="png"):

    sim_label = artifact["labels"]

    gvals = data[:, 3]
    alphavals = data[:, 4]
    Sy = data[:, 9]

    x_data = alphavals if plot_setup["use_alpha"] else gvals
    curve_label = r"$\alpha$" if plot_setup["use_alpha"] else r"$g$"

    fig, ax = plt.subplots(1, 1, figsize=[10, 8])
    ax.set_ylabel(r"$\langle S_y \rangle_{GS}$", fontsize=18)
    ax.set_xlabel(rf"{curve_label}")

    ax.plot(x_data, Sy, color="blue", label=r"MPS")

    fig.savefig(
        write_folder + f"Energy_{sim_label}.{output_format}",
        dpi=600,
        bbox_inches="tight",
    )


def szVScoupling(data, artifact, plot_setup, write_folder, output_format="png"):

    sim_label = artifact["labels"]

    w0 = int(data[:, 0][0])
    delta = float(data[:, 1][0])
    wc = int(data[:, 2][0])

    gvals = data[:, 3]
    alphavals = data[:, 4]
    Sz = data[:, 9]

    x_data = alphavals if plot_setup["use_alpha"] else gvals
    curve_label = r"$\alpha$" if plot_setup["use_alpha"] else r"$g$"

    fig, ax = plt.subplots(1, 1, figsize=[10, 8])
    ax.set_ylabel(r"$\langle S_z \rangle_{GS}$", fontsize=18)
    ax.set_xlabel(rf"{curve_label}")

    if plot_setup["plot_theo"]:
        ohmic_model = artifact["SIM"]["SB_params"]
        Sz_teo = calc_sz_teo(ohmic_model, w0, wc, delta)
        ax.plot(x_data, Sz_teo, color="purple", label=r"$Polaron_{teo}$")
        ax.plot(x_data, Sz, color="purple", ls="", marker="o", ms=2, label=r"MPS")
        ax.grid()
        ax.legend()
    else:
        ax.plot(x_data, Sz, color="purple")

    fig.savefig(
        write_folder + f"Energy_{sim_label}.{output_format}",
        dpi=600,
        bbox_inches="tight",
    )
