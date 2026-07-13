import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import scipy


def w_min(ohmic_model, w0, wc):

    if ohmic_model == "SubSubOhmic":
        return w0 * wc / np.sqrt(w0**2 + 4 * wc**2)


def occupationVSsites(data, artifact, plot_setup, write_folder, output_format="png"):

    # OCUPPATION VS SITES
    sim_label = artifact["labels"]

    w0 = int(data[:, 0][0])
    delta = float(data[:, 1][0])
    wc = int(data[:, 2][0])
    Nk = int(data[:, 3][0])

    first = plot_setup["first"]
    interval = plot_setup["interval"]

    gvals = data[:, 4][first::interval]
    alphavals = data[:, 5][first::interval]
    indexes = np.arange(data.shape[0])[first::interval]

    x_data = alphavals if plot_setup["use_alpha"] else gvals
    curve_label = r"$\alpha$" if plot_setup["use_alpha"] else r"$g$"

    nprime = wc / w0
    ohmic_model = artifact["SIM"]["SB_parmas"]["ohmic_model"]
    wmin = w_min(ohmic_model, w0=w0, wc=wc)

    fig1, ax1 = plt.subplots(figsize=[10, 8])
    ax1.set_title(
        r"$Ground\;state\;photons \quad (n^\star=%.2f)$" % (nprime), fontsize=18
    )
    ax1.set_xlabel(r"$n/n^\star$", fontsize=18)
    ax1.set_ylabel(r"$\langle a_n^\dagger a_n \rangle_{GS}$", fontsize=18)
    ax1.set_xlim(1 / nprime - 0.001, Nk // 2 / nprime)

    ax1.text(
        1,
        0.1,
        r"$n^\star = \frac{\omega_c}{\omega_0}$",
        transform=transforms.blended_transform_factory(ax1.transData, ax1.transAxes),
        fontsize=12,
    )
    ax1.text(
        0.4,
        0.9,
        (
            rf"$\Delta = {delta:.3f}$"
            "\n"
            rf"$\omega_c = {wc:.3f}$"
            "\n"
            rf"$\omega_{{\min}} = {wmin:.3f}$"
        ),
        transform=transforms.blended_transform_factory(ax1.transAxes, ax1.transAxes),
        fontsize=12,
    )
    cmap = plt.colormaps["inferno"](np.linspace(0, 1, len(gvals)))
    y_min = 1

    field_paths = artifact["results"]["field"]["x"]
    field_paths = [field_paths[i] for i in indexes]

    for i, (coupling, field_path) in enumerate(zip(x_data, field_paths)):

        Nx = np.loadtxt(
            field_path,
            dtype=complex,
        ).real

        Nxhalf = Nx[Nk // 2 :]
        n_values = np.arange(len(Nxhalf))
        if min(Nxhalf) < y_min:
            y_min = min(Nxhalf)

        # TEORIC
        if plot_setup["plot_theo"]:
            Nteo = (
                coupling**2
                / (np.pi**2 * w0**2 * nprime**2)
                * (scipy.special.k1(n_values / nprime) / n_values) ** 2
            )
            ax1.plot(np.arange(0, len(n_values)) / nprime, Nteo, color=cmap[i])

        # MPS RESULTS
        ax1.plot(
            np.arange(len(Nxhalf)) / nprime,
            Nxhalf,
            color=cmap[i],
            marker="o",
            ls="",
            ms=3,
            label=rf"${curve_label}={coupling}$",
        )

    ax1.set_ylim(y_min, 1e1)
    ax1.vlines(1, y_min, 1e1, color="grey", ls="--")
    ax1.legend()
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.grid()
    fig1.savefig(
        write_folder + f"Nx_RealSpace_occupation_{sim_label}.{output_format}",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig1)
