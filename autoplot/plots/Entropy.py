import matplotlib.pyplot as plt


def entropyVScoupling(data, artifact, plot_setup, write_folder, output_format="png"):

    sim_label = artifact["labels"]

    # ENERGY
    gvals = data[:, 4]
    alphavals = data[:, 5]
    Sbond = data[:, 10]

    x_data = alphavals if plot_setup["use_alpha"] else gvals
    coup_label = r"$\alpha$" if plot_setup["use_alpha"] else r"$g$"

    fig, ax = plt.subplots(figsize=[10, 8])
    ax.set_title(rf"$Energy\;vs.\;{coup_label} \quad (n^\star=%.2f)$", fontsize=18)
    ax.set_xlabel(rf"${coup_label}$", fontsize=18)
    ax.set_ylabel(r"$\langle H \rangle_{\GS}$", fontsize=18)
    ax.set_ylim(min(Sbond) + min(Sbond) / 10, max(Sbond) - max(Sbond))
    ax.grid()
    ax.plot(x_data, Sbond)

    fig.savefig(
        write_folder + f"Energy_{sim_label}.{output_format}",
        dpi=600,
        bbox_inches="tight",
    )
