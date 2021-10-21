from sys import stderr
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats

from pathlib import Path

from matplotlib import pyplot
from matplotlib.colors import LinearSegmentedColormap


color_list = ["#ca0020", "#f4a582", "#f7f7f7", "#92c5de", "#0571b0"]
PALETTE = LinearSegmentedColormap.from_list(
    "nothing", color_list[::-1], N=len(color_list)
)
CLIP_LIMITS = {
    r"CATE $r_e$": (-2, 2),
    r"CATE $CF_w$": (-0.05, 0.05),
    r"CATE $\tau$": (-5.5, 5.5),
    r"CATE $LWP$": (-40, 40),
}
DATA_LIMITS = {
    "w500": (-0.6, 0.6),
    "EIS": (-15, 27),
    "RH900": (-0.1, 1.2),
    "RH850": (-0.1, 1.2),
    "RH700": (-0.1, 1.2),
    "whoi_sst": (270, 305),
}
_ = sns.set(style="darkgrid", palette=sns.color_palette(color_list[::-1]))
pyplot.rcParams.update(
    {
        "figure.constrained_layout.use": True,
        "figure.facecolor": "white",
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.frameon": False,
    }
)


def plot(
    csv_path, output_dir,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = pd.read_csv(csv_path, index_col=0)

    ate = dataframe[[c for c in dataframe.columns if "CATE l_re" in c]].mean(0).mean()
    ate_std = (
        dataframe[[c for c in dataframe.columns if "CATE l_re" in c]].mean(0).std()
    )
    print(f"ATE l_re: {ate:.03f}\pm{ate_std:.03f}")

    ate = dataframe[[c for c in dataframe.columns if "CATE liq_pc" in c]].mean(0).mean()
    ate_std = (
        dataframe[[c for c in dataframe.columns if "CATE liq_pc" in c]].mean(0).std()
    )
    print(f"ATE liq_pc: {ate:.03f}\pm{ate_std:.03f}")

    ate = dataframe[[c for c in dataframe.columns if "CATE cod" in c]].mean(0).mean()
    ate_std = dataframe[[c for c in dataframe.columns if "CATE cod" in c]].mean(0).std()
    print(f"ATE cod: {ate:.03f}\pm{ate_std:.03f}")

    ate = dataframe[[c for c in dataframe.columns if "CATE cwp" in c]].mean(0).mean()
    ate_std = dataframe[[c for c in dataframe.columns if "CATE cwp" in c]].mean(0).std()
    print(f"ATE cwp: {ate:.03f}\pm{ate_std:.03f}")

    effect_histograms(dataframe=dataframe, output_dir=output_dir)
    effect_heatmaps(dataframe=dataframe, output_dir=output_dir)
    outcome_scatterplots(dataframe=dataframe, output_dir=output_dir)


def outcome_scatterplots(dataframe, output_dir):
    text_color = ".15"
    facecolor = "#EAEAF2"
    gridcolor = "white"

    pyplot.rcParams.update(
        {
            "axes.facecolor": facecolor,
            "axes.labelcolor": text_color,
            "axes.edgecolor": gridcolor,
            "grid.color": gridcolor,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": text_color,
            "ytick.color": text_color,
        }
    )
    fig = pyplot.figure(figsize=(12, 3))

    ax = pyplot.subplot(1, 4, 1)
    g = sns.scatterplot(
        data=dataframe, x=r"Observed $CF_w$", y=r"Predicted $CF_w$", s=0.5
    )
    slope, intercept, r, p, stderr = stats.linregress(
        dataframe["Observed $CF_w$"], dataframe["Predicted $CF_w$"]
    )
    domain = np.arange(-0.05, 1.05, 0.01)
    _ = pyplot.plot(domain, domain * slope + intercept)
    _ = g.set_title(r"$CF_w$: " + f"$R^2$={r ** 2:.02f}")
    _ = g.set(ylabel="Predicted")
    _ = g.set(xlabel="Observed")
    _ = g.set_ylim(-0.05, 1.05)
    _ = g.set_xlim(-0.05, 1.05)

    ax = pyplot.subplot(1, 4, 2)
    g = sns.scatterplot(
        data=dataframe, x=r"Observed $r_e$", y=r"Predicted $r_e$", s=0.5
    )
    slope, intercept, r, p, stderr = stats.linregress(
        dataframe["Observed $r_e$"], dataframe["Predicted $r_e$"]
    )
    domain = np.arange(3, 32, 1)
    _ = pyplot.plot(domain, domain * slope + intercept)
    _ = g.set_title(r"$r_e$: " + f"$R^2$={r ** 2:.02f}")
    _ = g.set(ylabel=None)
    _ = g.set(xlabel="Observed")
    _ = g.set_ylim(3, 32)
    _ = g.set_xlim(3, 32)

    ax = pyplot.subplot(1, 4, 3)
    g = sns.scatterplot(
        data=dataframe, x=r"Observed $\tau$", y=r"Predicted $\tau$", s=0.5
    )
    slope, intercept, r, p, stderr = stats.linregress(
        dataframe[r"Observed $\tau$"], dataframe[r"Predicted $\tau$"]
    )
    domain = np.arange(-10, 100, 1)
    _ = pyplot.plot(domain, domain * slope + intercept)
    _ = g.set_title(r"$\tau$: " + f"$R^2$={r ** 2:.02f}")
    _ = g.set(ylabel=None)
    _ = g.set(xlabel="Observed")
    _ = g.set_ylim(-10, 100)
    _ = g.set_xlim(-10, 100)

    ax = pyplot.subplot(1, 4, 4)
    g = sns.scatterplot(
        data=dataframe, x=r"Observed $LWP$", y=r"Predicted $LWP$", s=0.5
    )
    slope, intercept, r, p, stderr = stats.linregress(
        dataframe["Observed $LWP$"], dataframe["Predicted $LWP$"]
    )
    domain = np.arange(-100, 1100, 1)
    _ = pyplot.plot(domain, domain * slope + intercept)
    _ = g.set_title(r"$LWP$: " + f"$R^2$={r ** 2:.02f}")
    _ = g.set(ylabel=None)
    _ = g.set(xlabel="Observed")
    _ = g.set_ylim(-100, 1100)
    _ = g.set_xlim(-100, 1100)

    _ = pyplot.savefig(output_dir / "scatterplots.png", dpi=150)


def effect_heatmaps(
    dataframe, output_dir,
):
    text_color = ".15"
    facecolor = "#dfdfe6"
    gridcolor = "white"

    params = {
        "axes.facecolor": facecolor,
        "axes.labelcolor": text_color,
        "axes.edgecolor": gridcolor,
        "grid.color": gridcolor,
        "xtick.color": text_color,
        "ytick.color": text_color,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 18,
        "legend.title_fontsize": 18,
    }
    pyplot.rcParams.update(params)

    fig = pyplot.figure(figsize=(12, 9))
    covars = ["w500", "EIS", "RH700", "RH850", "whoi_sst"]
    for i, x_var in enumerate(covars):
        covars = covars[1:]
        for j, y_var in enumerate(covars):
            ax = pyplot.subplot(4, 4, 4 * j + i + 1 + 4 * i)
            g = plot_heatmap(
                dataframe=dataframe,
                x_var=x_var,
                y_var=y_var,
                hue_var=r"CATE $r_e$",
                s_var=r"CATE Uncertainty $r_e$",
            )
            g.legend_.remove()
            if i != 0:
                g.set(yticklabels=[])
                g.set(ylabel=None)
            if j != len(covars) - 1:
                g.set(xticklabels=[])
                g.set(xlabel=None)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    legend = fig.legend(
        lines_labels[0][0],
        lines_labels[0][1],
        loc="upper right",
        title="Effect of AOD on $r_e$",
    )
    for text in legend.get_texts():
        text.set_color(text_color)
    legend._legend_title_box._text.set_color(text_color)
    for lh in legend.legendHandles:
        lh._sizes = [100]

    _ = pyplot.savefig(output_dir / "heatmaps_lre.png", dpi=150)

    fig = pyplot.figure(figsize=(12, 9))
    covars = ["w500", "EIS", "RH700", "RH850", "whoi_sst"]
    for i, x_var in enumerate(covars):
        covars = covars[1:]
        for j, y_var in enumerate(covars):
            ax = pyplot.subplot(4, 4, 4 * j + i + 1 + 4 * i)
            g = plot_heatmap(
                dataframe=dataframe,
                x_var=x_var,
                y_var=y_var,
                hue_var=r"CATE $\tau$",
                s_var=r"CATE Uncertainty $\tau$",
            )
            g.legend_.remove()
            if i != 0:
                g.set(yticklabels=[])
                g.set(ylabel=None)
            if j != len(covars) - 1:
                g.set(xticklabels=[])
                g.set(xlabel=None)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    legend = fig.legend(
        lines_labels[0][0],
        lines_labels[0][1],
        loc="upper right",
        title=r"Effect of AOD on $\tau$",
    )
    for text in legend.get_texts():
        text.set_color(text_color)
    legend._legend_title_box._text.set_color(text_color)
    for lh in legend.legendHandles:
        lh._sizes = [100]

    _ = pyplot.savefig(output_dir / "heatmaps_cod.png", dpi=150)

    fig = pyplot.figure(figsize=(12, 9))
    covars = ["w500", "EIS", "RH700", "RH850", "whoi_sst"]
    for i, x_var in enumerate(covars):
        covars = covars[1:]
        for j, y_var in enumerate(covars):
            ax = pyplot.subplot(4, 4, 4 * j + i + 1 + 4 * i)
            g = plot_heatmap(
                dataframe=dataframe,
                x_var=x_var,
                y_var=y_var,
                hue_var=r"CATE $LWP$",
                s_var=r"CATE Uncertainty $LWP$",
            )
            g.legend_.remove()
            if i != 0:
                g.set(yticklabels=[])
                g.set(ylabel=None)
            if j != len(covars) - 1:
                g.set(xticklabels=[])
                g.set(xlabel=None)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    legend = fig.legend(
        lines_labels[0][0],
        lines_labels[0][1],
        loc="upper right",
        title=r"Effect of AOD on $LWP$",
    )
    for text in legend.get_texts():
        text.set_color(text_color)
    legend._legend_title_box._text.set_color(text_color)
    for lh in legend.legendHandles:
        lh._sizes = [100]

    _ = pyplot.savefig(output_dir / "heatmaps_lwp.png", dpi=150)

    fig = pyplot.figure(figsize=(12, 9))
    covars = ["w500", "EIS", "RH700", "RH850", "whoi_sst"]
    for i, x_var in enumerate(covars):
        covars = covars[1:]
        for j, y_var in enumerate(covars):
            ax = pyplot.subplot(4, 4, 4 * j + i + 1 + 4 * i)
            g = plot_heatmap(
                dataframe=dataframe,
                x_var=x_var,
                y_var=y_var,
                hue_var=r"CATE $CF_w$",
                s_var=r"CATE Uncertainty $CF_w$",
            )
            g.legend_.remove()
            if i != 0:
                g.set(yticklabels=[])
                g.set(ylabel=None)
            if j != len(covars) - 1:
                g.set(xticklabels=[])
                g.set(xlabel=None)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    legend = fig.legend(
        lines_labels[0][0],
        lines_labels[0][1],
        loc="upper right",
        title=r"Effect of AOD on $CF_w$",
    )
    for text in legend.get_texts():
        text.set_color(text_color)
    legend._legend_title_box._text.set_color(text_color)
    for lh in legend.legendHandles:
        lh._sizes = [100]

    _ = pyplot.savefig(output_dir / "heatmaps_cf.png", dpi=150)


def plot_heatmap(
    dataframe, x_var, y_var, hue_var, s_var,
):
    x = dataframe[x_var]
    y = dataframe[y_var]
    s = dataframe[s_var]
    s = s / s.max()
    hue = dataframe[hue_var].clip(CLIP_LIMITS[hue_var][0], CLIP_LIMITS[hue_var][1])
    g = sns.scatterplot(
        x=x_var, y=y_var, hue=hue, s=10 * s, data=dataframe, palette=PALETTE,
    )
    _ = pyplot.xlim(DATA_LIMITS[x_var])
    _ = pyplot.ylim(DATA_LIMITS[y_var])
    return g


def effect_histograms(
    dataframe, output_dir,
):
    figsize = (12, 2.25)
    text_color = ".15"
    facecolor = "#dfdfe6"
    gridcolor = "white"

    params = {
        "axes.facecolor": facecolor,
        "axes.labelcolor": text_color,
        "axes.edgecolor": gridcolor,
        "grid.color": gridcolor,
        "xtick.color": text_color,
        "ytick.color": text_color,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
    }
    pyplot.rcParams.update(params)
    covars = ["w500", "EIS", "RH700", "RH850", "whoi_sst"]

    df_grid = dataframe[covars + [r"CATE $r_e$"]]
    df_grid[r"CATE $r_e$"] = pd.cut(
        df_grid[r"CATE $r_e$"],
        bins=[-2.0, -1.2, -0.4, 0.4, 1.2, 2.0],
        labels=[-1.6, -0.8, 0.0, 0.8, 1.6],
    )
    hue_order = [
        1.6,
        0.8,
        0.0,
        -0.8,
        -1.6,
    ]

    fig = pyplot.figure(figsize=figsize)
    for i, x_var in enumerate(covars):
        ax = pyplot.subplot(1, 5, i + 1)
        g = plot_kde(
            x=x_var,
            hue=r"CATE $r_e$",
            dataframe=df_grid,
            hue_order=hue_order,
            palette=sns.color_palette(color_list),
        )
        if i != 0:
            g.set(ylabel=None)
            g.legend_.remove()
    _ = pyplot.savefig(output_dir / "histograms_lre.png", dpi=150)

    df_grid = dataframe[covars + [r"CATE $CF_w$"]]
    df_grid[r"CATE $CF_w$"] = pd.cut(
        df_grid[r"CATE $CF_w$"],
        bins=[-0.05, -0.03, -0.01, 0.01, 0.03, 0.05],
        labels=[-0.04, -0.02, 0.00, 0.02, 0.04],
    )
    hue_order = [
        -0.04,
        -0.02,
        0.00,
        0.02,
        0.04,
    ]

    fig = pyplot.figure(figsize=figsize)
    for i, x_var in enumerate(covars):
        ax = pyplot.subplot(1, 5, i + 1)
        g = plot_kde(
            x=x_var,
            hue=r"CATE $CF_w$",
            dataframe=df_grid,
            hue_order=hue_order,
            palette=sns.color_palette(color_list[::-1]),
        )
        if i != 0:
            g.set(ylabel=None)
            g.legend_.remove()
    _ = pyplot.savefig(output_dir / "histograms_cf.png", dpi=150)

    df_grid = dataframe[covars + [r"CATE $\tau$"]]
    df_grid[r"CATE $\tau$"] = pd.cut(
        df_grid[r"CATE $\tau$"],
        bins=[-5.0, -3.0, -1.0, 1.0, 3.0, 5.0],
        labels=[-4.0, -2.0, 0.0, 2.0, 4.0],
    )
    hue_order = [
        -4.0,
        -2.0,
        0.0,
        2.0,
        4.0,
    ]

    fig = pyplot.figure(figsize=figsize)
    for i, x_var in enumerate(covars):
        ax = pyplot.subplot(1, 5, i + 1)
        g = plot_kde(
            x=x_var,
            hue=r"CATE $\tau$",
            dataframe=df_grid,
            hue_order=hue_order,
            palette=sns.color_palette(color_list[::-1]),
        )
        if i != 0:
            g.set(ylabel=None)
            g.legend_.remove()
    _ = pyplot.savefig(output_dir / "histograms_cod.png", dpi=150)

    df_grid = dataframe[covars + [r"CATE $LWP$"]]
    df_grid[r"CATE $LWP$"] = pd.cut(
        df_grid[r"CATE $LWP$"],
        bins=[-37.5, -22.5, -7.5, 7.5, 22.5, 37.5],
        labels=[-30, -15, 0, 15, 30],
    )
    hue_order = [
        -30,
        -15,
        0,
        15,
        30,
    ]

    fig = pyplot.figure(figsize=figsize)
    for i, x_var in enumerate(covars):
        ax = pyplot.subplot(1, 5, i + 1)
        g = plot_kde(
            x=x_var,
            hue=r"CATE $LWP$",
            dataframe=df_grid,
            hue_order=hue_order,
            palette=sns.color_palette(color_list[::-1]),
        )
        if i != 0:
            g.set(ylabel=None)
            g.legend_.remove()
    _ = pyplot.savefig(output_dir / "histograms_lwp.png", dpi=150)


def plot_kde(
    x, hue, dataframe, hue_order, palette,
):
    g = sns.histplot(
        x=x,
        hue=hue,
        data=dataframe,
        hue_order=hue_order,
        palette=palette,
        fill=True,
        alpha=0.3,
    )
    return g
