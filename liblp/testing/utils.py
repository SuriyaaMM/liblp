import itertools
from collections.abc import Callable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from liblp.testing.config import __get_normal_dist_config
from torch import Tensor

from liblp.lptensor import lptensor
from liblp.lpdtype import lpdtype

def __plot(
    df: pd.DataFrame,
    column: str,
    aggregate: str,
    title: str,
):
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=1.2,
    )

    g = sns.relplot(
        data=df,
        x=aggregate,
        y=column,
        hue="var",
        col="n",
        col_wrap=3,
        kind="line",
        palette="coolwarm",
        marker="o",
        markersize=7,
        linewidth=2,
        height=6,
        aspect=1.5,
    )

    g.set(yscale="symlog")

    for ax in g.axes.flat:
        ax.set_xlabel(
            aggregate,
            fontsize=15,
            fontweight="bold",
        )
        ax.set_ylabel(
            f"{column} (log scale)",
            fontsize=15,
            fontweight="bold",
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=12,
            width=1.2,
            length=6,
        )

        ax.grid(
            True,
            which="major",
            linewidth=0.8,
            alpha=0.5,
        )
        ax.grid(
            True,
            which="minor",
            linewidth=0.4,
            alpha=0.25,
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)

    g.set_titles(
        col_template="n = {col_name}",
        size=15,
        weight="bold",
    )

    g.figure.suptitle(
        title,
        fontsize=20,
        fontweight="bold",
    )

    g.figure.subplots_adjust(
        top=0.90,
        bottom=0.12,
        left=0.10,
        right=0.90,
        hspace=0.30,
        wspace=0.20,
    )

    g.figure.savefig(
        f"{title}.pdf",
        bbox_inches="tight",
    )

    plt.close(g.figure)

    n_values = sorted(df["n"].unique())

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11),
        constrained_layout=True,
    )

    axes = axes.flat

    for ax, n in zip(axes, n_values):
        subset = df[df["n"] == n]

        heatmap_data = subset.pivot(
            index="mean",
            columns="var",
            values=column,
        )

        sns.heatmap(
            heatmap_data,
            ax=ax,
            annot=True,
            fmt=".3g",
            cmap="coolwarm",
            cbar=True,
        )

        ax.set_title(f"n = {n}")
        ax.set_xlabel("Variance")
        ax.set_ylabel("Mean")

    for ax in axes[len(n_values) :]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=18)

    fig.savefig(
        f"{title}_heatmap.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def __plot_summary(df: pd.DataFrame, title: str):
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=1.2,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 11),
    )

    ax1, ax2, ax3, ax4 = axes.flat

    summary = (
        df.groupby("n")["mean_relative_error"]
        .agg(["mean", "median", "max"])
        .reset_index()
    )

    ax1.plot(
        summary["n"],
        summary["mean"],
        marker="o",
        linewidth=2,
        label="Mean",
    )

    ax1.plot(
        summary["n"],
        summary["median"],
        marker="s",
        linewidth=2,
        label="Median",
    )

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("n", fontsize=13, fontweight="bold")
    ax1.set_ylabel(
        "Mean relative error",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_title(
        "Relative Error vs Problem Size",
        fontsize=15,
        fontweight="bold",
    )
    ax1.legend()

    q95 = df.groupby("n")["mean_absolute_error"].quantile(0.95).reset_index()

    max_error = df.groupby("n")["max_absolute_error"].max().reset_index()

    ax2.plot(
        q95["n"],
        q95["mean_absolute_error"],
        marker="o",
        linewidth=2,
        label="95th percentile",
    )

    ax2.plot(
        max_error["n"],
        max_error["max_absolute_error"],
        marker="^",
        linewidth=2,
        label="Worst case",
    )

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")

    ax2.set_xlabel("n", fontsize=13, fontweight="bold")
    ax2.set_ylabel(
        "Absolute error",
        fontsize=13,
        fontweight="bold",
    )

    ax2.set_title(
        "Typical vs Worst-Case Error",
        fontsize=15,
        fontweight="bold",
    )

    ax2.legend()

    heatmap = (
        df.groupby(["mean", "var"])["mean_relative_error"].mean().unstack().sort_index()
    )

    sns.heatmap(
        heatmap,
        annot=True,
        fmt=".3f",
        linewidths=0.6,
        ax=ax3,
        cbar_kws={"label": "Mean relative error"},
        cmap="rocket_r",
    )

    ax3.set_xlabel(
        "var",
        fontsize=13,
        fontweight="bold",
    )

    ax3.set_ylabel(
        "mean",
        fontsize=13,
        fontweight="bold",
    )

    ax3.set_title(
        "Relative Error: Mean vs Var",
        fontsize=15,
        fontweight="bold",
    )

    values = df.loc[
        df["mean_relative_error"] > 0,
        "mean_relative_error",
    ].dropna()

    values = np.sort(values)

    cdf = np.arange(1, len(values) + 1) / len(values)

    ax4.plot(
        values,
        cdf,
        linewidth=2,
    )

    ax4.set_xscale("log")

    ax4.set_xlabel(
        "Mean relative error",
        fontsize=13,
        fontweight="bold",
    )

    ax4.set_ylabel(
        "Cumulative fraction",
        fontsize=13,
        fontweight="bold",
    )

    ax4.set_title(
        "Distribution of Relative Error",
        fontsize=15,
        fontweight="bold",
    )

    ax4.axhline(
        0.95,
        linestyle="--",
        linewidth=1,
    )

    for ax in axes.flat:
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=11,
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

        ax.grid(
            True,
            which="major",
            linewidth=0.7,
            alpha=0.45,
        )

    fig.suptitle(
        title,
        fontsize=20,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(
        f"{title}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

def __gen_input_np(n, var, mean):
    a = var * np.random.normal(size=(n,)) + mean
    b = var * np.random.normal(size=(n,)) + mean
    return (a, b)

def __gen_input_torch(n, var, mean, dtype):
    a, b = __gen_input_np(n, var, mean)
    return (torch.from_numpy(a).to(dtype), torch.from_numpy(b).to(dtype))

def __get_errors_np(c_ref, c_lp_ref_dtype):
    absolute_error = np.abs(c_ref - c_lp_ref_dtype)
    relative_error = np.abs(c_ref - c_lp_ref_dtype) / (
        np.abs(c_ref) + np.finfo(np.float32).eps
    )
    return (absolute_error, relative_error)

def __get_errors_torch(c_ref, c_lp_ref_dtype):
    absolute_error = torch.abs(c_ref - c_lp_ref_dtype)
    relative_error = torch.abs(c_ref - c_lp_ref_dtype) / (
        torch.abs(c_ref) + torch.finfo(torch.float32).eps
    )
    return (absolute_error, relative_error)

def __test_one(
    distribution: Literal["normal", "exponential"],
    parameters: tuple,
    operation_ref: Callable[[Tensor | np.ndarray, Tensor | np.ndarray], Tensor],
    operation_lp: Callable[[lptensor, lptensor], lptensor],
):
    if distribution == "normal":
        dtype = parameters[0]
        lpadtype = parameters[1]
        device = parameters[2]
        n = parameters[3]
        mean = parameters[4]
        var = parameters[5]

        if device == "cuda":
            a, b = __gen_input_torch(n, var, mean, dtype)
            a = a.to(device)
            b = b.to(device)
        elif device == "cpu":
            a, b = __gen_input_np(n, var, mean)
            a = a.astype(dtype)
            b = b.astype(dtype)
        else:
            raise NotImplementedError(f"test not implemented for device = {device}")
        
        a_lpa = lptensor(data=a, dtype=lpadtype)
        b_lpa = lptensor(data=b, dtype=lpadtype)

        c_ref = operation_ref(a, b)
        c_lpa = operation_lp(a_lpa, b_lpa)

        c_lp_ref_dtype = c_lpa.get()

        
        if device == "cuda":
            absolute_error, relative_error = __get_errors_torch(c_ref, c_lp_ref_dtype)
            absolute_error = absolute_error.cpu().numpy()
            relative_error = relative_error.cpu().numpy()
        elif device == "cpu":
            absolute_error, relative_error = __get_errors_np(c_ref, c_lp_ref_dtype)
        else:
            raise NotImplementedError(f"test not implemented for device = {device}")

        return (
            float(absolute_error.mean().item()),
            float(np.median(absolute_error).item()),
            float(absolute_error.std().item()),
            float(absolute_error.max().item()),
            float(relative_error.mean().item()),
            float(np.median(relative_error).item()),
            float(relative_error.std().item()),
            float(relative_error.max().item()),
        )

    else:
        raise NotImplementedError(
            f"test one for distribution = {distribution} isn't implemented yet!"
        )


def __test_normal_distribution(
    dtype: torch.dtype,
    lpdtype: lpdtype,
    device: str,
    operation_ref: Callable[[Tensor, Tensor], Tensor],
    operation_lp: Callable[[lptensor, lptensor], Tensor],
    savefile_base: str,
):
    n, mean, var = __get_normal_dist_config()

    distribution_n = []
    distribution_mean = []
    distribution_var = []

    mean_absolute_errors = []
    std_absolute_errors = []
    median_absolute_errors = []
    max_absolute_errors = []

    mean_relative_errors = []
    std_relative_errors = []
    median_relative_errors = []
    max_relative_errors = []

    for p1, p2, p3 in itertools.product(n, mean, var):
        distribution_n.append(p1)
        distribution_mean.append(p2)
        distribution_var.append(p3)
        (
            absolute_error_mean,
            absolute_error_median,
            absolute_error_std,
            absolute_error_max,
            relative_error_mean,
            relative_error_median,
            relative_error_std,
            relative_error_max,
        ) = __test_one(
            distribution="normal",
            parameters=(dtype, lpdtype, device, p1, p2, p3),
            operation_ref=operation_ref,
            operation_lp=operation_lp,
        )

        mean_absolute_errors.append(absolute_error_mean)
        median_absolute_errors.append(absolute_error_median)
        std_absolute_errors.append(absolute_error_std)
        max_absolute_errors.append(absolute_error_max)

        mean_relative_errors.append(relative_error_mean)
        median_relative_errors.append(relative_error_median)
        std_relative_errors.append(relative_error_std)
        max_relative_errors.append(relative_error_max)

    df = pd.DataFrame(
        {
            "n": distribution_n,
            "mean": distribution_mean,
            "var": distribution_var,
            "mean_absolute_error": mean_absolute_errors,
            "median_absolute_error": median_absolute_errors,
            "std_absolute_error": std_absolute_errors,
            "max_absolute_error": max_absolute_errors,
            "mean_relative_error": mean_relative_errors,
            "median_relative_error": median_relative_errors,
            "std_relative_error": std_relative_errors,
            "max_relative_error": max_relative_errors,
        }
    )

    print(df.to_markdown())
    df.to_csv(savefile_base + ".csv")

    __plot(
        df=df,
        column="mean_relative_error",
        aggregate="mean",
        title=savefile_base + "_mean_relative_error",
    )
    __plot(
        df=df,
        column="max_relative_error",
        aggregate="mean",
        title=savefile_base + "_max_relative_error",
    )
    __plot(
        df=df,
        column="mean_absolute_error",
        aggregate="mean",
        title=savefile_base + "_mean_absolute_error",
    )
    __plot(
        df=df,
        column="max_absolute_error",
        aggregate="mean",
        title=savefile_base + "_max_relative_error",
    )
    __plot_summary(df=df, title=savefile_base + "_summary")
