"""Diagnostic plots for ordinary least-squares models."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from statsmodels.graphics.gofplots import qqplot
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.linear_model import OLS, OLSResults, RegressionResultsWrapper


def _add_trend(ax: Axes, fitted: np.ndarray, values: np.ndarray) -> None:
    # Grouped models such as ANOVA have repeated fitted values.
    distinct, groups, counts = np.unique(
        fitted, return_inverse=True, return_counts=True
    )
    if distinct.size < 2:
        return

    means = np.bincount(groups, weights=values) / counts
    if distinct.size >= 4:
        means = lowess(means, distinct, is_sorted=True, return_sorted=False)
    ax.plot(distinct, means, color="C1", linewidth=1.5)


def dgplots(result: OLSResults | RegressionResultsWrapper) -> Figure:
    """Return four diagnostic plots for a fitted statsmodels OLS model.

    Use the result of calling ``.fit()`` on a statsmodels OLS model.
    The panels show residuals versus predicted values, a normal Q-Q plot,
    location-scale, and Cook's D by observation. The Q-Q and location-scale
    plots use standardised residuals.

    Orange lines show the trend across fitted values. The trend uses the
    mean within each distinct fitted value, with LOWESS smoothing when
    there are at least four distinct values. Cook's D labels use
    the fitted rows' index labels when available.
    """
    if not isinstance(getattr(result, "model", None), OLS):
        raise TypeError("Pass the fitted result of a statsmodels OLS model.")

    residual_variance = result.mse_resid
    if (
        result.df_resid <= 0
        or not np.isfinite(residual_variance)
        or residual_variance <= 0
    ):
        raise ValueError(
            "Diagnostic plots require positive residual variance and degrees of freedom."
        )

    fitted = np.asarray(result.fittedvalues)
    residuals = np.asarray(result.resid)
    influence = result.get_influence()
    with np.errstate(divide="ignore", invalid="ignore"):
        standardised_residuals = influence.resid_studentized_internal
        cooks = influence.cooks_distance[0]

    valid = (
        (1 - influence.hat_matrix_diag > 8 * np.finfo(float).eps)
        & np.isfinite(standardised_residuals)
        & np.isfinite(cooks)
    )
    if not np.any(valid):
        raise ValueError(
            "Standardised residuals and Cook's D are undefined for this fit."
        )
    omitted = np.count_nonzero(~valid)
    if omitted:
        warnings.warn(
            f"Excluded {omitted} fitted observation(s) with undefined influence "
            "from the Q-Q, location-scale, and Cook's D plots.",
            RuntimeWarning,
            stacklevel=2,
        )

    fig, ((resid_ax, qq_ax), (scale_ax, cook_ax)) = plt.subplots(
        2, 2, figsize=(9, 6.5), layout="constrained"
    )

    resid_ax.scatter(fitted, residuals, s=20, color="C0", alpha=0.75)
    resid_ax.axhline(0, color="0.45", linestyle="--", linewidth=1)
    _add_trend(resid_ax, fitted, residuals)
    resid_ax.set(title="Residual Plot", xlabel="Predicted Values", ylabel="Residuals")

    qqplot(standardised_residuals[valid], line="q", ax=qq_ax)
    qq_ax.set(
        title="Q-Q Plot",
        xlabel="Theoretical Quantiles",
        ylabel="Sample Quantiles",
    )

    spread = np.sqrt(np.abs(standardised_residuals[valid]))
    scale_ax.scatter(fitted[valid], spread, s=20, color="C0", alpha=0.75)
    _add_trend(scale_ax, fitted[valid], spread)
    scale_ax.set(
        title="Location-Scale Plot",
        xlabel="Predicted Values",
        ylabel=r"$\sqrt{|\mathrm{Standardised\ Residuals}|}$",
    )

    positions = np.arange(1, len(cooks) + 1)
    cook_ax.vlines(positions[valid], 0, cooks[valid], color="C0", alpha=0.7)
    cook_ax.scatter(positions[valid], cooks[valid], s=14, color="C0")
    for level in (0.5, 1.0):
        cook_ax.axhline(level, color="0.5", linestyle="--", linewidth=1, label=str(level))
    cook_ax.legend(title="Reference", frameon=False)
    cook_ax.set(
        title="Cook's D versus Index Plot",
        xlabel="Observation Number",
        ylabel="Cook's D",
        ylim=(0, None),
    )

    labels = result.model.data.row_labels
    valid_rows = np.flatnonzero(valid)
    largest = valid_rows[np.argsort(-cooks[valid], kind="stable")[:3]]
    for row in largest:
        label = labels[row] if labels is not None else positions[row]
        cook_ax.annotate(
            str(label),
            (positions[row], cooks[row]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize="small",
        )

    # The inline backend also displays registered figures at the end of a cell.
    if plt.get_backend().lower() in {
        "inline",
        "module://ipykernel.pylab.backend_inline",
        "module://matplotlib_inline.backend_inline",
    }:
        plt.close(fig)

    return fig
