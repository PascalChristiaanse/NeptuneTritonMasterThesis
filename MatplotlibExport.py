"""MatplotlibExport.py — Publication-quality PDF figure export for thesis.

Reads the same .pkl files as DashInteractivePlotFull.py (by importing its
data-loading logic) and saves selected plots as a multi-page PDF.

QUICK-START
-----------
1. Set DATASET_LABEL to the dataset key you want (or leave None for first).
2. Set SELECTED_SIMS and SIM_LABELS (or leave None to use all).
3. Edit FIGURES_TO_EXPORT to pick which figures to include.
4. Run:  python MatplotlibExport.py
   → saves OUTPUT_PDF in the working directory.

AVAILABLE FIGURE TYPES
-----------------------
  'rms_compare'        — bar chart of final RMS vs SPICE per simulation
  'rsw_compare'        — 3-row RSW difference vs SPICE time series
  'formal_compare'     — 3-row formal errors (σ_R/S/W) time series
  'rsw_stats'          — 3×3 grid: rows=R/S/W, cols=Mean/RMS/Max
  'gof'                — WRMS/RMS/cost comparison, optional initial overlay
  'corr_heatmap'       — |correlation| matrix heatmap (one sim at a time)
  'residual_histogram' — RA/Dec residual histograms with optional Gaussian fit
"""

import sys
import os
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('PDF')           # must come before any pyplot import
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy import stats as sp_stats

# ── Import data-processing functions from the Dash script ───────────────────
# This also triggers module-level data loading (DATA_FILES → all_datasets).
# app.run() is guarded by __name__ == '__main__', so no server is started.
from DashInteractivePlotFull import (
    all_datasets,
    get_active_data,
    convert_time_array_to_datetime,
    get_rsw_times,
    compute_rsw_statistics,
    compute_formal_error_statistics,
    compute_wrms_and_cost,
    get_parameter_labels,
    get_parameter_info,
)

# ============================================================================
# GLOBAL STYLE  (edit once — inherited by all figures)
# ============================================================================

plt.rcParams.update({
    'font.size':         11,
    'axes.titlesize':    12,
    'axes.labelsize':    11,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    # Uncomment the two lines below if LaTeX is installed:
    # 'text.usetex':     True,
    # 'font.family':     'serif',
    'font.family':       'sans-serif',
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'lines.linewidth':   1.2,
    'lines.markersize':  4,
})

# Figure widths (inches).  3.5" = single column, 6.5" = double column.
FIG_W_SINGLE  = 3.5
FIG_W_DOUBLE  = 6.5
FIG_H_DEFAULT = 4.5

# Named RSW colors for consistency across plots.
RSW_COLORS = {'R': '#1f77b4', 'S': '#d62728', 'W': '#2ca02c'}

# Fall-back color cycle (matplotlib default).
_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Per-simulation colors and linestyles — loaded from the active config.
# Falls back to empty dicts so unknown sim names use the default color cycle.


# ============================================================================
# CONFIGURATION  — set ACTIVE_CONFIG to the desired ExportConfigs/*.py name
# ============================================================================

#ACTIVE_CONFIG = 'CASE1_Manual_Bias'
#ACTIVE_CONFIG = 'WeightAnalysis'
ACTIVE_CONFIG = 'WeightAnalysis_Old'
#ACTIVE_CONFIG = 'WeightAnalysis'        # ← change this to switch configs

import importlib
_cfg = importlib.import_module(f'ExportConfigs.{ACTIVE_CONFIG}')

DATASET_LABEL      = _cfg.DATASET_LABEL
SELECTED_SIMS      = _cfg.SELECTED_SIMS
SIM_LABELS         = _cfg.SIM_LABELS
OUTPUT_DIR         = _cfg.OUTPUT_DIR
FIGURES_TO_EXPORT  = _cfg.FIGURES_TO_EXPORT
SINGLE_SIM_TABLES  = _cfg.SINGLE_SIM_TABLES
_SIM_COLORS        = getattr(_cfg, 'SIM_COLORS',    {})
_SIM_LINESTYLE     = getattr(_cfg, 'SIM_LINESTYLE', {})


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _get_data():
    """Return (simulations_dict, sim_names_list) from the configured dataset."""
    label = DATASET_LABEL
    if label is None:
        keys = list(all_datasets.keys())
        if not keys:
            sys.exit("ERROR: No datasets loaded.  Check DATA_FILES in DashInteractivePlotFull.py.")
        label = keys[0]
    return get_active_data(label)


def _get_sims_and_labels(sims, all_names):
    """Return (selected_sim_names, display_labels) respecting configuration."""
    names = SELECTED_SIMS if SELECTED_SIMS is not None else list(all_names)
    names = [n for n in names if n in sims]   # drop any that no longer exist
    if SIM_LABELS is not None and len(SIM_LABELS) == len(names):
        labels = list(SIM_LABELS)
    else:
        labels = list(names)
    return names, labels


def _color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


def _apply_date_formatter(ax):
    """Tidy date formatter: major ticks every 5 years, rotated labels."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')


# ============================================================================
# PLOT FUNCTIONS
# ============================================================================

def mpl_rms_compare(sims, names, labels,
                    title='RMS vs SPICE Comparison'):
    """Dot plot of final total RMS per simulation."""
    vals = [sims[n].get('rms_SPICE', np.nan) for n in names]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(FIG_W_DOUBLE, 0.8 * len(names)), FIG_H_DEFAULT))

    # Connecting line through all non-NaN points
    valid_x = [xi for xi, v in enumerate(vals) if not np.isnan(v)]
    valid_v = [v  for v  in vals               if not np.isnan(v)]
    if len(valid_x) > 1:
        ax.plot(valid_x, valid_v, '-', color='gray', linewidth=1.0,
                alpha=0.5, zorder=1)

    for i, (xi, v) in enumerate(zip(x, vals)):
        if not np.isnan(v):
            ax.plot(xi, v, 'o', color=_color(i), markersize=8,
                    markeredgecolor='black', markeredgewidth=0.5, zorder=2)
            ax.text(xi, v, f'  {v:.3f}',
                    ha='left', va='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('RMS [km]')
    ax.set_title(title)
    ax.set_xlim(-0.6, len(names) - 0.4)
    fig.tight_layout()
    return fig


def mpl_rsw_compare(sims, names, labels,
                    show_initial=False,
                    title='RSW Difference vs SPICE',
                    figsize=None):
    """3-row time-series of RSW difference vs SPICE for multiple simulations."""
    if figsize is None:
        figsize = (FIG_W_DOUBLE, 7.0)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    ylabels = [r'$\Delta R$ [km]', r'$\Delta S$ [km]', r'$\Delta W$ [km]']

    for i, (ax, ylab) in enumerate(zip(axes, ylabels)):
        for j, (sn, lbl) in enumerate(zip(names, labels)):
            if 'diff_SPICE_RSW' not in sims.get(sn, {}):
                continue
            dr = sims[sn]['diff_SPICE_RSW']
            times = get_rsw_times(sims[sn], n_points=len(dr))
            rms = sims[sn].get('rms_SPICE', None)
            leg = f"{lbl} (RMS: {rms:.2f} km)" if rms else lbl
            ax.plot(times, dr[:, i],
                    color=_color(j),
                    label=leg if i == 0 else '_nolegend_',
                    linewidth=1.0, alpha=0.9)

            if show_initial and 'diff_SPICE_RSW_initial' in sims.get(sn, {}):
                dr_init = sims[sn]['diff_SPICE_RSW_initial']
                t_init = convert_time_array_to_datetime(
                    sims[sn]['time_column_initial'].reshape(-1, 1))
                ax.plot(t_init, dr_init[:, i],
                        color=_color(j), linestyle='--',
                        linewidth=0.8, alpha=0.5,
                        label='_nolegend_')

        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.set_ylabel(ylab)
        _apply_date_formatter(ax)

    axes[0].set_title(title)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[-1].set_xlabel('Date')
    fig.tight_layout()
    return fig


def mpl_formal_compare(sims, names, labels,
                       title='Formal Errors RSW',
                       figsize=None):
    """3-row time-series of formal errors σ_R, σ_S, σ_W."""
    if figsize is None:
        figsize = (FIG_W_DOUBLE, 7.0)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    ylabels = [r'$\sigma_R$ [km]', r'$\sigma_S$ [km]', r'$\sigma_W$ [km]']
    comps = ['R', 'S', 'W']

    for i, (ax, ylab, comp) in enumerate(zip(axes, ylabels, comps)):
        for j, (sn, lbl) in enumerate(zip(names, labels)):
            if 'formal_errors_RSW_km' not in sims.get(sn, {}):
                continue
            fe = sims[sn]['formal_errors_RSW_km']
            if 'state_history_array' in sims[sn]:
                times = convert_time_array_to_datetime(
                    sims[sn]['state_history_array'][:, 0])
            else:
                times = list(range(len(fe)))
            st = compute_formal_error_statistics(fe)
            mx = st[comp]['max']
            leg = f"{lbl} (max: {mx:.2f} km)"
            ax.plot(times, fe[:, i],
                    color=_color(j),
                    label=leg if i == 0 else '_nolegend_',
                    linewidth=1.0)

        ax.set_ylabel(ylab)
        _apply_date_formatter(ax)

    axes[0].set_title(title)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[-1].set_xlabel('Date')
    fig.tight_layout()
    return fig


def mpl_rsw_stats(sims, names, labels,
                  sim_subset=None,
                  title='RSW Statistics',
                  figsize=None):
    """5×3 grid: cols = R / S / W  |  rows = RSW diff (Mean/RMS/Max) + Formal σ (Max/RMS).
    A line connects simulation points to show the trend across simulations."""
    if sim_subset is not None:
        pairs  = [(n, labels[names.index(n)]) for n in sim_subset if n in sims and n in names]
        names  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

    # (row_label, data_source, stat_key, y_unit)
    row_defs = [
        ('RSW diff — Mean',  'diff',   'mean', 'km'),
        ('RSW diff — RMS',   'diff',   'rms',  'km'),
        ('RSW diff — Max',   'diff',   'max',  'km'),
        ('Formal σ — Max',   'formal', 'max',  'km'),
        ('Formal σ — RMS',   'formal', 'rms',  'km'),
    ]
    comps = ['R', 'S', 'W']
    x = np.arange(len(names))

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.8, FIG_H_DEFAULT * len(row_defs) / 1.5)

    fig, axes = plt.subplots(len(row_defs), 3, figsize=figsize, sharey='row')

    for ri, (row_label, src, stat_key, unit) in enumerate(row_defs):
        for ci, comp in enumerate(comps):
            ax = axes[ri, ci]
            col = RSW_COLORS[comp]

            vals = []
            for sn in names:
                sd = sims.get(sn, {})
                if src == 'diff' and 'diff_SPICE_RSW' in sd:
                    ds = compute_rsw_statistics(sd['diff_SPICE_RSW'])
                    vals.append(ds[comp][stat_key])
                elif src == 'formal' and 'formal_errors_RSW_km' in sd:
                    st = compute_formal_error_statistics(sd['formal_errors_RSW_km'])
                    vals.append(st[comp][stat_key])
                else:
                    vals.append(np.nan)

            # Connecting line through all non-NaN points
            valid_x = [xi for xi, v in enumerate(vals) if not np.isnan(v)]
            valid_v = [v  for v  in vals               if not np.isnan(v)]
            if len(valid_x) > 1:
                ax.plot(valid_x, valid_v, '-', color=col, linewidth=1.2,
                        alpha=0.5, zorder=1)

            # Dots + value annotations
            for xi, v in enumerate(vals):
                if not np.isnan(v):
                    ax.plot(xi, v, 'o', color=col, markersize=6,
                            markeredgecolor='black', markeredgewidth=0.4, zorder=2)
                    ax.text(xi, v, f'  {v:.2f}',
                            ha='left', va='center', fontsize=7)

            ax.set_xticks(x)
            ax.set_xlim(-0.5, len(names) - 0.5)
            if ri == len(row_defs) - 1:
                ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
            else:
                ax.set_xticklabels([])
            if ci == 0:
                ax.set_ylabel(f'{row_label} [{unit}]', fontsize=9)
            if ri == 0:
                ax.set_title(comp, fontsize=11)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def mpl_gof(sims, names, labels,
            metric='rms',
            show_initial=False,
            log_y=False,
            title=None):
    """Goodness-of-fit: WRMS / RMS / cost-function comparison across sims."""
    wm = compute_wrms_and_cost(sims, names)
    if not wm:
        print("WARNING: No WRMS/RMS data, skipping GoF figure.")
        return None

    metric_map = {
        'wrms': ('final_wrms_combined_method1_mas',
                 'initial_wrms_combined_method1_mas',
                 'WRMS [mas]'),
        'rms':  ('final_rms_combined_mas',
                 'initial_rms_combined_mas',
                 'RMS [mas]'),
        'cost': ('final_cost_function',
                 'initial_cost_function',
                 'Cost Function'),
    }
    fk, ik, ylabel = metric_map.get(metric, metric_map['rms'])

    avs      = [n for n in names if n in wm]
    avlabels = [labels[names.index(n)] for n in avs]
    fv       = [wm[n].get(fk) for n in avs]
    iv       = [wm[n].get(ik) for n in avs]
    x        = np.arange(len(avs))

    if title is None:
        title = f'{ylabel} Comparison'

    fig, ax = plt.subplots(
        figsize=(max(FIG_W_DOUBLE, 0.8 * len(avs)), FIG_H_DEFAULT))

    if show_initial:
        ax.plot(x, iv, 'o--', color='lightcoral', label='Initial',
                linewidth=1.2, markersize=6)
    ax.plot(x, fv, 'o-', color='steelblue', label='Final',
            linewidth=1.5, markersize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(avlabels, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale('log')
    if show_initial:
        ax.legend()

    fig.tight_layout()
    return fig


def mpl_gof_combined(sims, names, labels,
                     show_initial=False,
                     log_y=False,
                     title='Goodness of Fit Comparison',
                     figsize=None):
    """1×3 subplots: WRMS [mas], RMS [mas], Cost Function — all simulations."""
    wm = compute_wrms_and_cost(sims, names)
    if not wm:
        print("WARNING: No WRMS/RMS data, skipping GoF combined figure.")
        return None

    avs      = [n for n in names if n in wm]
    avlabels = [labels[names.index(n)] for n in avs]
    x        = np.arange(len(avs))

    metrics = [
        ('final_wrms_combined_method1_mas', 'initial_wrms_combined_method1_mas', 'WRMS [mas]'),
        ('final_rms_combined_mas',          'initial_rms_combined_mas',          'RMS [mas]'),
        ('final_cost_function',             'initial_cost_function',             'Cost Function'),
    ]

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.4, FIG_H_DEFAULT * len(metrics) / 1.5)

    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)

    for i, (ax, (fk, ik, ylabel)) in enumerate(zip(axes, metrics)):
        fv = [wm[n].get(fk) for n in avs]
        iv = [wm[n].get(ik) for n in avs]

        if show_initial:
            ax.plot(x, iv, 'o--', color='lightcoral', label='Initial',
                    linewidth=1.2, markersize=6)
        ax.plot(x, fv, 'o-', color='steelblue', label='Final',
                linewidth=1.5, markersize=7)

        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=10)
        if log_y:
            ax.set_yscale('log')

    # x-tick labels only on bottom subplot
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(avlabels, rotation=30, ha='right')

    if show_initial:
        axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def mpl_corr_heatmap(sims, sim_name,
                     title=None, figsize=None):
    """Absolute correlation matrix heatmap for a single simulation."""
    if 'correlations' not in sims.get(sim_name, {}):
        print(f"WARNING: No correlation data for '{sim_name}', skipping.")
        return None

    cm     = np.abs(sims[sim_name]['correlations'])
    labels = get_parameter_labels(sims[sim_name].get('est_parameters', []))
    n      = len(labels)

    if figsize is None:
        side    = max(3.5, 0.55 * n)
        figsize = (side, side)
    if title is None:
        title = f'|Correlations|: {sim_name}'

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap='Greys', vmin=0, vmax=1, aspect='auto')
    fig.colorbar(im, ax=ax, label='|Correlation|', shrink=0.8)

    fontsize = max(5, 9 - n // 3)
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            ax.text(j, i, f'{v:.2f}',
                    ha='center', va='center',
                    fontsize=fontsize,
                    color='white' if v > 0.5 else 'black')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def mpl_residual_histogram(sims, names, labels,
                           iteration='final',
                           fit_gauss=True,
                           bins=50,
                           figsize=None,
                           title='Residual Histogram'):
    """RA / Dec residual histograms with optional Gaussian fit."""
    if figsize is None:
        figsize = (FIG_W_DOUBLE, FIG_H_DEFAULT)

    col_ra  = ('ra_residual_final_mas'   if iteration == 'final'
               else 'ra_residual_initial_mas')
    col_dec = ('dec_residual_final_mas'  if iteration == 'final'
               else 'dec_residual_initial_mas')

    fig, (ax_ra, ax_dec) = plt.subplots(1, 2, figsize=figsize)

    for j, (sn, lbl) in enumerate(zip(names, labels)):
        if 'residual_df' not in sims.get(sn, {}):
            continue
        df  = sims[sn]['residual_df']
        ra  = df[col_ra].dropna().values
        dec = df[col_dec].dropna().values

        ax_ra.hist(ra,  bins=bins, alpha=0.5,
                   color=_color(j), label=lbl, density=True)
        ax_dec.hist(dec, bins=bins, alpha=0.5,
                    color=_color(j), density=True)

        if fit_gauss:
            for ax, data in [(ax_ra, ra), (ax_dec, dec)]:
                mu, sigma = sp_stats.norm.fit(data)
                xr = np.linspace(data.min(), data.max(), 200)
                ax.plot(xr, sp_stats.norm.pdf(xr, mu, sigma),
                        color=_color(j), linewidth=1.5, linestyle='--')

    ax_ra.set_xlabel('RA Residual [mas]')
    ax_dec.set_xlabel('Dec Residual [mas]')
    ax_ra.set_ylabel('Density')
    ax_ra.set_title('RA Residuals')
    ax_dec.set_title('Dec Residuals')
    ax_ra.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# ============================================================================
# PARAMETER UPDATE HELPERS
# ============================================================================

def _get_iau_reference(sims, all_names):
    """Build reference dicts from IAUPole_* sims' initial parameter values.

    Returns (iau_ref_inertial, iau_ref_rsw) — both are dicts mapping
    parameter label → initial value.  Falls back to empty dicts if no
    IAU sim has parameter_history.
    Since all IAU sims share identical initial conditions, the first IAU
    sim that provides each label wins.
    """
    iau_ref_inertial = {}
    iau_ref_rsw      = {}
    rsw_map = {'X': 'R', 'Y': 'S', 'Z': 'W', 'VX': 'VR', 'VY': 'VS', 'VZ': 'VW'}

    for sn in all_names:
        if not sn.startswith('IAUPole'):
            continue
        sd = sims.get(sn, {})
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            continue
        ph_in    = sd['parameter_history']
        ph_rsw   = sd.get('parameter_history_RSW', ph_in)
        lbls, _, _ = get_parameter_info(sd['est_parameters'])
        rsw_lbls = [rsw_map.get(l, l) for l in lbls]
        for i, lbl in enumerate(lbls):
            if lbl not in iau_ref_inertial:
                iau_ref_inertial[lbl] = ph_in[i, 0]
        for i, lbl in enumerate(rsw_lbls):
            if lbl not in iau_ref_rsw:
                iau_ref_rsw[lbl] = ph_rsw[i, 0]

    return iau_ref_inertial, iau_ref_rsw


def _param_update(sim_data, group_filter, scale=1.0, iau_ref=None):
    """Extract the final–initial parameter update for a given group.

    Returns (values, labels, units) filtered to `group_filter` (list of group
    names).  `scale` is applied to every value (e.g. rad→deg).
    When `iau_ref` is provided (dict label→value) the reference is taken from
    that dict instead of the sim's own initial, falling back to own initial for
    any label not present in iau_ref.
    Returns (None, None, None) if the sim has no parameter_history.
    """
    if 'parameter_history' not in sim_data or 'est_parameters' not in sim_data:
        return None, None, None
    ph = sim_data['parameter_history']
    labels, groups, units = get_parameter_info(sim_data['est_parameters'])
    sel_v, sel_l, sel_u = [], [], []
    for i, (lbl, grp, unt) in enumerate(zip(labels, groups, units)):
        if grp in group_filter:
            ref = iau_ref.get(lbl, ph[i, 0]) if iau_ref is not None else ph[i, 0]
            sel_v.append((ph[i, -1] - ref) * scale)
            sel_l.append(lbl)
            sel_u.append(unt)
    return np.array(sel_v) if sel_v else None, sel_l, sel_u


def _param_update_rsw(sim_data, group_filter, scale=1.0, iau_ref=None):
    """Same as _param_update but prefers parameter_history_RSW if available.

    `iau_ref` should use RSW-remapped labels (R/S/W/VR/VS/VW) when provided.
    """
    if 'parameter_history' not in sim_data or 'est_parameters' not in sim_data:
        return None, None, None
    if 'parameter_history_RSW' in sim_data:
        ph      = sim_data['parameter_history_RSW']
        rsw_map = {'X': 'R', 'Y': 'S', 'Z': 'W', 'VX': 'VR', 'VY': 'VS', 'VZ': 'VW'}
    else:
        ph      = sim_data['parameter_history']
        rsw_map = {}
    labels, groups, units = get_parameter_info(sim_data['est_parameters'])
    labels = [rsw_map.get(l, l) for l in labels]
    sel_v, sel_l, sel_u = [], [], []
    for i, (lbl, grp, unt) in enumerate(zip(labels, groups, units)):
        if grp in group_filter:
            ref = iau_ref.get(lbl, ph[i, 0]) if iau_ref is not None else ph[i, 0]
            sel_v.append((ph[i, -1] - ref) * scale)
            sel_l.append(lbl)
            sel_u.append(unt)
    return np.array(sel_v) if sel_v else None, sel_l, sel_u


# ── Shared bar-chart helper ───────────────────────────────────────────────────

def _bar_ax(ax, vals, bar_color, x_labels, unit_str, param_title,
            show_xticks=True, fontsize_annot=7):
    """Draw a bar chart on `ax` with value annotations above/below each bar."""
    x = np.arange(len(vals))
    bars = ax.bar(x, vals,
                  color=bar_color, edgecolor='black', linewidth=0.5, alpha=0.85)
    for bar, v in zip(bars, vals):
        if np.isnan(v):
            continue
        if v >= 0:
            ypos = bar.get_y() + bar.get_height()   # tip of positive bar
            va_  = 'bottom'                          # text goes above tip
        else:
            ypos = bar.get_y()                       # tip of negative bar
            va_  = 'top'                             # text hangs below tip
        ax.text(bar.get_x() + bar.get_width() / 2,
                ypos, f'{v:.3g}',
                ha='center', va=va_, fontsize=fontsize_annot)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_title(f'Δ{param_title}  [{unit_str}]', fontsize=10)
    ax.set_xticks(x)
    if show_xticks:
        ax.set_xticklabels(x_labels, rotation=35, ha='right', fontsize=8)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)


# ── PDF 1: Total position and velocity magnitude update ───────────────────────

_RAD_TO_DEG = 180.0 / np.pi

def mpl_param_state(sims, names, labels,
                    title='Initial State Update — Total Magnitude',
                    figsize=None,
                    variant='iau'):
    """1×2 bar chart: |Δpos| [km] and |Δvel| [km/s] across simulations."""
    iau_prefix = 'IAUPole'
    sim_prefix = 'SimPole'
    iau_ref_in, _ = _get_iau_reference(sims, names)

    if variant == 'iau':
        f_names      = [n for n in names if n.startswith(iau_prefix)]
        ref_map      = {n: None for n in f_names}
        title_suffix = ' — IAU'
    elif variant == 'simpole':
        f_names      = [n for n in names if n.startswith(sim_prefix)]
        ref_map      = {n: None for n in f_names}
        title_suffix = ' — SimPole'
    else:  # combined
        f_names      = list(names)
        ref_map      = {n: (iau_ref_in if n.startswith(sim_prefix) else None)
                        for n in f_names}
        title_suffix = ' — Combined (SimPole vs IAU ref)'

    f_labels = [labels[names.index(n)] for n in f_names]
    title    = title + title_suffix

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.4, FIG_H_DEFAULT)

    pos_mags, vel_mags = [], []
    for sn in f_names:
        vals, plabels, _ = _param_update(
            sims.get(sn, {}), ['Position', 'Velocity'], iau_ref=ref_map[sn])
        if vals is None:
            pos_mags.append(np.nan)
            vel_mags.append(np.nan)
            continue
        pos_idx = [i for i, l in enumerate(plabels) if l in ('X', 'Y', 'Z')]
        vel_idx = [i for i, l in enumerate(plabels) if l in ('VX', 'VY', 'VZ')]
        pos_mags.append(np.linalg.norm(vals[pos_idx]) / 1000.0 if pos_idx else np.nan)
        vel_mags.append(np.linalg.norm(vals[vel_idx]) / 1000.0 if vel_idx else np.nan)

    fig, (ax_p, ax_v) = plt.subplots(1, 2, figsize=figsize)
    _bar_ax(ax_p, pos_mags, _color(0), f_labels, 'km',   '|pos|')
    _bar_ax(ax_v, vel_mags, _color(1), f_labels, 'km/s', '|vel|')
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


# ── PDF 2: RSW position and velocity update ───────────────────────────────────

def mpl_param_rsw(sims, names, labels,
                  title='Initial State Update — RSW',
                  figsize=None,
                  variant='iau'):
    """2×3 bar-chart grid: ΔR/ΔS/ΔW (top) and ΔVR/ΔVS/ΔVW (bottom) [km, km/s]."""
    iau_prefix = 'IAUPole'
    sim_prefix = 'SimPole'
    _, iau_ref_rsw = _get_iau_reference(sims, names)

    if variant == 'iau':
        f_names      = [n for n in names if n.startswith(iau_prefix)]
        ref_map      = {n: None for n in f_names}
        title_suffix = ' — IAU'
    elif variant == 'simpole':
        f_names      = [n for n in names if n.startswith(sim_prefix)]
        ref_map      = {n: None for n in f_names}
        title_suffix = ' — SimPole'
    else:  # combined
        f_names      = list(names)
        ref_map      = {n: (iau_ref_rsw if n.startswith(sim_prefix) else None)
                        for n in f_names}
        title_suffix = ' — Combined (SimPole vs IAU ref)'

    f_labels   = [labels[names.index(n)] for n in f_names]
    title      = title + title_suffix
    pos_labels = ['R',  'S',  'W' ]
    vel_labels = ['VR', 'VS', 'VW']

    def _vals_for(comp_label):
        base = comp_label.lstrip('V')
        col  = RSW_COLORS.get(base, _color(0))
        row  = []
        for sn in f_names:
            v_all, pl, _ = _param_update_rsw(
                sims.get(sn, {}), ['Position', 'Velocity'], iau_ref=ref_map[sn])
            if v_all is None or comp_label not in pl:
                row.append(np.nan)
            else:
                row.append(v_all[pl.index(comp_label)] / 1000.0)
        return row, col

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.5, FIG_H_DEFAULT * 1.8)
    fig, axes = plt.subplots(2, 3, figsize=figsize, squeeze=False)

    for ci, (pl, vl) in enumerate(zip(pos_labels, vel_labels)):
        top_vals, top_col = _vals_for(pl)
        bot_vals, bot_col = _vals_for(vl)
        _bar_ax(axes[0, ci], top_vals, top_col, f_labels, 'km',   pl,  show_xticks=False)
        _bar_ax(axes[1, ci], bot_vals, bot_col, f_labels, 'km/s', vl,  show_xticks=True)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


# ── PDF 3: Pole position update ───────────────────────────────────────────────

def mpl_param_pole_pos(sims, names, labels,
                       title='Pole Position Update  (α₀, δ₀)',
                       figsize=None,
                       variant='iau'):
    """1×2 bar chart: Δα₀ and Δδ₀ across simulations [deg]."""
    iau_prefix = 'IAUPole'
    sim_prefix = 'SimPole'
    iau_ref_in, _ = _get_iau_reference(sims, names)

    if variant == 'iau':
        base_names   = [n for n in names if n.startswith(iau_prefix)]
        ref_map      = {n: None for n in base_names}
        title_suffix = ' — IAU'
    elif variant == 'simpole':
        base_names   = [n for n in names if n.startswith(sim_prefix)]
        ref_map      = {n: None for n in base_names}
        title_suffix = ' — SimPole'
    else:  # combined
        base_names   = list(names)
        ref_map      = {n: (iau_ref_in if n.startswith(sim_prefix) else None)
                        for n in base_names}
        title_suffix = ' — Combined (SimPole vs IAU ref)'

    title = title + title_suffix

    # Keep only sims (within variant filter) that estimate pole position.
    filt = [(sn, labels[names.index(sn)]) for sn in base_names
            if _param_update(sims.get(sn, {}), ['Pole Position'])[0] is not None]
    f_names, f_labels = (zip(*filt) if filt else ([], []))

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.2, FIG_H_DEFAULT)

    fig, (ax_a, ax_d) = plt.subplots(1, 2, figsize=figsize)
    pole_colors = {'α₀': '#9467bd', 'δ₀': '#e377c2'}

    for ax, plabel in zip((ax_a, ax_d), ('α₀', 'δ₀')):
        vals = []
        for sn in f_names:
            v_all, pl, _ = _param_update(
                sims.get(sn, {}), ['Pole Position'],
                scale=_RAD_TO_DEG, iau_ref=ref_map[sn])
            vals.append(v_all[pl.index(plabel)] if plabel in pl else np.nan)
        _bar_ax(ax, vals, pole_colors[plabel], f_labels, 'deg', plabel, fontsize_annot=8)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


# ── PDF 4: Pole libration update ──────────────────────────────────────────────

def mpl_param_pole_lib(sims, names, labels,
                       title='Pole Libration Update  (α₁, δ₁)',
                       figsize=None,
                       variant='iau'):
    """1×2 bar chart: Δα₁ and Δδ₁ across simulations [deg]."""
    iau_prefix = 'IAUPole'
    sim_prefix = 'SimPole'
    iau_ref_in, _ = _get_iau_reference(sims, names)

    if variant == 'iau':
        base_names   = [n for n in names if n.startswith(iau_prefix)]
        ref_map      = {n: None for n in base_names}
        title_suffix = ' — IAU'
    elif variant == 'simpole':
        base_names   = [n for n in names if n.startswith(sim_prefix)]
        ref_map      = {n: None for n in base_names}
        title_suffix = ' — SimPole'
    else:  # combined
        base_names   = list(names)
        ref_map      = {n: (iau_ref_in if n.startswith(sim_prefix) else None)
                        for n in base_names}
        title_suffix = ' — Combined (SimPole vs IAU ref)'

    title = title + title_suffix

    # Keep only sims (within variant filter) that estimate pole librations.
    filt = [(sn, labels[names.index(sn)]) for sn in base_names
            if _param_update(sims.get(sn, {}), ['Pole Librations'])[0] is not None]
    f_names, f_labels = (zip(*filt) if filt else ([], []))

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.2, FIG_H_DEFAULT)

    fig, (ax_a, ax_d) = plt.subplots(1, 2, figsize=figsize)
    lib_colors = {'α₁': '#17becf', 'δ₁': '#bcbd22'}

    for ax, plabel in zip((ax_a, ax_d), ('α₁', 'δ₁')):
        vals = []
        for sn in f_names:
            v_all, pl, _ = _param_update(
                sims.get(sn, {}), ['Pole Librations'],
                scale=_RAD_TO_DEG, iau_ref=ref_map[sn])
            vals.append(v_all[pl.index(plabel)] if plabel in pl else np.nan)
        _bar_ax(ax, vals, lib_colors[plabel], f_labels, 'deg', plabel, fontsize_annot=8)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


# ── RSW difference: full + zoomed subfigure ───────────────────────────────────

def _setup_zoom_xaxis(ax):
    """Apply auto date locator + '%d %b' formatter to a zoomed axis."""
    loc = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)
    ax.tick_params(axis='x', labelsize=10, length=4)


def _hide_xticklabels(ax):
    """Hide tick labels (but keep tick marks) on a shared-x axis."""
    ax.tick_params(labelbottom=False, bottom=True)


def mpl_rsw_with_zoom(sims, names, labels,
                      sim_subset=None,
                      zoom_days=90,
                      title='RSW Difference vs NEP097',
                      figsize=None):
    """3×2 figure: left = full time range, right = first zoom_days of data.

    Uses _SIM_COLORS / _SIM_LINESTYLE (IAU = cool, SimPole = warm).
    X-tick labels only on bottom row; zoom column uses '%d %b' date format.
    """
    # ── resolve sim subset ───────────────────────────────────────────────────
    if sim_subset is not None:
        pairs = [
            (n, labels[names.index(n)] if n in names else n.replace('_cov', ''))
            for n in sim_subset if n in sims
        ]
    else:
        pairs = list(zip(names, labels))
    pairs = [(n, lb) for n, lb in pairs if 'diff_SPICE_RSW' in sims.get(n, {})]
    if not pairs:
        return None
    plot_names, plot_labels = zip(*pairs)

    # ── zoom window: earliest data point + zoom_days ─────────────────────────
    zoom_start = zoom_end = None
    for sn in plot_names:
        dr    = sims[sn]['diff_SPICE_RSW']
        times = get_rsw_times(sims[sn], n_points=len(dr))
        if times is not None and len(times) > 0:
            zoom_start = times[0]
            zoom_end   = zoom_start + timedelta(days=zoom_days)
            break

    # ── figure layout: 3 rows × 2 cols ───────────────────────────────────────
    if figsize is None:
        figsize = (FIG_W_DOUBLE * 2, 8.5)
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(3, 2, figure=fig,
                   hspace=0.08, wspace=0.30,
                   left=0.07, right=0.98, top=0.91, bottom=0.10)

    axes_full = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    axes_zoom = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    for i in range(1, 3):
        axes_full[i].sharex(axes_full[0])
        axes_zoom[i].sharex(axes_zoom[0])

    ylabels = [r'$\Delta R$ [km]', r'$\Delta S$ [km]', r'$\Delta W$ [km]']

    # ── plot data ─────────────────────────────────────────────────────────────
    for axes, apply_zoom in [(axes_full, False), (axes_zoom, True)]:
        for row, (ax, ylab) in enumerate(zip(axes, ylabels)):
            for j, (sn, lbl) in enumerate(zip(plot_names, plot_labels)):
                dr    = sims[sn]['diff_SPICE_RSW']
                times = get_rsw_times(sims[sn], n_points=len(dr))
                rms   = sims[sn].get('rms_SPICE')
                leg   = f'{lbl} (RMS: {rms:.0f} km)' if rms else lbl
                col   = _SIM_COLORS.get(sn, _color(j))
                ls    = _SIM_LINESTYLE.get(sn, '-')
                ax.plot(times, dr[:, row],
                        color=col, linestyle=ls, linewidth=1.8, alpha=0.92,
                        label=leg if row == 0 else '_nolegend_')
            ax.axhline(0, color='gray', linewidth=0.6, linestyle=':')
            ax.set_ylabel(ylab, fontsize=11)
            ax.tick_params(axis='y', labelsize=10, length=4)
            if apply_zoom and zoom_start and zoom_end:
                ax.set_xlim(zoom_start, zoom_end)

    # ── x-axis formatting: labels only on bottom row ─────────────────────────
    for ax in axes_full[:2]:
        _hide_xticklabels(ax)
    for ax in axes_zoom[:2]:
        _hide_xticklabels(ax)

    # Full column bottom row: yearly ticks
    _apply_date_formatter(axes_full[-1])
    axes_full[-1].tick_params(axis='x', labelsize=10, length=4)
    axes_full[-1].set_xlabel('Date', fontsize=11)

    # Zoom column bottom row: auto-picked ticks, '%d %b' labels
    _setup_zoom_xaxis(axes_zoom[-1])
    axes_zoom[-1].set_xlabel('Date', fontsize=11)

    # ── titles & legend ───────────────────────────────────────────────────────
    axes_full[0].set_title('(a) Full time range', fontsize=11, loc='left')
    zoom_year = f' ({zoom_start.year})' if zoom_start else ''
    axes_zoom[0].set_title(f'(b) Zoomed{zoom_year}', fontsize=11, loc='left')
    axes_full[0].legend(loc='upper right', fontsize=9)

    fig.suptitle(title, fontsize=13)
    return fig


# ── Initial propagation RSW diff (pre-estimation) ────────────────────────────

def mpl_rsw_initial(sims, names, labels,
                    sim_subset=None,
                    unit='km',
                    title='Initial RSW Difference vs NEP097 (pre-estimation)',
                    figsize=None):
    """3-row time-series of RSW difference vs SPICE for the initial (pre-estimation)
    propagation.  Uses diff_SPICE_RSW_initial and time_column_initial.

    Typical use: pass one IAUPole sim and one SimPole sim to compare starting points.
    If sim_subset is None, auto-selects the first IAUPole and first SimPole sim found.
    unit: 'km' (default) or 'm' — data is stored in km, 'm' multiplies by 1000.
    """
    # ── resolve which sims to plot ────────────────────────────────────────────
    if sim_subset is not None:
        pairs = [
            (n, labels[names.index(n)] if n in names else n.replace('_cov', ''))
            for n in sim_subset if n in sims
        ]
    else:
        # Auto-pick first IAUPole and first SimPole that have initial data
        auto = []
        for prefix in ('IAUPole', 'SimPole'):
            for n in names:
                if n.startswith(prefix) and 'diff_SPICE_RSW_initial' in sims.get(n, {}):
                    auto.append((n, labels[names.index(n)]))
                    break
        pairs = auto

    pairs = [(n, lb) for n, lb in pairs if 'diff_SPICE_RSW_initial' in sims.get(n, {})]
    if not pairs:
        print("WARNING: No initial RSW data found, skipping mpl_rsw_initial.")
        return None
    plot_names, plot_labels = zip(*pairs)

    # ── unit scaling ─────────────────────────────────────────────────────────
    scale    = 1000.0 if unit == 'm' else 1.0
    unit_str = unit

    # ── figure ────────────────────────────────────────────────────────────────
    if figsize is None:
        figsize = (FIG_W_DOUBLE, 7.0)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    ylabels = [rf'$\Delta R$ [{unit_str}]',
               rf'$\Delta S$ [{unit_str}]',
               rf'$\Delta W$ [{unit_str}]']

    for row, (ax, ylab) in enumerate(zip(axes, ylabels)):
        for j, (sn, lbl) in enumerate(zip(plot_names, plot_labels)):
            dr    = sims[sn]['diff_SPICE_RSW_initial'] * scale
            t_col = sims[sn].get('time_column_initial')
            if t_col is not None:
                times = convert_time_array_to_datetime(t_col.reshape(-1, 1))
            else:
                times = list(range(len(dr)))

            col = _SIM_COLORS.get(sn, _color(j))
            ls  = _SIM_LINESTYLE.get(sn, '-')
            rms = float(np.sqrt(np.mean(dr ** 2)))
            leg = f'{lbl} (RMS: {rms:.0f} {unit_str})'
            ax.plot(times, dr[:, row],
                    color=col, linestyle=ls, linewidth=1.8, alpha=0.92,
                    label=leg if row == 0 else '_nolegend_')

        ax.axhline(0, color='gray', linewidth=0.6, linestyle=':')
        ax.set_ylabel(ylab, fontsize=11)
        _apply_date_formatter(ax)

    axes[0].set_title(title)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[-1].set_xlabel('Date', fontsize=11)
    fig.tight_layout()
    return fig


# ── Formal errors: full + zoomed subfigure ────────────────────────────────────

def mpl_formal_with_zoom(sims, names, labels,
                         sim_subset=None,
                         zoom_days=90,
                         title='Formal Errors RSW',
                         figsize=None):
    """3×2 figure: left = full time range, right = first zoom_days of data.

    Uses _SIM_COLORS / _SIM_LINESTYLE.
    X-tick labels only on bottom row; zoom column uses '%d %b' date format.
    """
    # ── resolve sim subset ───────────────────────────────────────────────────
    if sim_subset is not None:
        pairs = [
            (n, labels[names.index(n)] if n in names else n.replace('_cov', ''))
            for n in sim_subset if n in sims
        ]
    else:
        pairs = list(zip(names, labels))
    pairs = [(n, lb) for n, lb in pairs if 'formal_errors_RSW_km' in sims.get(n, {})]
    if not pairs:
        return None
    plot_names, plot_labels = zip(*pairs)

    # ── zoom window ───────────────────────────────────────────────────────────
    zoom_start = zoom_end = None
    for sn in plot_names:
        sha = sims[sn].get('state_history_array')
        if sha is not None and len(sha) > 0:
            times = convert_time_array_to_datetime(sha[:, 0])
            if times is not None and len(times) > 0:
                zoom_start = times[0]
                zoom_end   = zoom_start + timedelta(days=zoom_days)
                break

    # ── figure layout ─────────────────────────────────────────────────────────
    if figsize is None:
        figsize = (FIG_W_DOUBLE * 2, 8.5)
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(3, 2, figure=fig,
                   hspace=0.08, wspace=0.30,
                   left=0.07, right=0.98, top=0.91, bottom=0.10)

    axes_full = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    axes_zoom = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    for i in range(1, 3):
        axes_full[i].sharex(axes_full[0])
        axes_zoom[i].sharex(axes_zoom[0])

    ylabels = [r'$\sigma_R$ [km]', r'$\sigma_S$ [km]', r'$\sigma_W$ [km]']

    # ── plot data ─────────────────────────────────────────────────────────────
    for axes, apply_zoom in [(axes_full, False), (axes_zoom, True)]:
        for row, (ax, ylab) in enumerate(zip(axes, ylabels)):
            for j, (sn, lbl) in enumerate(zip(plot_names, plot_labels)):
                fe  = sims[sn]['formal_errors_RSW_km']
                sha = sims[sn].get('state_history_array')
                times = (convert_time_array_to_datetime(sha[:, 0])
                         if sha is not None else list(range(len(fe))))
                st  = compute_formal_error_statistics(fe)
                mx  = st[['R', 'S', 'W'][row]]['max']
                leg = f'{lbl} (max: {mx:.1f} km)'
                col = _SIM_COLORS.get(sn, _color(j))
                ls  = _SIM_LINESTYLE.get(sn, '-')
                ax.plot(times, fe[:, row],
                        color=col, linestyle=ls, linewidth=1.8,
                        label=leg if row == 0 else '_nolegend_')
            ax.set_ylabel(ylab, fontsize=11)
            ax.tick_params(axis='y', labelsize=10, length=4)
            if apply_zoom and zoom_start and zoom_end:
                ax.set_xlim(zoom_start, zoom_end)

    # ── x-axis formatting: labels only on bottom row ─────────────────────────
    for ax in axes_full[:2]:
        _hide_xticklabels(ax)
    for ax in axes_zoom[:2]:
        _hide_xticklabels(ax)

    _apply_date_formatter(axes_full[-1])
    axes_full[-1].tick_params(axis='x', labelsize=10, length=4)
    axes_full[-1].set_xlabel('Date', fontsize=11)

    _setup_zoom_xaxis(axes_zoom[-1])
    axes_zoom[-1].set_xlabel('Date', fontsize=11)

    # ── titles & legend ───────────────────────────────────────────────────────
    axes_full[0].set_title('(a) Full time range', fontsize=11, loc='left')
    zoom_year = f' ({zoom_start.year})' if zoom_start else ''
    axes_zoom[0].set_title(f'(b) Zoomed{zoom_year}', fontsize=11, loc='left')
    axes_full[0].legend(loc='upper right', fontsize=9)

    fig.suptitle(title, fontsize=13)
    return fig


# ============================================================================
# DISPATCH TABLE
# ============================================================================

_PLOT_REGISTRY = {
    'rms_compare':         mpl_rms_compare,
    'rsw_compare':         mpl_rsw_compare,
    'formal_compare':      mpl_formal_compare,
    'rsw_stats':           mpl_rsw_stats,
    'gof':                 mpl_gof,
    'gof_combined':        mpl_gof_combined,
    'corr_heatmap':        mpl_corr_heatmap,
    'residual_histogram':  mpl_residual_histogram,
    'param_state':         mpl_param_state,
    'param_rsw':           mpl_param_rsw,
    'param_pole_pos':      mpl_param_pole_pos,
    'param_pole_lib':      mpl_param_pole_lib,
    'rsw_initial':         mpl_rsw_initial,
    'rsw_with_zoom':       mpl_rsw_with_zoom,
    'formal_with_zoom':    mpl_formal_with_zoom,
}


# ============================================================================
# PARAMETER TABLE EXPORT
# ============================================================================

# Unit scales applied per parameter group for display.
_GROUP_SCALE = {
    'Position':        (1e-3,         'km'),
    'Velocity':        (1e-3,         'km/s'),
    'Pole Position':   (_RAD_TO_DEG,  'deg'),
    'Pole Rate':       (_RAD_TO_DEG,  'deg/yr'),
    'Pole Librations': (_RAD_TO_DEG,  'deg'),
}
_TARGET_GROUPS = {'Position', 'Velocity', 'Pole Position', 'Pole Librations'}


def generate_single_sim_table(sims, sim_name, names, out_path):
    """Write a LaTeX table (.tex) for one simulation (sim_name).

    Columns: Parameter | Group | Unit | IAU initial | SimPole initial
             | Final | Δ (final − IAU initial) | Δ [%]
    Rows cover all target parameter groups (state + pole + librations).

    Requires preamble: \\usepackage{booktabs,siunitx,graphicx}
    """

    def _esc(s):
        return s.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

    def _p2tex(lbl):
        _MAP = {
            'α₀': r'$\alpha_0$',       'δ₀': r'$\delta_0$',
            'α₁': r'$\alpha_1$',       'δ₁': r'$\delta_1$',
            'α̇₀': r'$\dot{\alpha}_0$', 'δ̇₀': r'$\dot{\delta}_0$',
        }
        return _MAP.get(lbl, _esc(lbl))

    def _num(v):
        return rf'\num{{{v:.5e}}}'

    sd = sims.get(sim_name, {})
    if 'parameter_history' not in sd or 'est_parameters' not in sd:
        print(f"  SKIP table for '{sim_name}': no parameter data.")
        return

    iau_ref_in, _ = _get_iau_reference(sims, names)

    # SimPole reference (first-seen across all SimPole sims).
    sp_ref_in = {}
    for sn in names:
        if not sn.startswith('SimPole'):
            continue
        sd2 = sims.get(sn, {})
        if 'parameter_history' not in sd2 or 'est_parameters' not in sd2:
            continue
        ph2    = sd2['parameter_history']
        lbls2, _, _ = get_parameter_info(sd2['est_parameters'])
        for i, lbl in enumerate(lbls2):
            if lbl not in sp_ref_in:
                sp_ref_in[lbl] = ph2[i, 0]

    # Canonical parameter list from all sims (same approach as generate_parameter_tables).
    seen       = set()
    param_rows = []
    for sn in names:
        sd2 = sims.get(sn, {})
        if 'parameter_history' not in sd2 or 'est_parameters' not in sd2:
            continue
        lbls2, grps2, _ = get_parameter_info(sd2['est_parameters'])
        for lbl, grp in zip(lbls2, grps2):
            if grp not in _TARGET_GROUPS or lbl in seen:
                continue
            seen.add(lbl)
            sc, unit = _GROUP_SCALE.get(grp, (1.0, '---'))
            param_rows.append((lbl, grp, sc, unit))

    ph_sim    = sd['parameter_history']
    lbls_sim, _, _ = get_parameter_info(sd['est_parameters'])

    sim_label = _esc(sim_name.replace('_cov', ''))
    lines = []
    lines.append(r'% ============================================================')
    lines.append(rf'% Single-simulation detail table: {sim_name}')
    lines.append(r'% Preamble: \usepackage{booktabs,siunitx,graphicx}')
    lines.append(r'% ============================================================')
    lines.append('')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(
        rf'  \caption{{Parameter values for simulation \texttt{{{sim_label}}}. '
        rf'IAU\textsubscript{{0}} and SimPole\textsubscript{{0}} are the respective '
        rf'initial values; Final is the estimated result; '
        rf'$\Delta = \text{{Final}} - \text{{IAU}}_0$; '
        rf'$\Delta[\%] = \Delta\,/\,|\text{{IAU}}_0| \times 100$.}}'
    )
    lines.append(rf'  \label{{tab:single_{_esc(sim_name)}}}')
    lines.append(r'  \resizebox{\textwidth}{!}{%')
    lines.append(r'  \begin{tabular}{llcrrrrrrr}')
    lines.append(r'    \toprule')
    lines.append(
        r'    Parameter & Group & Unit'
        r' & IAU$_0$ & SimPole$_0$ & Final'
        r' & $\Delta$ & $\Delta\,[\%]$ \\'
    )
    lines.append(r'    \midrule')

    prev_grp = None
    for lbl, grp, sc, unit in param_rows:
        if prev_grp is not None and grp != prev_grp:
            lines.append(r'    \midrule')
        prev_grp = grp

        lbl_tex = _p2tex(lbl)
        iau_raw = iau_ref_in.get(lbl)
        sp_raw  = sp_ref_in.get(lbl)

        # Final value from this sim (--- if not estimated).
        if lbl in lbls_sim:
            final_raw = ph_sim[lbls_sim.index(lbl), -1]
            final_s   = final_raw * sc
            final_str = _num(final_s)
        else:
            final_raw = None
            final_str = '---'

        if iau_raw is None:
            lines.append(f'    {lbl_tex} & {_esc(grp)} & {unit}'
                         r' & --- & --- & ' + final_str + r' & --- & --- \\')
            continue

        iau_s  = iau_raw * sc
        sp_str = _num(sp_raw * sc) if sp_raw is not None else '---'

        if final_raw is not None:
            diff  = final_s - iau_s
            pct   = diff / abs(iau_s) * 100 if iau_s != 0 else 0.0
            lines.append(
                f'    {lbl_tex} & {_esc(grp)} & {unit}'
                f' & {_num(iau_s)} & {sp_str} & {final_str}'
                rf' & {_num(diff)} & {pct:.2f} \\'
            )
        else:
            lines.append(
                f'    {lbl_tex} & {_esc(grp)} & {unit}'
                f' & {_num(iau_s)} & {sp_str} & --- & --- & --- \\'
            )

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'  }% end resizebox')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


def generate_parameter_tables(sims, names, labels, out_path):
    """Write two LaTeX tables (.tex) for direct inclusion in Overleaf.

    Requires in the preamble:
        \\usepackage{booktabs}
        \\usepackage{siunitx}
        \\usepackage{graphicx}   % for \\resizebox

    Table 1 — Initial values: IAU initial, SimPole initial, Δ [unit], Δ [%].
               Covers ALL parameter groups (state + pole + librations).
    Table 2 — Split IAU / SimPole sub-tables.  Rows = parameters,
               columns = simulations.  Cells = final − IAU_initial in
               physical units (km, km/s, deg).
    """

    # ── small helpers ────────────────────────────────────────────────────────
    def _esc(s):
        """Escape LaTeX special characters."""
        return s.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

    def _p2tex(lbl):
        """Convert parameter label to LaTeX math."""
        _MAP = {
            'α₀': r'$\alpha_0$',     'δ₀': r'$\delta_0$',
            'α₁': r'$\alpha_1$',     'δ₁': r'$\delta_1$',
            'α̇₀': r'$\dot{\alpha}_0$', 'δ̇₀': r'$\dot{\delta}_0$',
        }
        return _MAP.get(lbl, _esc(lbl))

    def _num(v):
        """Format a number as \\num{} for siunitx."""
        return rf'\num{{{v:.5e}}}'

    def _pct(v):
        """Format a percentage to 2 decimal places."""
        return f'{v:.2f}'

    def _sim_short(lb):
        s = lb.replace('IAUPole_', '').replace('SimPole_', '')
        return _esc(s)

    def _diff(raw_diff, sc):
        """Physical difference formatted with siunitx."""
        return rf'\num{{{raw_diff * sc:.5e}}}'

    # ── build canonical param list from ALL sims (fixes missing pole params) ──
    # iau_ref_in already aggregates from all IAUPole sims.
    iau_ref_in, _ = _get_iau_reference(sims, names)

    # SimPole reference: initial value for each param, first-seen across all SimPole sims.
    sp_ref_in = {}
    for sn in names:
        if not sn.startswith('SimPole'):
            continue
        sd = sims.get(sn, {})
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            continue
        ph_     = sd['parameter_history']
        lbls_, _, _ = get_parameter_info(sd['est_parameters'])
        for i, lbl in enumerate(lbls_):
            if lbl not in sp_ref_in:
                sp_ref_in[lbl] = ph_[i, 0]

    # Iterate all sims in order to build the canonical param list.
    seen       = set()
    param_rows = []   # list of (lbl, grp, sc, unit)
    for sn in names:
        sd = sims.get(sn, {})
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            continue
        lbls_, grps_, _ = get_parameter_info(sd['est_parameters'])
        for lbl, grp in zip(lbls_, grps_):
            if grp not in _TARGET_GROUPS or lbl in seen:
                continue
            seen.add(lbl)
            sc, unit = _GROUP_SCALE.get(grp, (1.0, '---'))
            param_rows.append((lbl, grp, sc, unit))

    iau_names = [n for n in names if n.startswith('IAUPole')]
    sp_names  = [n for n in names if n.startswith('SimPole')]
    iau_lbls  = [labels[names.index(n)] for n in iau_names]
    sp_lbls   = [labels[names.index(n)] for n in sp_names]

    lines = []
    lines.append(r'% ============================================================')
    lines.append(r'% Parameter tables — Neptune/Triton orbit estimation thesis')
    lines.append(r'% Preamble: \usepackage{booktabs,siunitx,graphicx}')
    lines.append(r'% ============================================================')
    lines.append('')

    # ── TABLE 1: IAU initial vs SimPole initial ───────────────────────────────
    lines.append(r'% ---- Table 1: initial parameter values (IAU vs SimPole) ----')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(
        r'  \caption{Initial parameter values for the IAU and SimPole rotation models. '
        r'$\Delta = p_{\mathrm{SimPole,0}} - p_{\mathrm{IAU,0}}$ in the listed unit; '
        r'$\Delta[\%] = \Delta\,/\,|p_{\mathrm{IAU,0}}| \times 100$.}'
    )
    lines.append(r'  \label{tab:initial_params}')
    lines.append(r'  \resizebox{\textwidth}{!}{%')
    lines.append(r'  \begin{tabular}{llcrrrrr}')
    lines.append(r'    \toprule')
    lines.append(
        r'    Parameter & Group & Unit'
        r' & IAU initial & SimPole initial'
        r' & $\Delta$ & $\Delta\,[\%]$ \\'
    )
    lines.append(r'    \midrule')

    prev_grp = None
    for lbl, grp, sc, unit in param_rows:
        if prev_grp is not None and grp != prev_grp:
            lines.append(r'    \midrule')
        prev_grp = grp

        lbl_tex = _p2tex(lbl)
        iau_raw = iau_ref_in.get(lbl)
        sp_raw  = sp_ref_in.get(lbl)

        if iau_raw is None:
            lines.append(f'    {lbl_tex} & {_esc(grp)} & {unit}'
                         r' & --- & --- & --- & --- \\')
            continue

        iau_s = iau_raw * sc
        if sp_raw is not None:
            sp_s  = sp_raw * sc
            diff  = sp_s - iau_s
            pct   = diff / abs(iau_s) * 100 if iau_s != 0 else 0.0
            lines.append(
                f'    {lbl_tex} & {_esc(grp)} & {unit}'
                f' & {_num(iau_s)} & {_num(sp_s)}'
                rf' & {_num(diff)} & {pct:.2f} \\'
            )
        else:
            lines.append(f'    {lbl_tex} & {_esc(grp)} & {unit}'
                         f' & {_num(iau_s)} & --- & --- & --- \\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'  }% end resizebox')
    lines.append(r'\end{table}')
    lines.append('')

    # ── TABLE 2: final − IAU_initial in physical units ───────────────────────
    if not iau_ref_in:
        lines.append(r'% TABLE 2: no IAU reference data available.')
        lines.append('')
    else:
        subtables = [
            (iau_names, iau_lbls, 'IAU',
             r'IAU-model simulations',
             'tab:update_iau'),
            (sp_names, sp_lbls, 'SimPole',
             r'SimPole-model simulations (all referenced to IAU initial)',
             'tab:update_simpole'),
        ]

        for grp_names, grp_labels, grp_tag, caption_detail, tbl_label in subtables:
            if not grp_names:
                continue

            col_spec = 'lll' + 'r' * len(grp_names)

            lines.append(f'% ---- Table 2 ({grp_tag}): final - IAU_initial [physical units] ----')
            lines.append(r'\begin{table}[htbp]')
            lines.append(r'  \centering')
            lines.append(
                rf'  \caption{{Parameter update $\Delta p = p_{{\mathrm{{final}}}} - '
                rf'p_{{\mathrm{{IAU,0}}}}$ in physical units for {caption_detail}.}}'
            )
            lines.append(rf'  \label{{{tbl_label}}}')
            lines.append(r'  \resizebox{\textwidth}{!}{%')
            lines.append(rf'  \begin{{tabular}}{{{col_spec}}}')
            lines.append(r'    \toprule')

            hdr_parts = ['Parameter', 'Group', 'Unit'] + [_sim_short(lb) for lb in grp_labels]
            lines.append('    ' + ' & '.join(hdr_parts) + r' \\')
            lines.append(r'    \midrule')

            prev_grp = None
            for lbl, grp, sc, unit in param_rows:
                if prev_grp is not None and grp != prev_grp:
                    lines.append(r'    \midrule')
                prev_grp = grp

                iau_ref  = iau_ref_in.get(lbl)
                row_parts = [_p2tex(lbl), _esc(grp), unit]

                for sn in grp_names:
                    if iau_ref is None:
                        row_parts.append('---')
                        continue
                    sd = sims.get(sn, {})
                    if 'parameter_history' not in sd or 'est_parameters' not in sd:
                        row_parts.append('---')
                        continue
                    ph_n         = sd['parameter_history']
                    lbls_n, _, _ = get_parameter_info(sd['est_parameters'])
                    if lbl not in lbls_n:
                        row_parts.append('---')
                        continue
                    final = ph_n[lbls_n.index(lbl), -1]
                    row_parts.append(_diff(final - iau_ref, sc))

                lines.append('    ' + ' & '.join(row_parts) + r' \\')

            lines.append(r'    \bottomrule')
            lines.append(rf'  \end{{tabular}}')
            lines.append(r'  }% end resizebox')
            lines.append(r'\end{table}')
            lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    sims, all_names = _get_data()
    names, labels   = _get_sims_and_labels(sims, all_names)

    if not names:
        sys.exit("ERROR: No simulations found.  Check SELECTED_SIMS and DATA_FILES.")

    # Create timestamped output directory.
    timestamp  = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir    = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Exporting {len(FIGURES_TO_EXPORT)} figure(s) | "
          f"{len(names)} simulation(s):")
    for n, l in zip(names, labels):
        print(f"  {n!r:50s}  →  {l!r}")
    print(f"Output directory: {out_dir}\n")

    for spec in FIGURES_TO_EXPORT:
        fn_name = spec[0]
        kwargs  = dict(spec[1]) if len(spec) > 1 else {}

        fn = _PLOT_REGISTRY.get(fn_name)
        if fn is None:
            print(f"  SKIP: unknown function '{fn_name}'")
            continue

        # fig_label is only for naming — pop before passing kwargs to fn.
        fig_label = kwargs.pop('fig_label', None)
        suffix    = (f'_{fig_label}' if fig_label
                     else (f'_{kwargs.get("variant", "")}' if kwargs.get('variant') else ''))

        # corr_heatmap operates on a single sim, not a list.
        if fn_name == 'corr_heatmap':
            sim_name = kwargs.pop('sim_name', None)
            if sim_name is None:
                sim_name = next(
                    (n for n in names if 'correlations' in sims.get(n, {})),
                    None)
            if sim_name is None:
                print("  SKIP: no simulation with correlation data.")
                continue
            fig = fn(sims, sim_name, **kwargs)
        else:
            fig = fn(sims, names, labels, **kwargs)

        if fig is None:
            print(f"  SKIP: {fn_name} returned None (data missing?)")
            continue

        out_path = os.path.join(out_dir, f'{fn_name}{suffix}.pdf')
        with PdfPages(out_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # Write parameter comparison tables (LaTeX).
    table_path = os.path.join(out_dir, 'parameter_tables.tex')
    generate_parameter_tables(sims, names, labels, table_path)

    # Write single-simulation detail tables (LaTeX).
    for sim_name in SINGLE_SIM_TABLES:
        safe  = sim_name.replace('_cov', '').replace('_', '-')
        tpath = os.path.join(out_dir, f'table_single_{safe}.tex')
        generate_single_sim_table(sims, sim_name, names, tpath)

    print(f"\nDone.  Output → {out_dir}")


if __name__ == '__main__':
    main()
