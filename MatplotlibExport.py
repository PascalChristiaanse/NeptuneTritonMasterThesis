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
import pandas as pd
from pathlib import Path

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
# CONFIGURATION  — set ACTIVE_5CONFIG to the desired ExportConfigs/*.py name
# ============================================================================

#ACTIVE_CONFIG = 'CASE1_Manual_Bias'          # Pole estimation — real obs., manual bias correction
#ACTIVE_CONFIG = 'WeightAnalysis'            # Weight scheme comparison (initial_state only)
#ACTIVE_CONFIG = 'WeightAnalysis_Old'        # Weight scheme comparison — old dataset
#ACTIVE_CONFIG = 'WeightAnalysis_Pole'       # Weight scheme comparison (initial_state + pole)
#ACTIVE_CONFIG = 'WeightComparison'          # Cross-dataset comparison of all three weight analyses
#ACTIVE_CONFIG = 'SimObs_ParameterAnalysis'  # Simulated-obs. parameter analysis (IAU vs Jacobson)
ACTIVE_CONFIG = 'ObservationalDataset'

import importlib
_cfg = importlib.import_module(f'ExportConfigs.{ACTIVE_CONFIG}')

DATASET_LABEL      = _cfg.DATASET_LABEL
SELECTED_SIMS      = _cfg.SELECTED_SIMS
SIM_LABELS         = _cfg.SIM_LABELS
OUTPUT_DIR         = _cfg.OUTPUT_DIR
FIGURES_TO_EXPORT  = _cfg.FIGURES_TO_EXPORT
SINGLE_SIM_TABLES  = _cfg.SINGLE_SIM_TABLES
_SIM_COLORS        = getattr(_cfg, 'SIM_COLORS',       {})
_SIM_LINESTYLE     = getattr(_cfg, 'SIM_LINESTYLE',    {})
_SIM_MARKERS       = getattr(_cfg, 'SIM_MARKERS',      {})
# Optional: maps sim-name prefix → group color for use_group_colors=True figures.
# e.g. {'IAUPole': '#0072B2', 'SimPole': '#D55E00'}
_SIM_GROUP_COLORS  = getattr(_cfg, 'SIM_GROUP_COLORS', {})


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
    """Return (selected_sim_names, display_labels) respecting configuration.

    Sims missing from the loaded data are dropped.  Labels stay aligned with
    names via index lookup into SIM_LABELS (using the original SELECTED_SIMS
    position), so missing sims do not shift label assignments.
    """
    selected = SELECTED_SIMS if SELECTED_SIMS is not None else list(all_names)
    # Build (name, label) pairs, skipping sims absent from the data.
    if SIM_LABELS is not None and len(SIM_LABELS) == len(selected):
        pairs = [(n, SIM_LABELS[i])
                 for i, n in enumerate(selected) if n in sims]
    else:
        pairs = [(n, n) for n in selected if n in sims]
    if not pairs:
        return [], []
    names, labels = zip(*pairs)
    return list(names), list(labels)


def _color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


def _marker(sn: str, fallback: str = 'o') -> str:
    return _SIM_MARKERS.get(sn, fallback)


def _group_color(sn: str, fallback_index: int = 0) -> str:
    """Return the group color for sim `sn` using _SIM_GROUP_COLORS lookup.

    Keys starting with '_' are treated as suffixes (sn.endswith(key));
    all other keys are treated as prefixes (sn.startswith(key)).
    Falls back to _SIM_COLORS then the default color cycle if no match is found.
    Used by figures with use_group_colors=True so that all sims in the same
    rotation-model group share one color while individual timeseries keep their
    own per-sim colors via _SIM_COLORS.
    """
    for key, col in _SIM_GROUP_COLORS.items():
        if key.startswith('_'):
            if sn.endswith(key):
                return col
        elif sn.startswith(key):
            return col
    return _SIM_COLORS.get(sn, _color(fallback_index))


# ============================================================================
# OBSERVATIONAL DATASET HELPERS
# ============================================================================

_TWO_PI        = 2.0 * np.pi
_RAD_TO_ARCSEC = 3600.0 * 180.0 / np.pi


def _get_observatory_name(obs_code: str) -> str:
    """Return the observatory name for a 3-digit MPC code from Observatories.txt.

    Pads single- or double-digit codes to 3 characters.  Returns the code
    itself if the file is not found or the code is not listed.
    """
    obs_code = obs_code.zfill(3)
    try:
        with open('Observations/Observatories.txt', 'r') as fh:
            for line in fh.readlines()[1:]:
                cols = line.split()
                if len(cols) >= 5 and cols[1] == obs_code:
                    # Format: Pl Code Lon Lat Alt rho_cos rho_sin region [name ...]
                    # Name starts at index 8 (index 7 = region string).
                    return ' '.join(cols[8:]) if len(cols) >= 9 else ' '.join(cols[7:])
    except FileNotFoundError:
        pass
    return obs_code


def _get_obs_type(nsdc_id: str,
                  raw_obs_folder=None,
                  obs_types_override: dict = None) -> str:
    """Return the observation type string ('Rel.' or 'Abs.') for an nsdc ID.

    Search order:
    1. *obs_types_override* dict (keyed by nsdc_id).
    2. Raw NSDC text files in each folder listed in *raw_obs_folder*.
       Accepts either a single path string or a list of path strings.
       Relative observations live in RawRelativeObservations/;
       absolute observations live in NeptuneObservations/.
    3. Falls back to '---' if no match is found.
    """
    if obs_types_override and nsdc_id in obs_types_override:
        return obs_types_override[nsdc_id]
    folders = []
    if raw_obs_folder:
        folders = [raw_obs_folder] if isinstance(raw_obs_folder, str) else list(raw_obs_folder)
    for folder in folders:
        raw_path = Path(folder) / f'{nsdc_id}.txt'
        if raw_path.exists():
            try:
                first_word = raw_path.read_text().split()[0].upper()
                if first_word == 'ABS':
                    return 'Abs.'
                if first_word in ('REL', 'SEP', 'DIF'):
                    return 'Rel.'
            except Exception:
                pass
    return '---'


def _load_spice_residuals_df(obs_folder: str) -> pd.DataFrame:
    """Load O-C residuals (vs NEP097) from all Triton_*.csv files in obs_folder.

    Returns a DataFrame with columns:
        ref_point_id     : str   — e.g. '689_nm0077'
        time_j2000       : float — seconds since J2000
        ra_resid_arcsec  : float — RA residual [arcsec]
        dec_resid_arcsec : float — Dec residual [arcsec]

    Convention in the CSV files:
        column 3 (O-C RA)  is stored as 2*pi + residual_rad  → subtract 2*pi
        column 4 (O-C Dec) is stored directly as residual_rad
    """
    rows = []
    for csv_path in sorted(Path(obs_folder).glob('Triton_*.csv')):
        parts = csv_path.stem.split('_')   # ['Triton', '<code>', '<nmXXXX>']
        if len(parts) < 3:
            continue
        ref_id = f'{parts[1]}_{parts[2]}'
        try:
            df_raw = pd.read_csv(csv_path)
        except Exception as exc:
            print(f'  WARNING: could not read {csv_path.name}: {exc}')
            continue
        times   = df_raw.iloc[:, 0].values
        ra_res  = (df_raw.iloc[:, 3].values - _TWO_PI) * _RAD_TO_ARCSEC
        dec_res =  df_raw.iloc[:, 4].values            * _RAD_TO_ARCSEC
        rows.append(pd.DataFrame({
            'ref_point_id':    ref_id,
            'time_j2000':      times,
            'ra_resid_arcsec': ra_res,
            'dec_resid_arcsec': dec_res,
        }))
    if not rows:
        return pd.DataFrame(columns=['ref_point_id', 'time_j2000',
                                     'ra_resid_arcsec', 'dec_resid_arcsec'])
    return pd.concat(rows, ignore_index=True)


def _apply_date_formatter(ax):
    """Tidy date formatter: major ticks every 5 years, rotated labels."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')


# ============================================================================
# PLOT FUNCTIONS
# ============================================================================

def mpl_rms_compare(sims, names, labels,
                    sim_subset=None,
                    use_group_colors=False,
                    title='RMS vs SPICE Comparison'):
    """Dot plot of final total RMS per simulation.

    use_group_colors is accepted for API consistency; colors are taken from
    _SIM_COLORS which already encodes group membership when set in the config.
    """
    if sim_subset is not None:
        pairs  = [(n, labels[names.index(n)]) for n in sim_subset if n in sims and n in names]
        names  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]
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
            col = (_group_color(names[i], i) if use_group_colors
                   else _SIM_COLORS.get(names[i], _color(i)))
            mk  = _marker(names[i])
            ax.plot(xi, v, marker=mk, color=col, markersize=8,
                    markeredgecolor='black', markeredgewidth=0.5,
                    linestyle='none', zorder=2)
            ax.text(xi, v, f'  {v:.3f}',
                    ha='left', va='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('RMS [km]')
    ax.set_title(title)
    ax.set_xlim(-0.6, len(names) - 0.4)
    fig.tight_layout()
    return fig


def mpl_rms_formal(sims, names, labels,
                   sim_subset=None,
                   use_group_colors=False,
                   title='RMS of Formal Errors (RSW)'):
    """Dot plot of total formal-error RMS per simulation.

    Total formal RMS = sqrt(mean(formal_errors_RSW_km ** 2)) over all R/S/W
    components — the scalar counterpart to rms_SPICE from mpl_rms_compare.
    use_group_colors is accepted for API consistency but has no effect beyond
    what is already encoded in _SIM_COLORS.
    """
    if sim_subset is not None:
        pairs  = [(n, labels[names.index(n)]) for n in sim_subset if n in sims and n in names]
        names  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

    vals = []
    for sn in names:
        sd = sims.get(sn, {})
        if 'formal_errors_RSW_km' in sd:
            vals.append(float(np.sqrt(np.mean(sd['formal_errors_RSW_km'] ** 2))))
        else:
            vals.append(np.nan)

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(FIG_W_DOUBLE, 0.8 * len(names)), FIG_H_DEFAULT))

    valid_x = [xi for xi, v in enumerate(vals) if not np.isnan(v)]
    valid_v = [v  for v  in vals               if not np.isnan(v)]
    if len(valid_x) > 1:
        ax.plot(valid_x, valid_v, '-', color='gray', linewidth=1.0, alpha=0.5, zorder=1)

    for i, (xi, v) in enumerate(zip(x, vals)):
        if not np.isnan(v):
            col = (_group_color(names[i], i) if use_group_colors
                   else _SIM_COLORS.get(names[i], _color(i)))
            mk  = _marker(names[i])
            ax.plot(xi, v, marker=mk, color=col, markersize=8,
                    markeredgecolor='black', markeredgewidth=0.5,
                    linestyle='none', zorder=2)
            ax.text(xi, v, f'  {v:.3f}', ha='left', va='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel(r'Formal Error RMS [km]')
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
                  show_formal=True,
                  show_diff=True,
                  fontsize_scale=1.0,
                  title='RSW Statistics',
                  figsize=None):
    """Grid of RSW statistics: cols = R / S / W.

    Row selection is controlled by show_diff and show_formal:
      show_diff=True,  show_formal=True  → 5×3 grid (diff Mean/RMS/Max + formal Max/RMS)
      show_diff=True,  show_formal=False → 3×3 grid (diff rows only)
      show_diff=False, show_formal=True  → 2×3 grid (formal rows only)

    fontsize_scale multiplies the default tick/label/annotation font sizes,
    making the figure easier to read when many simulations are compared.

    A line connects simulation points to show the trend across simulations."""
    if sim_subset is not None:
        pairs  = [(n, labels[names.index(n)]) for n in sim_subset if n in sims and n in names]
        names  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

    # (row_label, data_source, stat_key, y_unit)
    diff_row_defs = [
        ('RSW diff — Mean',  'diff',   'mean', 'km'),
        ('RSW diff — RMS',   'diff',   'rms',  'km'),
        ('RSW diff — Max',   'diff',   'max',  'km'),
    ]
    formal_row_defs = [
        ('Formal σ — Max',   'formal', 'max',  'km'),
        ('Formal σ — RMS',   'formal', 'rms',  'km'),
    ]
    row_defs = []
    if show_diff:
        row_defs += diff_row_defs
    if show_formal:
        row_defs += formal_row_defs

    if not row_defs:
        print("WARNING: mpl_rsw_stats called with show_diff=False and show_formal=False — nothing to plot.")
        return None

    comps = ['R', 'S', 'W']
    x = np.arange(len(names))

    fs_tick   = max(6, int(8  * fontsize_scale))
    fs_annot  = max(5, int(7  * fontsize_scale))
    fs_ylabel = max(7, int(9  * fontsize_scale))
    fs_title  = max(9, int(11 * fontsize_scale))

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.8, FIG_H_DEFAULT * len(row_defs) / 1.5)

    fig, axes = plt.subplots(len(row_defs), 3, figsize=figsize)
    if len(row_defs) == 1:
        axes = axes.reshape(1, -1)

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
                    ax.plot(xi, v, marker=_marker(names[xi]), color=col, markersize=6,
                            markeredgecolor='black', markeredgewidth=0.4,
                            linestyle='none', zorder=2)
                    ax.text(xi, v, f'  {v:.2f}',
                            ha='left', va='center', fontsize=fs_annot)

            ax.set_xticks(x)
            ax.set_xlim(-0.5, len(names) - 0.5)
            if ri == len(row_defs) - 1:
                ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=fs_tick)
            else:
                ax.set_xticklabels([])
            if ci == 0:
                ax.set_ylabel(f'{row_label} [{unit}]', fontsize=fs_ylabel)
            if ri == 0:
                ax.set_title(comp, fontsize=fs_title)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def mpl_rms_ratio(sims, names, labels,
                  use_group_colors=False,
                  title='RMS / Formal Error RMS'):
    """Dot plot of total RMS vs SPICE divided by total formal error RMS per simulation.

    use_group_colors is accepted for API consistency; colors are taken from
    _SIM_COLORS which already encodes group membership when set in the config.

    Total formal error RMS is computed as sqrt(mean(formal_errors_RSW_km ** 2))
    across all R/S/W components, matching the scalar nature of rms_SPICE.
    """
    ratios = []
    for sn in names:
        sd = sims.get(sn, {})
        rms_spice = sd.get('rms_SPICE', np.nan)
        if 'formal_errors_RSW_km' in sd and not np.isnan(rms_spice):
            formal_rms = np.sqrt(np.mean(sd['formal_errors_RSW_km'] ** 2))
            ratios.append(rms_spice / formal_rms if formal_rms > 0 else np.nan)
        else:
            ratios.append(np.nan)

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(FIG_W_DOUBLE, 0.8 * len(names)), FIG_H_DEFAULT))

    valid_x = [xi for xi, v in enumerate(ratios) if not np.isnan(v)]
    valid_v = [v  for v  in ratios               if not np.isnan(v)]
    if len(valid_x) > 1:
        ax.plot(valid_x, valid_v, '-', color='gray', linewidth=1.0, alpha=0.5, zorder=1)

    for i, (xi, v) in enumerate(zip(x, ratios)):
        if not np.isnan(v):
            col = (_group_color(names[i], i) if use_group_colors
                   else _SIM_COLORS.get(names[i], _color(i)))
            mk  = _marker(names[i])
            ax.plot(xi, v, marker=mk, color=col, markersize=8,
                    markeredgecolor='black', markeredgewidth=0.5,
                    linestyle='none', zorder=2)
            ax.text(xi, v, f'  {v:.2f}', ha='left', va='center', fontsize=9)

    ax.axhline(1.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6,
               label='ratio = 1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('RMS$_{\\mathrm{SPICE}}$ / RMS$_{\\sigma}$  [—]')
    ax.set_title(title)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def mpl_rsw_ratio(sims, names, labels,
                  sim_subset=None,
                  title='RSW RMS / Formal σ RMS'):
    """3-subplot dot plot: per-direction ratio of RSW diff RMS to formal error RMS.

    For each component c ∈ {R, S, W}:
        ratio_c = RMS(diff_SPICE_RSW[:, c]) / RMS(formal_errors_RSW_km[:, c])
    """
    if sim_subset is not None:
        pairs  = [(n, labels[names.index(n)]) for n in sim_subset if n in sims and n in names]
        names  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

    comps = ['R', 'S', 'W']
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_DOUBLE * 1.6, FIG_H_DEFAULT))

    for ci, (ax, comp) in enumerate(zip(axes, comps)):
        col = RSW_COLORS[comp]
        ratios = []
        for sn in names:
            sd = sims.get(sn, {})
            if 'diff_SPICE_RSW' in sd and 'formal_errors_RSW_km' in sd:
                diff_rms   = compute_rsw_statistics(sd['diff_SPICE_RSW'])[comp]['rms']
                formal_rms = compute_formal_error_statistics(sd['formal_errors_RSW_km'])[comp]['rms']
                ratios.append(diff_rms / formal_rms if formal_rms > 0 else np.nan)
            else:
                ratios.append(np.nan)

        valid_x = [xi for xi, v in enumerate(ratios) if not np.isnan(v)]
        valid_v = [v  for v  in ratios               if not np.isnan(v)]
        if len(valid_x) > 1:
            ax.plot(valid_x, valid_v, '-', color=col, linewidth=1.2, alpha=0.5, zorder=1)

        for xi, v in enumerate(ratios):
            if not np.isnan(v):
                ax.plot(xi, v, marker=_marker(names[xi]), color=col, markersize=6,
                        markeredgecolor='black', markeredgewidth=0.4,
                        linestyle='none', zorder=2)
                ax.text(xi, v, f'  {v:.2f}', ha='left', va='center', fontsize=7)

        ax.axhline(1.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        ax.set_title(comp, fontsize=11)
        ax.set_xlim(-0.5, len(names) - 0.5)
        if ci == 0:
            ax.set_ylabel('RMS$_{\\mathrm{SPICE}}$ / RMS$_{\\sigma}$  [—]')

    fig.suptitle(title)
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

    # Connecting lines (neutral color, behind markers)
    valid_f = [(xi, v) for xi, v in enumerate(fv) if v is not None]
    if len(valid_f) > 1:
        ax.plot([p[0] for p in valid_f], [p[1] for p in valid_f],
                '-', color='steelblue', linewidth=1.0, alpha=0.4, zorder=1)
    if show_initial:
        valid_i = [(xi, v) for xi, v in enumerate(iv) if v is not None]
        if len(valid_i) > 1:
            ax.plot([p[0] for p in valid_i], [p[1] for p in valid_i],
                    '--', color='lightcoral', linewidth=1.0, alpha=0.4, zorder=1)

    # Per-sim markers
    for xi, sn in enumerate(avs):
        col = _SIM_COLORS.get(sn, _color(xi))
        mk  = _marker(sn)
        if fv[xi] is not None:
            ax.plot(xi, fv[xi], marker=mk, color=col, markersize=7,
                    markeredgecolor='black', markeredgewidth=0.5,
                    linestyle='none', zorder=2, label=avlabels[xi])
        if show_initial and iv[xi] is not None:
            ax.plot(xi, iv[xi], marker=mk, color=col, markersize=5,
                    markeredgecolor='black', markeredgewidth=0.5,
                    linestyle='none', zorder=2, alpha=0.45,
                    markerfacecolor='none')

    # Manual legend entries for initial/final if show_initial
    if show_initial:
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], linestyle='-',  color='steelblue',   linewidth=1.2, label='Final'),
            Line2D([0], [0], linestyle='--', color='lightcoral',  linewidth=1.2, label='Initial'),
        ])

    ax.set_xticks(x)
    ax.set_xticklabels(avlabels, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale('log')

    fig.tight_layout()
    return fig


def mpl_gof_combined(sims, names, labels,
                     show_initial=False,
                     log_y=False,
                     thick_lines=False,
                     use_group_colors=False,
                     title='Goodness of Fit Comparison',
                     figsize=None):
    """1×3 subplots: WRMS [mas], RMS [mas], Cost Function — all simulations.

    Parameters
    ----------
    thick_lines : bool
        If True, connecting lines are drawn thicker (lw=2.0, alpha=0.65) so the
        initial-value line remains visible when it overlaps with the final line.
    use_group_colors : bool
        If True, the show_initial legend is replaced with a two-entry group
        legend (IAUPole / FitPole) instead of one entry per simulation.
        Colors are still taken from _SIM_COLORS (which should already encode
        group membership in the config's SIM_COLORS dict).
    """
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

    line_lw    = 2.0 if thick_lines else 1.0
    line_alpha = 0.65 if thick_lines else 0.4

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.4, FIG_H_DEFAULT * len(metrics) / 1.5)

    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)

    for i, (ax, (fk, ik, ylabel)) in enumerate(zip(axes, metrics)):
        fv = [wm[n].get(fk) for n in avs]
        iv = [wm[n].get(ik) for n in avs]

        # Connecting lines (neutral, behind markers)
        valid_f = [(xi, v) for xi, v in enumerate(fv) if v is not None]
        if len(valid_f) > 1:
            ax.plot([p[0] for p in valid_f], [p[1] for p in valid_f],
                    '-', color='steelblue', linewidth=line_lw, alpha=line_alpha, zorder=1)
        if show_initial:
            valid_i = [(xi, v) for xi, v in enumerate(iv) if v is not None]
            if len(valid_i) > 1:
                ax.plot([p[0] for p in valid_i], [p[1] for p in valid_i],
                        '--', color='lightcoral', linewidth=line_lw, alpha=line_alpha, zorder=1)

        # Per-sim markers
        for xi, sn in enumerate(avs):
            col = (_group_color(sn, xi) if use_group_colors
                   else _SIM_COLORS.get(sn, _color(xi)))
            mk  = _marker(sn)
            if fv[xi] is not None:
                ax.plot(xi, fv[xi], marker=mk, color=col, markersize=7,
                        markeredgecolor='black', markeredgewidth=0.5,
                        linestyle='none', zorder=2)
            if show_initial and iv[xi] is not None:
                ax.plot(xi, iv[xi], marker=mk, color=col, markersize=5,
                        markeredgecolor='black', markeredgewidth=0.5,
                        linestyle='none', zorder=2, alpha=0.45,
                        markerfacecolor='none')

        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=10)
        if log_y:
            ax.set_yscale('log')

    # x-tick labels only on bottom subplot
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(avlabels, rotation=30, ha='right')

    if show_initial:
        from matplotlib.lines import Line2D
        if use_group_colors:
            # Build a group legend: one entry per unique group color + marker
            seen_groups = {}
            for sn in avs:
                col = _group_color(sn)
                mk  = _marker(sn)
                key = (col, mk)
                if key not in seen_groups:
                    grp_lbl = ('IAUPole' if sn.startswith('IAUPole')
                               else 'FitPole' if sn.startswith(('FitPole', 'SimPole'))
                               else sn.split('_')[0])
                    seen_groups[key] = grp_lbl
            group_handles = [
                Line2D([0], [0], color=col, marker=mk, linestyle='-',
                       linewidth=1.5, markersize=7,
                       markeredgecolor='black', markeredgewidth=0.4,
                       label=lbl)
                for (col, mk), lbl in seen_groups.items()
            ]
            group_handles += [
                Line2D([0], [0], linestyle='-',  color='steelblue',  linewidth=line_lw, label='Final'),
                Line2D([0], [0], linestyle='--', color='lightcoral', linewidth=line_lw, label='Initial'),
            ]
            axes[0].legend(handles=group_handles, fontsize=9)
        else:
            axes[0].legend(handles=[
                Line2D([0], [0], linestyle='-',  color='steelblue',  linewidth=line_lw, label='Final'),
                Line2D([0], [0], linestyle='--', color='lightcoral', linewidth=line_lw, label='Initial (hollow)'),
            ], fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def mpl_corr_heatmap(sims, sim_name,
                     title=None, figsize=None, cmap='YlOrRd'):
    """Absolute correlation matrix heatmap for a single simulation.

    Parameters
    ----------
    cmap : str
        Matplotlib colormap name.  Default 'YlOrRd' (yellow→red, colorblind-safe).
        Use 'Blues', 'viridis', or any valid colormap string.
    """
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
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    fig.colorbar(im, ax=ax, label='|Correlation|', shrink=0.8)

    fontsize = max(5, 9 - n // 3)
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            ax.text(j, i, f'{v:.2f}',
                    ha='center', va='center',
                    fontsize=fontsize,
                    color='white' if v > 0.6 else 'black')

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


def mpl_residual_timeseries(sims, names, labels,
                            sim_subset=None,
                            title='Observation Residuals RA/DEC',
                            figsize=None):
    """Final-iteration RA / Dec residuals [mas] as a scatter, coloured by ref_point_id.

    One figure per simulation in sim_subset.  Layout: 2 stacked rows (RA top,
    Dec bottom) sharing the x-axis (observation datetime).
    """
    if figsize is None:
        figsize = (FIG_W_DOUBLE, 6.0)

    targets = sim_subset if sim_subset is not None else names
    figs = []

    for sn in targets:
        if sn not in sims:
            print(f"  SKIP residual_timeseries for '{sn}': not in sims.")
            continue
        sd = sims[sn]
        if 'residual_df' not in sd:
            print(f"  SKIP residual_timeseries for '{sn}': no residual_df.")
            continue
        df  = sd['residual_df']
        lbl = labels[names.index(sn)] if sn in names else sn

        fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        ref_ids = sorted(df['ref_point_id'].unique())
        for i, rid in enumerate(ref_ids):
            mask = df['ref_point_id'] == rid
            t    = df.loc[mask, 'datetime']
            ra   = df.loc[mask, 'ra_residual_final_mas']
            dec  = df.loc[mask, 'dec_residual_final_mas']
            col  = _color(i)
            ax_ra.scatter(t,   ra,  s=4, color=col, alpha=0.5, label=str(rid))
            ax_dec.scatter(t,  dec, s=4, color=col, alpha=0.5)

        for ax in (ax_ra, ax_dec):
            ax.axhline(0, color='k', linewidth=0.8, linestyle='--')

        ax_ra.set_ylabel('RA Residual [mas]')
        ax_dec.set_ylabel('Dec Residual [mas]')
        ax_dec.set_xlabel('Date')
        ax_ra.legend(title='Obs. file ID', fontsize=7, ncol=2,
                     loc='upper right', markerscale=2)
        _apply_date_formatter(ax_dec)
        fig.suptitle(f'{title} — {lbl}')
        fig.tight_layout()
        figs.append(fig)

    if not figs:
        return None
    return figs[0] if len(figs) == 1 else figs


# ============================================================================
# PARAMETER UPDATE HELPERS
# ============================================================================

def _get_iau_reference(sims, all_names):
    """Build reference dicts from IAUPole_pole_pos_cov_pole_lib_cov initial values.

    This sim always estimates [initial_state, iau_rotation_model_pole,
    iau_rotation_model_pole_librations] in that fixed order, giving a complete
    reference for all parameter types (state + pole pos + pole lib).

    For FitPole sims that do not estimate all parameters, _param_update falls
    back to the sim's own initial value for any label not present here.

    Returns (iau_ref_inertial, iau_ref_rsw) — both are dicts mapping
    parameter label → initial value.  Returns empty dicts if the reference sim
    is not found or has no parameter data.
    """
    REF_SIM = 'IAUPole_pole_pos_cov_pole_lib_cov'
    iau_ref_inertial = {}
    iau_ref_rsw      = {}
    rsw_map = {'X': 'R', 'Y': 'S', 'Z': 'W', 'VX': 'VR', 'VY': 'VS', 'VZ': 'VW'}

    sd = sims.get(REF_SIM, {})
    if 'parameter_history' not in sd or 'est_parameters' not in sd:
        print(f"WARNING: Reference sim '{REF_SIM}' not found or has no parameter data. "
              "Parameter update plots will use each sim's own initial values as reference.")
        return iau_ref_inertial, iau_ref_rsw

    ph_in    = sd['parameter_history']
    ph_rsw   = sd.get('parameter_history_RSW', ph_in)
    lbls, _, _ = get_parameter_info(sd['est_parameters'])
    rsw_lbls = [rsw_map.get(l, l) for l in lbls]

    for i, lbl in enumerate(lbls):
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
                    variant='iau',
                    sim_subset=None):
    """1×2 bar chart: |Δpos| [km] and |Δvel| [km/s] across simulations."""
    iau_prefix = 'IAUPole'
    fit_prefix = 'FitPole'
    iau_ref_in, _ = _get_iau_reference(sims, names)

    if variant == 'iau':
        f_names      = [n for n in names if n.startswith(iau_prefix)]
        ref_map      = {n: None for n in f_names}
        title_suffix = ' — IAU'
    elif variant in ('fitpole', 'simpole'):
        f_names      = [n for n in names if not n.startswith(iau_prefix)]
        ref_map      = {n: None for n in f_names}
        title_suffix = f' — {fit_prefix}'
    else:  # combined
        f_names      = list(names)
        ref_map      = {n: (iau_ref_in if not n.startswith(iau_prefix) else None)
                        for n in f_names}
        title_suffix = f' — Combined ({fit_prefix} vs IAU ref)'

    if sim_subset is not None:
        f_names = [n for n in f_names if n in sim_subset]
        ref_map = {n: ref_map[n] for n in f_names}

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
                  variant='iau',
                  sim_subset=None):
    """2×3 bar-chart grid: ΔR/ΔS/ΔW (top) and ΔVR/ΔVS/ΔVW (bottom) [km, km/s]."""
    iau_prefix = 'IAUPole'
    fit_prefix = 'FitPole'
    _, iau_ref_rsw = _get_iau_reference(sims, names)

    if variant == 'iau':
        f_names      = [n for n in names if n.startswith(iau_prefix)]
        ref_map      = {n: None for n in f_names}
        title_suffix = ' — IAU'
    elif variant in ('fitpole', 'simpole'):
        f_names      = [n for n in names if not n.startswith(iau_prefix)]
        ref_map      = {n: None for n in f_names}
        title_suffix = f' — {fit_prefix}'
    else:  # combined
        f_names      = list(names)
        ref_map      = {n: (iau_ref_rsw if not n.startswith(iau_prefix) else None)
                        for n in f_names}
        title_suffix = f' — Combined ({fit_prefix} vs IAU ref)'

    if sim_subset is not None:
        f_names = [n for n in f_names if n in sim_subset]
        ref_map = {n: ref_map[n] for n in f_names}

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
                       variant='iau',
                       sim_subset=None):
    """1×2 bar chart: Δα₀ and Δδ₀ across simulations [deg]."""
    iau_prefix = 'IAUPole'
    fit_prefix = 'FitPole'
    iau_ref_in, _ = _get_iau_reference(sims, names)

    if variant == 'iau':
        base_names   = [n for n in names if n.startswith(iau_prefix)]
        ref_map      = {n: None for n in base_names}
        title_suffix = ' — IAU'
    elif variant in ('fitpole', 'simpole'):
        base_names   = [n for n in names if not n.startswith(iau_prefix)]
        ref_map      = {n: None for n in base_names}
        title_suffix = f' — {fit_prefix}'
    else:  # combined
        base_names   = list(names)
        ref_map      = {n: (iau_ref_in if not n.startswith(iau_prefix) else None)
                        for n in base_names}
        title_suffix = f' — Combined ({fit_prefix} vs IAU ref)'

    if sim_subset is not None:
        base_names = [n for n in base_names if n in sim_subset]
        ref_map    = {n: ref_map[n] for n in base_names if n in ref_map}
        title_suffix = ''

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
                       variant='iau',
                       sim_subset=None):
    """1×2 bar chart: Δα₁ and Δδ₁ across simulations [deg]."""
    iau_prefix = 'IAUPole'
    fit_prefix = 'FitPole'
    iau_ref_in, _ = _get_iau_reference(sims, names)

    if variant == 'iau':
        base_names   = [n for n in names if n.startswith(iau_prefix)]
        ref_map      = {n: None for n in base_names}
        title_suffix = ' — IAU'
    elif variant in ('fitpole', 'simpole'):
        base_names   = [n for n in names if not n.startswith(iau_prefix)]
        ref_map      = {n: None for n in base_names}
        title_suffix = f' — {fit_prefix}'
    else:  # combined
        base_names   = list(names)
        ref_map      = {n: (iau_ref_in if not n.startswith(iau_prefix) else None)
                        for n in base_names}
        title_suffix = f' — Combined ({fit_prefix} vs IAU ref)'

    if sim_subset is not None:
        base_names = [n for n in base_names if n in sim_subset]
        ref_map    = {n: ref_map[n] for n in base_names if n in ref_map}
        title_suffix = ''

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

    Uses _SIM_COLORS / _SIM_LINESTYLE (IAU = cool, FitPole = warm).
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

    Typical use: pass one IAUPole sim and one FitPole sim to compare starting points.
    If sim_subset is None, auto-selects the first IAUPole and first FitPole sim found.
    unit: 'km' (default) or 'm' — data is stored in km, 'm' multiplies by 1000.
    """
    # ── resolve which sims to plot ────────────────────────────────────────────
    if sim_subset is not None:
        pairs = [
            (n, labels[names.index(n)] if n in names else n.replace('_cov', ''))
            for n in sim_subset if n in sims
        ]
    else:
        # Auto-pick first IAUPole and first FitPole (or SimPole) that have initial data
        auto = []
        iau_found = fit_found = False
        for n in names:
            sd = sims.get(n, {})
            if not iau_found and n.startswith('IAUPole') and 'diff_SPICE_RSW_initial' in sd:
                auto.append((n, labels[names.index(n)]))
                iau_found = True
            elif not fit_found and not n.startswith('IAUPole') and 'diff_SPICE_RSW_initial' in sd:
                auto.append((n, labels[names.index(n)]))
                fit_found = True
            if iau_found and fit_found:
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


# ── Standalone legend figure ──────────────────────────────────────────────────

def mpl_legend(sims, names, labels,
               title='Simulation Legend',
               ncols=2,
               shape_labels=None,
               notes=None,
               figsize=None):
    """Standalone legend figure for use across thesis figures.

    Produces a two-section legend:
      • Shape key  — marker shape encodes the rotation model (IAU vs FitPole).
      • Color key  — one entry per simulation with its color, marker, linestyle.
    Optionally adds an abbreviation glossary below the legend box.

    Parameters
    ----------
    shape_labels : dict, optional
        Maps marker string → human-readable label for the shape-key section.
        Default: {'o': 'IAU 2015 rotation model',
                  'D': 'FitPole rotation model'}.
    notes : list of str, optional
        Lines of explanatory text rendered as a monospaced glossary below the
        legend box (e.g. abbreviation expansions).  Each string is one line.
    ncols : int
        Number of columns in the legend (default 2).
    """
    from matplotlib.lines import Line2D

    if shape_labels is None:
        shape_labels = {
            'o': 'IAU 2015 rotation model',
            'D': 'FitPole rotation model',
        }

    # ── Collect unique marker shapes that actually appear in this dataset ──
    used_markers = dict.fromkeys(                       # ordered, deduped
        _marker(sn) for sn in names
    )
    shape_handles = []
    for mk in used_markers:
        lbl = shape_labels.get(mk, f'marker: {mk}')
        ls  = '-' if mk == 'o' else '--'                # mirror linestyle convention
        shape_handles.append(
            Line2D([0], [0], color='#555555', marker=mk, linestyle=ls,
                   linewidth=1.5, markersize=9,
                   markeredgecolor='black', markeredgewidth=0.5,
                   label=lbl)
        )

    # ── Per-simulation entries ─────────────────────────────────────────────
    sim_handles = []
    for i, (sn, lbl) in enumerate(zip(names, labels)):
        col = _SIM_COLORS.get(sn, _color(i))
        mk  = _marker(sn)
        ls  = _SIM_LINESTYLE.get(sn, '-')
        sim_handles.append(
            Line2D([0], [0], color=col, marker=mk, linestyle=ls,
                   linewidth=1.5, markersize=8,
                   markeredgecolor='black', markeredgewidth=0.4,
                   label=lbl)
        )

    # ── Combine: shape key first, blank separator, then per-sim entries ───
    sep = Line2D([], [], linestyle='none', label='')
    all_handles = shape_handles + [sep] + sim_handles
    all_labels  = [h.get_label() for h in all_handles]

    # ── Figure size: auto-scale to legend rows + optional notes block ─────
    notes_lines = len(notes) if notes else 0
    if figsize is None:
        n_rows = -(-len(all_handles) // ncols)          # ceiling division
        figsize = (FIG_W_DOUBLE * max(1, ncols / 2),
                   0.42 * n_rows + 0.9 + 0.22 * notes_lines)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # ── Notes block — rendered below the legend in a fixed-width font ─────
    notes_height = 0.22 * notes_lines / figsize[1] if notes_lines else 0.0
    legend_bottom = notes_height + 0.02            # legend sits above the notes

    if notes:
        notes_text = '\n'.join(notes)
        fig.text(0.04, notes_height / 2,           # vertically centred in notes strip
                 notes_text,
                 ha='left', va='center',
                 fontsize=9,
                 fontfamily='monospace',
                 transform=fig.transFigure)

    ax.legend(handles=all_handles, labels=all_labels,
              loc='center',
              bbox_to_anchor=(0.5, (legend_bottom + 1.0) / 2),
              ncol=ncols,
              fontsize=10,
              frameon=True,
              framealpha=0.8,
              title=title,
              title_fontsize=11,
              handlelength=2.5,
              handleheight=1.0,
              borderpad=0.9,
              labelspacing=0.55)
    fig.tight_layout(rect=[0, notes_height, 1, 1])
    return fig


# ── Initial propagation difference: (sim_a − sim_b) ──────────────────────────

def mpl_rsw_initial_diff(sims, names, labels,
                         sim_subset=None,
                         title='RSW Initial Propagation Difference vs NEP097',
                         figsize=None):
    """3-row time series of (sim_a − sim_b) initial RSW difference.

    sim_subset must be a list of exactly two simulation names:
        [sim_a, sim_b]  →  diff = diff_SPICE_RSW_initial[sim_a] − diff_SPICE_RSW_initial[sim_b]

    Both sims must have diff_SPICE_RSW_initial and time_column_initial.
    Arrays are trimmed to the shorter length if they differ.
    """
    if sim_subset is None or len(sim_subset) < 2:
        print("WARNING: rsw_initial_diff requires sim_subset=[sim_a, sim_b], skipping.")
        return None

    sn_a, sn_b = sim_subset[0], sim_subset[1]

    def _get_rsw(sn):
        """Return RSW array, preferring initial diff then falling back to final."""
        sd = sims.get(sn, {})
        for key in ('diff_SPICE_RSW_initial', 'diff_SPICE_RSW'):
            if key in sd:
                return sd[key]
        return None

    dr_a = _get_rsw(sn_a)
    dr_b = _get_rsw(sn_b)
    for sn, dr in ((sn_a, dr_a), (sn_b, dr_b)):
        if dr is None:
            print(f"WARNING: '{sn}' has no RSW diff data, skipping rsw_initial_diff.")
            return None

    lbl_a = labels[names.index(sn_a)] if sn_a in names else sn_a
    lbl_b = labels[names.index(sn_b)] if sn_b in names else sn_b

    t_col = sims[sn_a].get('time_column_initial') or sims[sn_a].get('time_column')
    if t_col is not None:
        times = convert_time_array_to_datetime(t_col.reshape(-1, 1))
    else:
        times = list(range(len(dr_a)))

    # Trim to common length
    n_pts   = min(len(dr_a), len(dr_b), len(times))
    dr_diff = dr_a[:n_pts] - dr_b[:n_pts]
    times   = times[:n_pts]

    if figsize is None:
        figsize = (FIG_W_DOUBLE, 7.0)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    ylabels = [
        r'$\Delta R_{\,\mathrm{IAU}-\mathrm{Jac.}}$ [km]',
        r'$\Delta S_{\,\mathrm{IAU}-\mathrm{Jac.}}$ [km]',
        r'$\Delta W_{\,\mathrm{IAU}-\mathrm{Jac.}}$ [km]',
    ]

    diff_color = '#7B2D8B'  # purple — neutral, not associated with either model

    for row, (ax, ylab) in enumerate(zip(axes, ylabels)):
        ax.plot(times, dr_diff[:, row],
                color=diff_color, linewidth=1.8, alpha=0.92)
        ax.axhline(0, color='gray', linewidth=0.6, linestyle=':')
        ax.set_ylabel(ylab, fontsize=11)
        _apply_date_formatter(ax)

    axes[0].set_title(title)
    axes[0].text(0.01, 0.97, f'({lbl_a}) − ({lbl_b})',
                 transform=axes[0].transAxes,
                 ha='left', va='top', fontsize=9, style='italic', color='gray')
    axes[-1].set_xlabel('Date', fontsize=11)
    fig.tight_layout()
    return fig


# ── Gravitational parameter update bar chart ──────────────────────────────────

def mpl_param_gm(sims, names, labels,
                 sim_subset=None,
                 title='Gravitational Parameter Update',
                 figsize=None):
    """1×2 bar chart: ΔGM_Nep [m³/s²] and ΔGM_Tri [m³/s²] across simulations.

    Keeps only simulations that estimate at least one GM parameter (group='Gravity').
    Each sim uses its own initial value as reference — appropriate for SimObs analysis
    where no external IAU/SimPole reference is defined.
    """
    if sim_subset is not None:
        pairs  = [(n, labels[names.index(n)]) for n in sim_subset if n in sims and n in names]
        names  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

    # Keep only sims that estimate at least one GM parameter
    filt = [(sn, labels[names.index(sn)]) for sn in names
            if _param_update(sims.get(sn, {}), ['Gravity'])[0] is not None]
    if not filt:
        print("WARNING: No simulations with GM parameters found, skipping mpl_param_gm.")
        return None
    f_names, f_labels = zip(*filt)

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.2, FIG_H_DEFAULT)

    gm_colors = {'GM_Nep': '#0072B2', 'GM_Tri': '#D55E00'}
    fig, (ax_n, ax_t) = plt.subplots(1, 2, figsize=figsize)

    for ax, plabel in zip((ax_n, ax_t), ('GM_Nep', 'GM_Tri')):
        vals = []
        for sn in f_names:
            v_all, pl, _ = _param_update(sims.get(sn, {}), ['Gravity'])
            vals.append(v_all[pl.index(plabel)] if (pl is not None and plabel in pl) else np.nan)
        _bar_ax(ax, vals, gm_colors[plabel], f_labels,
                r'm$^3$/s$^2$', plabel, fontsize_annot=7)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


# ── Pole rotation rate update bar chart ───────────────────────────────────────

def mpl_param_pole_rate(sims, names, labels,
                        sim_subset=None,
                        title='Pole Rotation Rate Update  (α̇₀, δ̇₀)',
                        figsize=None,
                        variant='combined'):
    """1×2 bar chart: Δα̇₀ and Δδ̇₀ across simulations [deg].

    Keeps only simulations that estimate at least one pole-rate parameter.
    Scale: _RAD_TO_DEG (values are stored in rad/s; display shows the raw
    radian→degree scaled update for comparison across sims).
    """
    iau_prefix = 'IAUPole'
    fit_prefix = 'FitPole'
    iau_ref_in, _ = _get_iau_reference(sims, names)

    if variant == 'iau':
        base_names   = [n for n in names if n.startswith(iau_prefix)]
        ref_map      = {n: None for n in base_names}
        title_suffix = ' — IAU'
    elif variant in ('fitpole', 'simpole'):
        base_names   = [n for n in names if not n.startswith(iau_prefix)]
        ref_map      = {n: None for n in base_names}
        title_suffix = f' — {fit_prefix}'
    else:  # combined
        base_names   = list(names)
        ref_map      = {n: (iau_ref_in if not n.startswith(iau_prefix) else None)
                        for n in base_names}
        title_suffix = f' — Combined ({fit_prefix} vs IAU ref)'

    if sim_subset is not None:
        base_names = [n for n in base_names if n in sim_subset]
        ref_map    = {n: ref_map[n] for n in base_names if n in ref_map}
        title_suffix = ''

    title = title + title_suffix

    filt = [(sn, labels[names.index(sn)]) for sn in base_names
            if _param_update(sims.get(sn, {}), ['Pole Rate'])[0] is not None]
    if not filt:
        print("WARNING: No simulations with pole-rate parameters found, skipping mpl_param_pole_rate.")
        return None
    f_names, f_labels = zip(*filt)

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.2, FIG_H_DEFAULT)

    rate_colors = {'α̇₀': '#4A148C', 'δ̇₀': '#7B1FA2'}
    fig, (ax_a, ax_d) = plt.subplots(1, 2, figsize=figsize)

    for ax, plabel in zip((ax_a, ax_d), ('α̇₀', 'δ̇₀')):
        vals = []
        for sn in f_names:
            v_all, pl, _ = _param_update(
                sims.get(sn, {}), ['Pole Rate'],
                scale=_RAD_TO_DEG, iau_ref=ref_map[sn])
            vals.append(v_all[pl.index(plabel)] if (pl is not None and plabel in pl) else np.nan)
        _bar_ax(ax, vals, rate_colors[plabel], f_labels, 'deg', plabel, fontsize_annot=8)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


# ── Spherical harmonics update bar chart ──────────────────────────────────────

def mpl_param_sh(sims, names, labels,
                 sim_subset=None,
                 title='Spherical Harmonics Update  (C₂₀, C₄₀)',
                 figsize=None):
    """1×2 bar chart: ΔC₂₀ and ΔC₄₀ across simulations [dimensionless].

    Keeps only simulations that estimate at least one spherical-harmonics
    parameter (group='Spherical Harmonics').
    """
    if sim_subset is not None:
        pairs  = [(n, labels[names.index(n)]) for n in sim_subset if n in sims and n in names]
        names  = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]

    filt = [(sn, labels[names.index(sn)]) for sn in names
            if _param_update(sims.get(sn, {}), ['Spherical Harmonics'])[0] is not None]
    if not filt:
        print("WARNING: No simulations with SH parameters found, skipping mpl_param_sh.")
        return None
    f_names, f_labels = zip(*filt)

    if figsize is None:
        figsize = (FIG_W_DOUBLE * 1.2, FIG_H_DEFAULT)

    sh_colors = {'C₂₀': '#E65100', 'C₄₀': '#EF6C00'}
    fig, (ax_c20, ax_c40) = plt.subplots(1, 2, figsize=figsize)

    for ax, plabel in zip((ax_c20, ax_c40), ('C₂₀', 'C₄₀')):
        vals = []
        for sn in f_names:
            v_all, pl, _ = _param_update(sims.get(sn, {}), ['Spherical Harmonics'])
            vals.append(v_all[pl.index(plabel)] if (pl is not None and plabel in pl) else np.nan)
        _bar_ax(ax, vals, sh_colors[plabel], f_labels, '[-]', plabel, fontsize_annot=8)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


# ============================================================================
# MULTI-DATASET HELPERS AND FIGURES
# ============================================================================

def _load_dataset_for_config(config_name):
    """Load (sims, names, labels, sim_colors, sim_linestyle, sim_markers) for a
    named ExportConfig module.  Uses the module's SELECTED_SIMS / SIM_LABELS to
    filter and order the simulations, then falls back to all available names."""
    cfg  = importlib.import_module(f'ExportConfigs.{config_name}')
    sims, all_names = get_active_data(cfg.DATASET_LABEL)

    sel = cfg.SELECTED_SIMS if cfg.SELECTED_SIMS is not None else list(all_names)
    names = [n for n in sel if n in sims]

    if cfg.SIM_LABELS is not None and len(cfg.SIM_LABELS) == len(sel):
        idx_map = {n: i for i, n in enumerate(sel)}
        labels  = [cfg.SIM_LABELS[idx_map[n]] for n in names]
    else:
        labels = list(names)

    colors    = getattr(cfg, 'SIM_COLORS',    {})
    linestyle = getattr(cfg, 'SIM_LINESTYLE', {})
    markers   = getattr(cfg, 'SIM_MARKERS',   {})
    return sims, names, labels, colors, linestyle, markers


_MULTI_PANEL_H   = 3.5    # figure height for each individual split figure
_MULTI_PANEL_W   = FIG_W_SINGLE * 1.4   # ~4.9" — fits in a half-page minipage at print size
_MULTI_FS_TICKS  = 13     # x-tick label fontsize
_MULTI_FS_ANNOT  = 12     # value annotation fontsize
_MULTI_FS_YLABEL = 14     # y-axis label fontsize
_MULTI_FS_TITLE  = 15     # panel title fontsize


def _multi_dot_panel(ax, sims, names, labels, vals,
                     colors, markers, col, panel_title):
    """Draw a single dot-plot panel (used by all *_multi figures)."""
    x = np.arange(len(names))
    valid_x = [xi for xi, v in enumerate(vals) if not np.isnan(v)]
    valid_v = [v  for v  in vals               if not np.isnan(v)]
    if len(valid_x) > 1:
        ax.plot(valid_x, valid_v, '-', color='gray', linewidth=1.0, alpha=0.5, zorder=1)
    for xi, v in enumerate(vals):
        if not np.isnan(v):
            sn  = names[xi]
            c   = colors.get(sn, col)
            mk  = markers.get(sn, 'o')
            ax.plot(xi, v, marker=mk, color=c, markersize=7,
                    markeredgecolor='black', markeredgewidth=0.4,
                    linestyle='none', zorder=2)
            ax.text(xi, v, f'  {v:.2f}', ha='left', va='center',
                    fontsize=_MULTI_FS_ANNOT)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=_MULTI_FS_TICKS)
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_title(panel_title, fontsize=_MULTI_FS_TITLE)


def mpl_rms_compare_multi(sims, names, labels,
                           datasets=None,
                           dataset_titles=None,
                           title='RMS vs NEP097 — Dataset Comparison'):
    """N×1 figure: one rms_compare panel per dataset, stacked vertically."""
    if not datasets:
        return mpl_rms_compare(sims, names, labels, title=title)

    n_ds = len(datasets)
    if dataset_titles is None:
        dataset_titles = datasets

    fig, axes = plt.subplots(n_ds, 1, figsize=(_MULTI_PANEL_W, _MULTI_PANEL_H * n_ds))
    if n_ds == 1:
        axes = [axes]

    for ax, cfg_name, ds_title in zip(axes, datasets, dataset_titles):
        ds, ns, ls, cols, lss, mks = _load_dataset_for_config(cfg_name)
        vals = [ds[n].get('rms_SPICE', np.nan) if n in ds else np.nan for n in ns]
        _multi_dot_panel(ax, ds, ns, ls, vals, cols, mks, '#1f77b4', ds_title)
        ax.set_ylabel('RMS [km]', fontsize=_MULTI_FS_YLABEL)

    fig.tight_layout()
    return fig


def mpl_gof_combined_multi(sims, names, labels,
                            datasets=None,
                            dataset_titles=None,
                            show_initial=False,
                            title='Goodness of Fit — Dataset Comparison'):
    """N×3 figure: rows = one dataset each, columns = WRMS / RMS / Cost.
    Each subplot has its own y-axis (no sharey) to avoid cross-dataset scale collapse."""
    if not datasets:
        return mpl_gof_combined(sims, names, labels,
                                show_initial=show_initial, title=title)

    n_ds = len(datasets)
    if dataset_titles is None:
        dataset_titles = datasets

    metrics = [
        ('final_wrms_combined_method1_mas', 'initial_wrms_combined_method1_mas', 'WRMS [mas]'),
        ('final_rms_combined_mas',          'initial_rms_combined_mas',          'RMS [mas]'),
        ('final_cost_function',             'initial_cost_function',             'Cost Function'),
    ]

    fig, axes = plt.subplots(n_ds, len(metrics),
                             figsize=(FIG_W_DOUBLE * 1.4, FIG_H_DEFAULT * 0.8 * n_ds))
    if n_ds == 1:
        axes = axes.reshape(1, -1)

    for ri, (cfg_name, ds_title) in enumerate(zip(datasets, dataset_titles)):
        ds, ns, ls, cols, lss, mks = _load_dataset_for_config(cfg_name)
        wm = compute_wrms_and_cost(ds, ns)
        avs = [n for n in ns if n in wm]
        avl = [ls[ns.index(n)] for n in avs]
        x   = np.arange(len(avs))

        for ci, (fk, ik, ylabel) in enumerate(metrics):
            ax = axes[ri, ci]
            fv = [wm[n].get(fk) for n in avs]
            iv = [wm[n].get(ik) for n in avs]

            valid_f = [(xi, v) for xi, v in enumerate(fv) if v is not None]
            if len(valid_f) > 1:
                ax.plot([p[0] for p in valid_f], [p[1] for p in valid_f],
                        '-', color='steelblue', linewidth=1.0, alpha=0.4, zorder=1)
            if show_initial:
                valid_i = [(xi, v) for xi, v in enumerate(iv) if v is not None]
                if len(valid_i) > 1:
                    ax.plot([p[0] for p in valid_i], [p[1] for p in valid_i],
                            '--', color='lightcoral', linewidth=1.0, alpha=0.4, zorder=1)

            for xi, sn in enumerate(avs):
                c  = cols.get(sn, _color(xi))
                mk = mks.get(sn, 'o')
                if fv[xi] is not None:
                    ax.plot(xi, fv[xi], marker=mk, color=c, markersize=7,
                            markeredgecolor='black', markeredgewidth=0.5,
                            linestyle='none', zorder=2)
                if show_initial and iv[xi] is not None:
                    ax.plot(xi, iv[xi], marker=mk, color=c, markersize=5,
                            markeredgecolor='black', markeredgewidth=0.5,
                            linestyle='none', zorder=2, alpha=0.45,
                            markerfacecolor='none')

            ax.set_xticks(x)
            ax.set_xticklabels(avl, rotation=35, ha='right', fontsize=8)
            ax.set_xlim(-0.5, len(avs) - 0.5)
            if ci == 0:
                ax.set_ylabel(ds_title, fontsize=9)
            if ri == 0:
                ax.set_title(ylabel, fontsize=10)

    if show_initial:
        from matplotlib.lines import Line2D
        axes[0, 0].legend(handles=[
            Line2D([0], [0], linestyle='-',  color='steelblue',  linewidth=1.2, label='Final'),
            Line2D([0], [0], linestyle='--', color='lightcoral', linewidth=1.2, label='Initial'),
        ], fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def mpl_gof_metric_multi(sims, names, labels,
                          datasets=None,
                          dataset_titles=None,
                          metric='wrms',
                          show_initial=False,
                          title=None):
    """N×1 figure: one panel per dataset, showing a single GoF metric.

    Parameters
    ----------
    metric : 'wrms' | 'rms' | 'cost'
        Which goodness-of-fit metric to plot.
    """
    _metric_map = {
        'wrms': ('final_wrms_combined_method1_mas', 'initial_wrms_combined_method1_mas', 'WRMS [mas]'),
        'rms':  ('final_rms_combined_mas',          'initial_rms_combined_mas',          'RMS [mas]'),
        'cost': ('final_cost_function',             'initial_cost_function',             'Cost Function'),
    }
    fk, ik, ylabel = _metric_map.get(metric, _metric_map['wrms'])

    if title is None:
        title = f'{ylabel} — Dataset Comparison'

    if not datasets:
        return mpl_gof(sims, names, labels, metric=metric,
                       show_initial=show_initial, title=title)

    if dataset_titles is None:
        dataset_titles = datasets

    n_ds = len(datasets)
    fig, axes = plt.subplots(n_ds, 1, figsize=(_MULTI_PANEL_W, _MULTI_PANEL_H * n_ds))
    if n_ds == 1:
        axes = [axes]

    for ax, cfg_name, ds_title in zip(axes, datasets, dataset_titles):
        ds, ns, ls, cols, lss, mks = _load_dataset_for_config(cfg_name)
        wm = compute_wrms_and_cost(ds, ns)
        avs = [n for n in ns if n in wm]
        avl = [ls[ns.index(n)] for n in avs]
        x   = np.arange(len(avs))

        fv = [wm[n].get(fk) for n in avs]
        iv = [wm[n].get(ik) for n in avs]

        valid_f = [(xi, v) for xi, v in enumerate(fv) if v is not None]
        if len(valid_f) > 1:
            ax.plot([p[0] for p in valid_f], [p[1] for p in valid_f],
                    '-', color='steelblue', linewidth=1.0, alpha=0.4, zorder=1)
        if show_initial:
            valid_i = [(xi, v) for xi, v in enumerate(iv) if v is not None]
            if len(valid_i) > 1:
                ax.plot([p[0] for p in valid_i], [p[1] for p in valid_i],
                        '--', color='lightcoral', linewidth=1.0, alpha=0.4, zorder=1)

        for xi, sn in enumerate(avs):
            c  = cols.get(sn, _color(xi))
            mk = mks.get(sn, 'o')
            if fv[xi] is not None:
                ax.plot(xi, fv[xi], marker=mk, color=c, markersize=7,
                        markeredgecolor='black', markeredgewidth=0.5,
                        linestyle='none', zorder=2)
            if show_initial and iv[xi] is not None:
                ax.plot(xi, iv[xi], marker=mk, color=c, markersize=5,
                        markeredgecolor='black', markeredgewidth=0.5,
                        linestyle='none', zorder=2, alpha=0.45,
                        markerfacecolor='none')

        ax.set_xticks(x)
        ax.set_xticklabels(avl, rotation=35, ha='right', fontsize=_MULTI_FS_TICKS)
        ax.set_xlim(-0.5, len(avs) - 0.5)
        ax.set_ylabel(ylabel, fontsize=_MULTI_FS_YLABEL)
        ax.set_title(ds_title, fontsize=_MULTI_FS_TITLE)

    if show_initial:
        from matplotlib.lines import Line2D
        axes[0].legend(handles=[
            Line2D([0], [0], linestyle='-',  color='steelblue',  linewidth=1.2, label='Final'),
            Line2D([0], [0], linestyle='--', color='lightcoral', linewidth=1.2, label='Initial'),
        ], fontsize=_MULTI_FS_TICKS)

    fig.tight_layout()
    return fig


def mpl_formal_rms_multi(sims, names, labels,
                          datasets=None,
                          dataset_titles=None,
                          sharey=False,
                          title='Formal Error RMS — Dataset Comparison'):
    """1×N figure: total formal error RMS per simulation per dataset.

    Total formal error RMS = sqrt(mean(formal_errors_RSW_km ** 2)) over all
    R/S/W components — the scalar counterpart to rms_SPICE.
    """
    if not datasets:
        # Fallback: single-panel using current dataset
        vals = []
        for sn in names:
            sd = sims.get(sn, {})
            if 'formal_errors_RSW_km' in sd:
                vals.append(np.sqrt(np.mean(sd['formal_errors_RSW_km'] ** 2)))
            else:
                vals.append(np.nan)
        fig, ax = plt.subplots(figsize=(max(FIG_W_DOUBLE, 0.8 * len(names)), FIG_H_DEFAULT))
        _multi_dot_panel(ax, sims, names, labels, vals, _SIM_COLORS, _SIM_MARKERS, '#2ca02c', title)
        ax.set_ylabel('Formal Error RMS [km]')
        fig.tight_layout()
        return fig

    if dataset_titles is None:
        dataset_titles = datasets

    n_ds = len(datasets)
    fig, axes = plt.subplots(n_ds, 1, figsize=(_MULTI_PANEL_W, _MULTI_PANEL_H * n_ds))
    if n_ds == 1:
        axes = [axes]

    for ax, cfg_name, ds_title in zip(axes, datasets, dataset_titles):
        ds, ns, ls, cols, lss, mks = _load_dataset_for_config(cfg_name)
        vals = []
        for sn in ns:
            sd = ds.get(sn, {})
            if 'formal_errors_RSW_km' in sd:
                vals.append(np.sqrt(np.mean(sd['formal_errors_RSW_km'] ** 2)))
            else:
                vals.append(np.nan)
        _multi_dot_panel(ax, ds, ns, ls, vals, cols, mks, '#2ca02c', ds_title)
        ax.set_ylabel('Formal Error RMS [km]', fontsize=_MULTI_FS_YLABEL)

    fig.tight_layout()
    return fig


def mpl_rms_ratio_multi(sims, names, labels,
                         datasets=None,
                         dataset_titles=None,
                         title='RMS / Formal Error RMS — Dataset Comparison'):
    """N×1 figure: rms_SPICE / total formal error RMS per simulation per dataset."""
    if not datasets:
        return mpl_rms_ratio(sims, names, labels, title=title)

    if dataset_titles is None:
        dataset_titles = datasets

    n_ds = len(datasets)
    fig, axes = plt.subplots(n_ds, 1, figsize=(_MULTI_PANEL_W, _MULTI_PANEL_H * n_ds))
    if n_ds == 1:
        axes = [axes]

    for ax, cfg_name, ds_title in zip(axes, datasets, dataset_titles):
        ds, ns, ls, cols, lss, mks = _load_dataset_for_config(cfg_name)
        vals = []
        for sn in ns:
            sd = ds.get(sn, {})
            rms_spice = sd.get('rms_SPICE', np.nan)
            if 'formal_errors_RSW_km' in sd and not np.isnan(rms_spice):
                formal_rms = np.sqrt(np.mean(sd['formal_errors_RSW_km'] ** 2))
                vals.append(rms_spice / formal_rms if formal_rms > 0 else np.nan)
            else:
                vals.append(np.nan)
        _multi_dot_panel(ax, ds, ns, ls, vals, cols, mks, '#9467bd', ds_title)
        ax.axhline(1.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_ylabel('RMS$_{\\mathrm{SPICE}}$ / RMS$_{\\sigma}$  [—]',
                      fontsize=_MULTI_FS_YLABEL)

    fig.tight_layout()
    return fig


# ============================================================================
# OBSERVATIONAL DATASET FIGURE FUNCTIONS
# ============================================================================

# Categorical color palette (12 qualitative colors, colorblind-aware).
_OBS_PALETTE = [
    '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78',
]


def _obs_color(i: int) -> str:
    return _OBS_PALETTE[i % len(_OBS_PALETTE)]


def mpl_obs_spice_timeseries(sims, names, labels,
                              obs_folder: str = 'Observations/AllModernJ2000',
                              title: str = r'O$-$C Residuals vs NEP097',
                              figsize=None):
    """RA / Dec residual time series vs NEP097 from all observation CSV files.

    One figure with two stacked panels (RA top, Dec bottom), each observation
    ID shown in a distinct colour.
    """
    df = _load_spice_residuals_df(obs_folder)
    if df.empty:
        print('  SKIP obs_spice_timeseries: no CSV data found.')
        return None

    if figsize is None:
        figsize = (FIG_W_DOUBLE, 6.0)
    fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ref_ids = sorted(df['ref_point_id'].unique())
    for i, rid in enumerate(ref_ids):
        mask = df['ref_point_id'] == rid
        sub  = df.loc[mask].copy()
        times_dt = convert_time_array_to_datetime(
            sub['time_j2000'].values.reshape(-1, 1))
        col = _obs_color(i)
        ax_ra.scatter(times_dt,  sub['ra_resid_arcsec'].values,
                      s=8, color=col, label=rid.replace('_', ' '), alpha=0.7)
        ax_dec.scatter(times_dt, sub['dec_resid_arcsec'].values,
                       s=8, color=col, alpha=0.7)

    ax_ra.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax_dec.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax_ra.set_ylabel('RA residual [$\'\'$]')
    ax_dec.set_ylabel('Dec residual [$\'\'$]')
    ax_dec.set_xlabel('Year')
    _apply_date_formatter(ax_dec)
    ax_ra.legend(fontsize=7, ncol=3, loc='upper right', markerscale=2)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def mpl_obs_initial_timeseries(sims, names, labels,
                                sim_name: str = None,
                                title: str = 'Residuals vs Initial Propagation',
                                figsize=None):
    """RA / Dec residual time series for the initial (pre-estimation) iteration.

    Uses *sim_name*'s ``residual_df``.  Each observation ID is coloured
    distinctly.  Layout identical to mpl_obs_spice_timeseries.
    """
    if sim_name is None and names:
        sim_name = names[0]
    if sim_name not in sims or 'residual_df' not in sims.get(sim_name, {}):
        print(f'  SKIP obs_initial_timeseries: no residual_df for {sim_name!r}.')
        return None

    df = sims[sim_name]['residual_df']
    if figsize is None:
        figsize = (FIG_W_DOUBLE, 6.0)
    fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ref_ids = sorted(df['ref_point_id'].unique())
    for i, rid in enumerate(ref_ids):
        mask = df['ref_point_id'] == rid
        t    = df.loc[mask, 'datetime']
        ra   = df.loc[mask, 'ra_residual_initial_mas'] / 1000.0   # mas → arcsec
        dec  = df.loc[mask, 'dec_residual_initial_mas'] / 1000.0
        col  = _obs_color(i)
        ax_ra.scatter(t,  ra.values,  s=8, color=col,
                      label=rid.replace('_', ' '), alpha=0.7)
        ax_dec.scatter(t, dec.values, s=8, color=col, alpha=0.7)

    ax_ra.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax_dec.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax_ra.set_ylabel('RA residual [$\'\'$]')
    ax_dec.set_ylabel('Dec residual [$\'\'$]')
    ax_dec.set_xlabel('Year')
    _apply_date_formatter(ax_dec)
    ax_ra.legend(fontsize=7, ncol=3, loc='upper right', markerscale=2)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def mpl_obs_histogram_per_id(sims, names, labels,
                              sim_name: str = None,
                              obs_folder: str = 'Observations/AllModernJ2000',
                              source: str = 'both',
                              bins: int = 30,
                              fit_gauss: bool = True,
                              title: str = 'Residual Histograms',
                              figsize=None):
    """Per-ID RA / Dec residual histograms with optional Gaussian fit.

    Parameters
    ----------
    source : {'spice', 'initial', 'both'}
        Which residuals to show.  'spice' reads O-C from CSV files (vs
        NEP097); 'initial' reads the first-iteration residuals from
        *sim_name*'s residual_df; 'both' overlays both on the same axes.

    Returns a list of (fig, ref_point_id) tuples — one figure per ID.
    """
    if sim_name is None and names:
        sim_name = names[0]

    # Load SPICE residuals if needed.
    spice_df = None
    if source in ('spice', 'both'):
        spice_df = _load_spice_residuals_df(obs_folder)
        if spice_df.empty:
            print('  WARNING obs_histogram_per_id: no SPICE residual CSV data.')
            spice_df = None

    # Load initial propagation residuals if needed.
    init_df = None
    if source in ('initial', 'both'):
        if sim_name in sims and 'residual_df' in sims.get(sim_name, {}):
            init_df = sims[sim_name]['residual_df']
        else:
            print(f'  WARNING obs_histogram_per_id: no residual_df for {sim_name!r}.')

    if spice_df is None and init_df is None:
        return None

    # Union of all IDs present in either source.
    ids_spice = set(spice_df['ref_point_id'].unique()) if spice_df is not None else set()
    ids_init  = set(init_df['ref_point_id'].unique())  if init_df  is not None else set()
    all_ids   = sorted(ids_spice | ids_init)

    if figsize is None:
        figsize = (FIG_W_DOUBLE, 4.0)

    figs = []
    for rid in all_ids:
        fig, (ax_ra, ax_dec) = plt.subplots(1, 2, figsize=figsize)

        def _plot_hist(ax, data, label, color):
            data = data[np.isfinite(data)]
            if len(data) == 0:
                return
            ax.hist(data, bins=bins, alpha=0.5, color=color,
                    label=label, density=True)
            if fit_gauss and len(data) >= 5:
                mu, sigma = sp_stats.norm.fit(data)
                xr = np.linspace(data.min(), data.max(), 200)
                ax.plot(xr, sp_stats.norm.pdf(xr, mu, sigma),
                        color=color, linewidth=1.5, linestyle='--')

        if spice_df is not None and rid in ids_spice:
            sub = spice_df.loc[spice_df['ref_point_id'] == rid]
            _plot_hist(ax_ra,  sub['ra_resid_arcsec'].values,  'NEP097', '#1f77b4')
            _plot_hist(ax_dec, sub['dec_resid_arcsec'].values, 'NEP097', '#1f77b4')

        if init_df is not None and rid in ids_init:
            sub = init_df.loc[init_df['ref_point_id'] == rid]
            ra_as  = sub['ra_residual_initial_mas'].values  / 1000.0
            dec_as = sub['dec_residual_initial_mas'].values / 1000.0
            _plot_hist(ax_ra,  ra_as,  'Init. prop.', '#d62728')
            _plot_hist(ax_dec, dec_as, 'Init. prop.', '#d62728')

        rid_display = rid.replace('_', ' ')
        ax_ra.set_xlabel('RA residual [$\'\'$]')
        ax_dec.set_xlabel('Dec residual [$\'\'$]')
        ax_ra.set_ylabel('Density')
        ax_ra.set_title('RA')
        ax_dec.set_title('Dec')
        ax_ra.legend(fontsize=8)
        fig.suptitle(f'{title} — {rid_display}')
        fig.tight_layout()
        figs.append((fig, rid))

    return figs if figs else None


# ============================================================================
# DISPATCH TABLE
# ============================================================================

_PLOT_REGISTRY = {
    'rms_compare':         mpl_rms_compare,
    'rms_formal':          mpl_rms_formal,
    'rms_ratio':           mpl_rms_ratio,
    'rsw_ratio':           mpl_rsw_ratio,
    'rms_compare_multi':   mpl_rms_compare_multi,
    'gof_combined_multi':  mpl_gof_combined_multi,
    'gof_metric_multi':    mpl_gof_metric_multi,
    'formal_rms_multi':    mpl_formal_rms_multi,
    'rms_ratio_multi':     mpl_rms_ratio_multi,
    'rsw_compare':         mpl_rsw_compare,
    'formal_compare':      mpl_formal_compare,
    'rsw_stats':           mpl_rsw_stats,
    'gof':                 mpl_gof,
    'gof_combined':        mpl_gof_combined,
    'corr_heatmap':        mpl_corr_heatmap,
    'residual_histogram':  mpl_residual_histogram,
    'residual_timeseries': mpl_residual_timeseries,
    'param_state':         mpl_param_state,
    'param_rsw':           mpl_param_rsw,
    'param_pole_pos':      mpl_param_pole_pos,
    'param_pole_lib':      mpl_param_pole_lib,
    'rsw_initial':         mpl_rsw_initial,
    'rsw_initial_diff':    mpl_rsw_initial_diff,
    'rsw_with_zoom':       mpl_rsw_with_zoom,
    'formal_with_zoom':    mpl_formal_with_zoom,
    'param_gm':            mpl_param_gm,
    'param_pole_rate':     mpl_param_pole_rate,
    'param_sh':            mpl_param_sh,
    'legend':              mpl_legend,
    # ── Observational dataset ─────────────────────────────────────────────────
    'obs_spice_timeseries':   mpl_obs_spice_timeseries,
    'obs_initial_timeseries': mpl_obs_initial_timeseries,
    'obs_histogram_per_id':   mpl_obs_histogram_per_id,
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

    Columns: Parameter | Group | Unit | IAU initial | FitPole initial
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

    # FitPole reference (first-seen across all non-IAUPole sims).
    sp_ref_in = {}
    for sn in names:
        if sn.startswith('IAUPole'):
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
        rf'IAU\textsubscript{{0}} and FitPole\textsubscript{{0}} are the respective '
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
        r' & IAU$_0$ & FitPole$_0$ & Final'
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


def generate_final_estimation_tables(sims, names, out_path,
                                     iau_sim='IAUPole_pole_lib_cov',
                                     fitpole_sim='SimPole_pole_lib_cov'):
    """Write two LaTeX tables comparing state+lib. IAU vs state+lib. Fit. results.

    Table 1 — IAUPole baseline:
        Columns: Parameter | Group | Unit | p0(IAU) | state+lib.(IAU) | state+lib.(Fit.) | Δ IAU | Δ Fit.
        Δ = final − iau_initial.  For unestimated params (α₀, δ₀) the model's
        own initial value is used in the estimation column, so Δ IAU = 0.

    Table 2 — FitPole baseline: same but Δ = final − fit_initial.

    Requires preamble: \\usepackage{booktabs,siunitx,graphicx}
    """

    # ── helpers ──────────────────────────────────────────────────────────────
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

    def _extract_all(sn):
        """Return (initial_dict, final_dict) — both keyed by param label."""
        sd = sims.get(sn, {})
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            return {}, {}
        ph   = sd['parameter_history']
        lbls, _, _ = get_parameter_info(sd['est_parameters'])
        return ({lbl: ph[i, 0]  for i, lbl in enumerate(lbls)},
                {lbl: ph[i, -1] for i, lbl in enumerate(lbls)})

    # Full initial values: use the most complete reference sim for each model.
    iau_ref_full, _ = _get_iau_reference(sims, names)
    # FitPole full initial: aggregate across all non-IAUPole sims (first-seen).
    fit_ref_full = {}
    for sn in names:
        if sn.startswith('IAUPole'):
            continue
        sd = sims.get(sn, {})
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            continue
        ph    = sd['parameter_history']
        lbls_, _, _ = get_parameter_info(sd['est_parameters'])
        for i, lbl in enumerate(lbls_):
            if lbl not in fit_ref_full:
                fit_ref_full[lbl] = ph[i, 0]

    # Final values for the two target sims (only params they estimated).
    iau_init, iau_final = _extract_all(iau_sim)
    fit_init, fit_final = _extract_all(fitpole_sim)

    if not iau_ref_full and not fit_ref_full:
        print("  SKIP generate_final_estimation_tables: no reference data.")
        return

    # ── canonical param list ──────────────────────────────────────────────────
    seen, param_rows = set(), []
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

    # For each param: resolved display values.
    # iau_col[lbl] = iau_final if estimated in iau_sim, else full IAU initial.
    # fit_col[lbl] = fit_final if estimated in fitpole_sim, else full Fit initial.
    def _col(lbl, final_d, full_ref):
        return final_d[lbl] if lbl in final_d else full_ref.get(lbl)

    lines = []
    lines.append(r'% ============================================================')
    lines.append(r'% Final estimation comparison tables')
    lines.append(r'% state+lib.(IAU): ' + iau_sim)
    lines.append(r'% state+lib.(Fit.): ' + fitpole_sim)
    lines.append(r'% Preamble: \usepackage{booktabs,siunitx,graphicx}')
    lines.append(r'% ============================================================')
    lines.append('')

    for tbl_idx, (baseline_label, baseline_ref) in enumerate([
        (r'$p_{\mathrm{IAU},0}$', iau_ref_full),
        (r'$p_{\mathrm{Fit},0}$', fit_ref_full),
    ], start=1):
        bname  = 'IAU' if tbl_idx == 1 else 'Fit'
        tlabel = f'tab:final_est_{bname.lower()}_baseline'

        if tbl_idx == 1:
            caption = (
                r'Estimated parameter values for \texttt{state+lib.}\ '
                r'simulations referenced to the IAU~2015 a~priori $p_{\mathrm{IAU},0}$. '
                r'$\Delta = p_{\mathrm{final}} - p_{\mathrm{IAU},0}$. '
                r'For unestimated parameters the model initial value is shown.'
            )
        else:
            caption = (
                r'Estimated parameter values for \texttt{state+lib.}\ '
                r'simulations referenced to the NEP097-fitted initial values '
                r'$p_{\mathrm{Fit},0}$. '
                r'$\Delta = p_{\mathrm{final}} - p_{\mathrm{Fit},0}$. '
                r'For unestimated parameters the model initial value is shown.'
            )

        lines.append(rf'% ---- Table {tbl_idx}: {bname} baseline ----')
        lines.append(r'\begin{table}[htbp]')
        lines.append(r'  \centering')
        lines.append(rf'  \caption{{{caption}}}')
        lines.append(rf'  \label{{{tlabel}}}')
        lines.append(r'  \resizebox{\textwidth}{!}{%')
        lines.append(r'  \begin{tabular}{llcrrrrr}')
        lines.append(r'    \toprule')
        lines.append(
            r'    Parameter & Group & Unit'
            rf' & {baseline_label}'
            r' & state+lib.\ (IAU) & state+lib.\ (Fit.)'
            r' & $\Delta$\,IAU & $\Delta$\,Fit. \\'
        )
        lines.append(r'    \midrule')

        prev_grp = None
        for lbl, grp, sc, unit in param_rows:
            if prev_grp is not None and grp != prev_grp:
                lines.append(r'    \midrule')
            prev_grp = grp

            lbl_tex  = _p2tex(lbl)
            base_raw = baseline_ref.get(lbl)

            if base_raw is None:
                lines.append(f'    {lbl_tex} & {_esc(grp)} & {unit}'
                             r' & --- & --- & --- & --- & --- \\')
                continue

            iau_v = _col(lbl, iau_final, iau_ref_full)
            fit_v = _col(lbl, fit_final, fit_ref_full)

            def _cell(v):
                return _num(v * sc) if v is not None else '---'

            def _delta(v, ref):
                if v is not None and ref is not None:
                    return _num((v - ref) * sc)
                return '---'

            lines.append(
                f'    {lbl_tex} & {_esc(grp)} & {unit}'
                f' & {_num(base_raw * sc)}'
                f' & {_cell(iau_v)} & {_cell(fit_v)}'
                rf' & {_delta(iau_v, base_raw)} & {_delta(fit_v, base_raw)} \\'
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

    Table 1 — Initial values: IAU initial, FitPole initial, Δ [unit], Δ [%].
               Covers ALL parameter groups (state + pole + librations).
    Table 2 — Split IAU / FitPole sub-tables.  Rows = parameters,
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
        s = lb.replace('IAUPole_', '').replace('SimPole_', '').replace('FitPole_', '')
        return _esc(s)

    def _diff(raw_diff, sc):
        """Physical difference formatted with siunitx."""
        return rf'\num{{{raw_diff * sc:.5e}}}'

    # ── build canonical param list from ALL sims (fixes missing pole params) ──
    # iau_ref_in already aggregates from all IAUPole sims.
    iau_ref_in, _ = _get_iau_reference(sims, names)

    # FitPole reference: initial value for each param, first-seen across all non-IAUPole sims.
    sp_ref_in = {}
    for sn in names:
        if sn.startswith('IAUPole'):
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
    sp_names  = [n for n in names if not n.startswith('IAUPole')]
    iau_lbls  = [labels[names.index(n)] for n in iau_names]
    sp_lbls   = [labels[names.index(n)] for n in sp_names]
    # Detect the FitPole prefix from actual sim names for captions.
    _fit_prefix = next(
        (n.split('_')[0] for n in sp_names if '_' in n),
        'FitPole'
    )

    lines = []
    lines.append(r'% ============================================================')
    lines.append(r'% Parameter tables — Neptune/Triton orbit estimation thesis')
    lines.append(r'% Preamble: \usepackage{booktabs,siunitx,graphicx}')
    lines.append(r'% ============================================================')
    lines.append('')

    # ── TABLE 1: IAU initial vs FitPole initial ───────────────────────────────
    lines.append(r'% ---- Table 1: initial parameter values (IAU vs FitPole) ----')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(
        r'  \caption{Initial parameter values for the IAU and FitPole rotation models. '
        r'$\Delta = p_{\mathrm{FitPole,0}} - p_{\mathrm{IAU,0}}$ in the listed unit; '
        r'$\Delta[\%] = \Delta\,/\,|p_{\mathrm{IAU,0}}| \times 100$.}'
    )
    lines.append(r'  \label{tab:initial_params}')
    lines.append(r'  \resizebox{\textwidth}{!}{%')
    lines.append(r'  \begin{tabular}{llcrrrrr}')
    lines.append(r'    \toprule')
    lines.append(
        r'    Parameter & Group & Unit'
        r' & IAU initial & FitPole initial'
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
            (sp_names, sp_lbls, _fit_prefix,
             rf'{_fit_prefix}-model simulations (all referenced to IAU initial)',
             f'tab:update_{_fit_prefix.lower()}'),
        ]

        for grp_names, grp_labels, grp_tag, caption_detail, tbl_label in subtables:
            if not grp_names:
                continue

            col_spec = 'lll' + 'r' * len(grp_names)

            lines.append(f'% ---- Table 2 ({grp_tag}): final - IAU\_initial [physical units] ----')
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
# NAMING CONVENTION TABLE
# ============================================================================

def generate_naming_table(selected_sims, sim_labels, sim_descriptions,
                           dataset_label, out_path):
    """Write a LaTeX table mapping figure labels to weight-scheme descriptions.

    Parameters
    ----------
    selected_sims    : list of str   — internal simulation keys (from config)
    sim_labels       : list of str   — short display labels used in figures
    sim_descriptions : dict          — sim_name → human-readable description
    dataset_label    : str           — used in the table caption / label
    out_path         : str           — destination .tex file path
    """
    if not sim_descriptions:
        print("  SKIP naming table: SIM_DESCRIPTIONS not defined in config.")
        return

    def _esc(s):
        """Escape special LaTeX characters in plain-text strings."""
        for ch, rep in [('_', r'\_'), ('&', r'\&'), ('%', r'\%'), ('#', r'\#')]:
            s = s.replace(ch, rep)
        return s

    safe_label = dataset_label.replace('_', '-').lower()
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(
        rf'  \caption{{Weight-scheme naming conventions used in figures for '
        rf'\texttt{{{_esc(dataset_label)}}}.}}'
    )
    lines.append(rf'  \label{{tab:weight-naming-{safe_label}}}')
    lines.append(r'  \begin{tabular}{l p{10cm}}')
    lines.append(r'    \toprule')
    lines.append(r'    \textbf{Label} & \textbf{Description} \\')
    lines.append(r'    \midrule')

    label_map = {}
    if sim_labels is not None and len(sim_labels) == len(selected_sims):
        label_map = dict(zip(selected_sims, sim_labels))

    for sn in selected_sims:
        lbl  = label_map.get(sn, sn)
        desc = sim_descriptions.get(sn, '—')
        lines.append(
            rf'    \texttt{{{_esc(lbl)}}} & {desc} \\'
        )

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


def generate_weight_scheme_table(table_rows, dataset_label, out_path):
    """Write a LaTeX overview table of weight-scheme naming conventions.

    Parameters
    ----------
    table_rows    : list of (label, weighting_scheme, comments)
    dataset_label : str — used in caption and \\label
    out_path      : str — destination .tex file path
    """
    if not table_rows:
        print("  SKIP weight-scheme table: SIM_TABLE_ROWS not defined in config.")
        return

    safe_label = dataset_label.replace('_', '-').lower()
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(
        rf'  \caption{{Overview of weight schemes used in the \texttt{{{dataset_label}}} analysis.}}'
    )
    lines.append(rf'  \label{{tab:weight-schemes-{safe_label}}}')
    lines.append(r'  \begin{tabular}{l l l}')
    lines.append(r'    \toprule')
    lines.append(r'    \textbf{Label} & \textbf{Weighting Scheme} & \textbf{Comments} \\')
    lines.append(r'    \midrule')

    for lbl, scheme, comments in table_rows:
        lines.append(rf'    \texttt{{{lbl}}} & {scheme} & {comments} \\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


# ============================================================================
# POLE COMPARISON TABLE
# ============================================================================

def generate_pole_comparison_table(sims, names, labels, pole_sims, out_path):
    """Write a LaTeX table comparing pole (and position) parameter estimates
    across a set of estimation variants.

    Layout
    ------
    Rows  : α₀, δ₀, α₁, δ₁ (when available), X, Y, Z
    Cols  : Parameter | Unit | Initial | {sim1}: Est. | Δ | % | {sim2}: … | …

    The initial column is taken from the first available sim's parameter_history[:,0].
    If a sim does not estimate a given parameter, the cell shows '---'.

    Parameters
    ----------
    pole_sims : list of str
        Subset of sim names to use as estimation-variant columns (e.g.
        ``['pole_pos_IAU', 'pole_lib_IAU', 'pole_pos_lib_IAU']``).
        Only sims present in *both* ``sims`` and ``names`` are used.
    """
    # ── local helpers ─────────────────────────────────────────────────────────
    def _esc(s):
        for ch, rep in [('_', r'\_'), ('&', r'\&'), ('%', r'\%'), ('#', r'\#')]:
            s = s.replace(ch, rep)
        return s

    _P2TEX = {
        'α₀':  r'$\alpha_0$',
        'δ₀':  r'$\delta_0$',
        'α₁':  r'$\alpha_1$',
        'δ₁':  r'$\delta_1$',
        'α̇₀':  r'$\dot{\alpha}_0$',
        'δ̇₀':  r'$\dot{\delta}_0$',
        'X':   r'$X$',
        'Y':   r'$Y$',
        'Z':   r'$Z$',
    }
    def _p2tex(lbl):
        return _P2TEX.get(lbl, _esc(lbl))

    def _num(v):
        return rf'\num{{{v:.5e}}}'

    # ── resolve sim list and labels ───────────────────────────────────────────
    valid_pole_sims = [sn for sn in pole_sims if sn in sims and sn in names]
    if not valid_pole_sims:
        print("  SKIP pole comparison table: none of POLE_TABLE_SIMS found in dataset.")
        return

    col_labels = [labels[names.index(sn)] for sn in valid_pole_sims]

    # ── collect initial parameter values (first sim that has each param) ──────
    # Groups and scaling for the rows we care about.
    _WANTED_GROUPS = {'Pole Position', 'Pole Librations', 'Position'}
    _SCALE = {
        'Pole Position':   (_RAD_TO_DEG, 'deg'),
        'Pole Librations': (_RAD_TO_DEG, 'deg'),
        'Position':        (1e-3,        'km'),
    }

    # Build the canonical row list from all pole_sims in order.
    seen       = set()
    param_rows = []   # (lbl, grp, scale, unit)
    for sn in valid_pole_sims:
        sd = sims.get(sn, {})
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            continue
        lbls, grps, _ = get_parameter_info(sd['est_parameters'])
        for lbl, grp in zip(lbls, grps):
            if grp not in _WANTED_GROUPS or lbl in seen:
                continue
            seen.add(lbl)
            sc, unit = _SCALE[grp]
            param_rows.append((lbl, grp, sc, unit))

    if not param_rows:
        print("  SKIP pole comparison table: no pole/position parameters found.")
        return

    # Build initial-value lookup (first-seen per label across all pole_sims).
    init_vals = {}   # lbl → raw value
    for sn in valid_pole_sims:
        sd = sims.get(sn, {})
        if 'parameter_history' not in sd or 'est_parameters' not in sd:
            continue
        ph    = sd['parameter_history']
        lbls, _, _ = get_parameter_info(sd['est_parameters'])
        for i, lbl in enumerate(lbls):
            if lbl not in init_vals:
                init_vals[lbl] = ph[i, 0]

    # Per-sim: final parameter values.
    final_vals = {}   # sn → {lbl: raw_final}
    for sn in valid_pole_sims:
        sd = sims.get(sn, {})
        fv = {}
        if 'parameter_history' in sd and 'est_parameters' in sd:
            ph    = sd['parameter_history']
            lbls, _, _ = get_parameter_info(sd['est_parameters'])
            for i, lbl in enumerate(lbls):
                fv[lbl] = ph[i, -1]
        final_vals[sn] = fv

    # ── build LaTeX ───────────────────────────────────────────────────────────
    n_est_cols = len(valid_pole_sims)
    # Each estimation sim contributes 3 columns: Est. | Δ | %
    n_total_cols = 3 + n_est_cols * 3   # Parameter + Unit + Initial + [Est|Δ|%] × n

    col_spec = 'l l r ' + ' '.join(['r r r'] * n_est_cols)

    lines = []
    lines.append(r'% ============================================================')
    lines.append(r'% Pole parameter comparison table — SimObs analysis')
    lines.append(r'% Preamble: \usepackage{booktabs,siunitx,graphicx}')
    lines.append(r'% ============================================================')
    lines.append('')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(
        r'  \caption{Estimated pole and position parameters for selected SimObs variants. '
        r'$\Delta = \text{Est.} - \text{Initial}$; '
        r'$\Delta[\%] = \Delta / |\text{Initial}| \times 100$.}'
    )
    lines.append(r'  \label{tab:pole-comparison-simobs}')
    lines.append(r'  \resizebox{\textwidth}{!}{%')
    lines.append(f'  \\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'    \toprule')

    # Header row 1: spanning column labels per sim
    header1_parts = [r'    \multicolumn{3}{l}{}']
    for col_lbl in col_labels:
        n_span = 3
        header1_parts.append(
            rf'& \multicolumn{{{n_span}}}{{c}}{{\texttt{{{_esc(col_lbl)}}}}}'
        )
    lines.append(' '.join(header1_parts) + r' \\')

    # Cmidrule underlines for each group of 3 cols
    crule_parts = []
    for k in range(n_est_cols):
        start = 4 + k * 3   # 1-based: cols 1-3 are Param, Unit, Initial
        end   = start + 2
        crule_parts.append(rf'\cmidrule(lr){{{start}-{end}}}')
    lines.append('    ' + ' '.join(crule_parts))

    # Header row 2: column names
    header2 = r'    Parameter & Unit & Initial'
    for _ in col_labels:
        header2 += r' & Est. & $\Delta$ & $\Delta\,[\%]$'
    header2 += r' \\'
    lines.append(header2)
    lines.append(r'    \midrule')

    prev_grp = None
    for lbl, grp, sc, unit in param_rows:
        if prev_grp is not None and grp != prev_grp:
            lines.append(r'    \midrule')
        prev_grp = grp

        lbl_tex  = _p2tex(lbl)
        init_raw = init_vals.get(lbl)
        init_str = _num(init_raw * sc) if init_raw is not None else '---'

        row = f'    {lbl_tex} & {unit} & {init_str}'
        for sn in valid_pole_sims:
            fin_raw = final_vals[sn].get(lbl)
            if fin_raw is None or init_raw is None:
                row += r' & --- & --- & ---'
            else:
                fin_s  = fin_raw  * sc
                init_s = init_raw * sc
                diff   = fin_s - init_s
                pct    = diff / abs(init_s) * 100 if init_s != 0 else 0.0
                row   += f' & {_num(fin_s)} & {_num(diff)} & {pct:.2f}'
        row += r' \\'
        lines.append(row)

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'  }% end resizebox')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


# ============================================================================
# OBSERVATIONAL DATASET TABLE
# ============================================================================

def generate_obs_dataset_table(sims: dict,
                                names: list,
                                obs_folder: str,
                                raw_obs_folder: str = '',
                                obs_types_override: dict = None,
                                sim_for_initial: str = None,
                                caption: str = 'Observational dataset summary.',
                                label: str = 'tab:obs-dataset-summary',
                                out_path: str = 'table_obs_dataset.tex'):
    """Write a LaTeX table summarising the observational dataset per ID.

    Columns
    -------
    Observatory (listing)    : observatory name with NSDC listing code in
                               parentheses, e.g. "U.S. Naval Observatory, Flagstaff
                               (nm0077)".  Underscores are replaced by spaces.
    Code                     : 3-digit MPC observatory code.
    N                        : number of observations.
    Type                     : 'Rel.' or 'Abs.' (from NSDC raw file or override).
    RMS_RA_NEP097            : RMS of RA  residual vs NEP097 [arcsec].
    RMS_Dec_NEP097           : RMS of Dec residual vs NEP097 [arcsec].
    RMS_RA_init              : RMS of RA  residual vs initial propagation [arcsec].
    RMS_Dec_init             : RMS of Dec residual vs initial propagation [arcsec].

    Rows are sorted by MPC observatory code (numeric), then by NSDC listing ID.

    Requires LaTeX preamble: \\usepackage{booktabs,siunitx}
    """

    def _esc(s: str) -> str:
        for ch, rep in [('_', r'\_'), ('&', r'\&'), ('%', r'\%'), ('#', r'\#'),
                        ('$', r'\$')]:
            s = s.replace(ch, rep)
        return s

    # ── Load SPICE residuals ──────────────────────────────────────────────────
    spice_df = _load_spice_residuals_df(obs_folder)

    # ── Load initial propagation residuals ───────────────────────────────────
    init_df = None
    if sim_for_initial and sim_for_initial in sims:
        init_df = sims[sim_for_initial].get('residual_df')

    # ── Build per-ID rows ─────────────────────────────────────────────────────
    all_ids = sorted(spice_df['ref_point_id'].unique()) if not spice_df.empty else []
    if init_df is not None:
        for rid in init_df['ref_point_id'].unique():
            if rid not in all_ids:
                all_ids.append(rid)

    # Sort: numeric MPC code first, then listing ID.
    def _sort_key(rid):
        parts = rid.split('_')
        try:
            code = int(parts[0])
        except ValueError:
            code = 9999
        nsdc = parts[1] if len(parts) > 1 else ''
        return (code, nsdc)

    all_ids.sort(key=_sort_key)

    table_rows = []
    for rid in all_ids:
        parts    = rid.split('_')
        mpc_code = parts[0]
        nsdc_id  = parts[1] if len(parts) > 1 else ''

        obs_name = _get_observatory_name(mpc_code)
        obs_type = _get_obs_type(nsdc_id, raw_obs_folder, obs_types_override)

        # SPICE residuals for this ID.
        rms_ra_spice = rms_dec_spice = float('nan')
        n_obs = 0
        if not spice_df.empty and rid in spice_df['ref_point_id'].values:
            sub = spice_df.loc[spice_df['ref_point_id'] == rid]
            n_obs        = len(sub)
            rms_ra_spice = float(np.sqrt(np.nanmean(sub['ra_resid_arcsec'].values  ** 2)))
            rms_dec_spice= float(np.sqrt(np.nanmean(sub['dec_resid_arcsec'].values ** 2)))

        # Initial propagation residuals for this ID.
        rms_ra_init = rms_dec_init = float('nan')
        if init_df is not None and rid in init_df['ref_point_id'].values:
            sub_i = init_df.loc[init_df['ref_point_id'] == rid]
            if n_obs == 0:
                n_obs = len(sub_i)
            ra_as  = sub_i['ra_residual_initial_mas'].values  / 1000.0   # mas→arcsec
            dec_as = sub_i['dec_residual_initial_mas'].values / 1000.0
            rms_ra_init  = float(np.sqrt(np.nanmean(ra_as  ** 2)))
            rms_dec_init = float(np.sqrt(np.nanmean(dec_as ** 2)))

        table_rows.append((obs_name, mpc_code, nsdc_id, n_obs, obs_type,
                           rms_ra_spice, rms_dec_spice,
                           rms_ra_init,  rms_dec_init))

    # ── Build LaTeX ───────────────────────────────────────────────────────────
    def _fmt(v):
        """Format a residual value [arcsec] to 3 decimal places, or '---'."""
        return f'{v:.3f}' if np.isfinite(v) else '---'

    safe_label = label.replace('_', '-')
    lines = []
    lines.append(r'% ============================================================')
    lines.append(r'% Observational dataset summary table')
    lines.append(r'% Preamble: \usepackage{booktabs,siunitx}')
    lines.append(r'% ============================================================')
    lines.append('')
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(rf'  \caption{{{_esc(caption)}}}')
    lines.append(rf'  \label{{{safe_label}}}')
    lines.append(r'  \resizebox{\textwidth}{!}{%')
    lines.append(r'  \begin{tabular}{l c c c r r r r}')
    lines.append(r'    \toprule')
    lines.append(
        r'    Observatory (listing) & Code & $N$ & Type'
        r' & \multicolumn{2}{c}{RMS vs NEP097 [$^{\prime\prime}$]}'
        r' & \multicolumn{2}{c}{RMS vs init.\ prop.\ [$^{\prime\prime}$]} \\'
    )
    lines.append(
        r'    & & & & RA & Dec & RA & Dec \\'
    )
    lines.append(r'    \midrule')

    prev_code = None
    for (obs_name, mpc_code, nsdc_id, n_obs, obs_type,
         rms_ra_sp, rms_dec_sp, rms_ra_in, rms_dec_in) in table_rows:
        if prev_code is not None and mpc_code != prev_code:
            lines.append(r'    \midrule')
        prev_code = mpc_code

        display_name  = f'{_esc(obs_name)} ({nsdc_id})'
        lines.append(
            f'    {display_name} & {mpc_code} & {n_obs} & {obs_type}'
            f' & {_fmt(rms_ra_sp)} & {_fmt(rms_dec_sp)}'
            f' & {_fmt(rms_ra_in)} & {_fmt(rms_dec_in)} \\\\'
        )

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'  }% end resizebox')
    lines.append(r'\end{table}')
    lines.append('')

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'  Saved: {out_path}')


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

        # fig_label, subfolder are routing/naming keys — pop before passing to fn.
        fig_label = kwargs.pop('fig_label', None)
        subfolder = kwargs.pop('subfolder', None)
        suffix    = (f'_{fig_label}' if fig_label
                     else (f'_{kwargs.get("variant", "")}' if kwargs.get('variant') else ''))

        # Resolve output directory (create subfolder on demand).
        if subfolder:
            fig_out_dir = os.path.join(out_dir, subfolder)
            os.makedirs(fig_out_dir, exist_ok=True)
        else:
            fig_out_dir = out_dir

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

        # Multi-dataset functions return list of (fig, label) — save each separately.
        if isinstance(fig, list):
            for sub_fig, sub_label in fig:
                safe_label = sub_label.replace(' ', '_').replace('/', '-')
                out_path = os.path.join(fig_out_dir,
                                        f'{fn_name}{suffix}_{safe_label}.pdf')
                with PdfPages(out_path) as pdf:
                    pdf.savefig(sub_fig, bbox_inches='tight')
                plt.close(sub_fig)
                print(f"  Saved: {out_path}")
            continue

        out_path = os.path.join(fig_out_dir, f'{fn_name}{suffix}.pdf')
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

    # Write naming-convention table if config provides SIM_DESCRIPTIONS.
    _sim_descriptions = getattr(_cfg, 'SIM_DESCRIPTIONS', {})
    if _sim_descriptions:
        naming_path = os.path.join(out_dir, 'table_naming_conventions.tex')
        generate_naming_table(
            selected_sims    = _cfg.SELECTED_SIMS or list(all_names),
            sim_labels       = _cfg.SIM_LABELS,
            sim_descriptions = _sim_descriptions,
            dataset_label    = DATASET_LABEL,
            out_path         = naming_path,
        )

    # Write weight-scheme overview table if config provides SIM_TABLE_ROWS.
    _table_rows = getattr(_cfg, 'SIM_TABLE_ROWS', [])
    if _table_rows:
        ws_path = os.path.join(out_dir, 'table_weight_schemes.tex')
        generate_weight_scheme_table(
            table_rows    = _table_rows,
            dataset_label = DATASET_LABEL,
            out_path      = ws_path,
        )

    # Write final estimation comparison tables if config provides FINAL_ESTIMATION_TABLES.
    _final_est = getattr(_cfg, 'FINAL_ESTIMATION_TABLES', None)
    if _final_est:
        import pathlib
        _fe_dir  = pathlib.Path(out_dir) / 'FinalEstimation'
        _fe_dir.mkdir(parents=True, exist_ok=True)
        _fe_path = _fe_dir / 'final_estimation_params.tex'
        generate_final_estimation_tables(
            sims,
            names,
            str(_fe_path),
            iau_sim=_final_est.get('iau_sim',     'IAUPole_pole_lib_cov'),
            fitpole_sim=_final_est.get('fitpole_sim', 'SimPole_pole_lib_cov'),
        )

    # Write pole comparison table if config provides POLE_TABLE_SIMS.
    _pole_table_sims = getattr(_cfg, 'POLE_TABLE_SIMS', [])
    if _pole_table_sims:
        pole_tpath = os.path.join(out_dir, 'table_pole_comparison.tex')
        generate_pole_comparison_table(
            sims      = sims,
            names     = names,
            labels    = labels,
            pole_sims = _pole_table_sims,
            out_path  = pole_tpath,
        )

    # Write observational dataset summary table if config provides OBS_DATASET_TABLE.
    _obs_table_cfg = getattr(_cfg, 'OBS_DATASET_TABLE', None)
    if _obs_table_cfg:
        obs_tpath = os.path.join(out_dir, 'table_obs_dataset.tex')
        generate_obs_dataset_table(
            sims               = sims,
            names              = names,
            obs_folder         = _obs_table_cfg.get('obs_folder', 'Observations/AllModernJ2000'),
            raw_obs_folder     = _obs_table_cfg.get('raw_obs_folder', ''),
            obs_types_override = _obs_table_cfg.get('obs_types', {}),
            sim_for_initial    = _obs_table_cfg.get('sim_for_initial', names[0] if names else None),
            caption            = _obs_table_cfg.get('caption', 'Observational dataset summary.'),
            label              = _obs_table_cfg.get('label', 'tab:obs-dataset-summary'),
            out_path           = obs_tpath,
        )

    print(f"\nDone.  Output → {out_dir}")


if __name__ == '__main__':
    main()
