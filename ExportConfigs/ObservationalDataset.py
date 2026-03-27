"""Export configuration: ObservationalDataset
Observational dataset summary — per-ID residuals vs NEP097 (SPICE) and vs
the initial propagation, histograms, and a LaTeX table.

Edit the settings below, then run:
    python MatplotlibExport.py
after setting ACTIVE_CONFIG = 'ObservationalDataset' in MatplotlibExport.py.
"""

# ── Dataset selection ─────────────────────────────────────────────────────────
# Set to the key in DashInteractivePlotFull DATA_FILES that contains a simulation
# with real-observation residuals (residual_history_arcseconds + weight_info).
DATASET_LABEL = 'PoleEst_MB'   # TODO: set to the matching pkl label

# The simulation used for the "initial propagation" residual columns.
# Any sim from a real-observations run with residual_history_arcseconds works.
_SIM_INITIAL = 'IAUPole_initial_state'        # TODO: set to a sim name in that dataset

SELECTED_SIMS = [_SIM_INITIAL]
SIM_LABELS    = ['initial propagation']

OUTPUT_DIR = 'ThesisFigures'

# ── Paths to observation files ────────────────────────────────────────────────
# Folder containing the processed CSVs (Triton_<code>_<nmXXXX>.csv).
# Each file must have 5 columns: time, RA, Dec, O-C RA, O-C Dec.
# O-C RA is stored as 2π + residual_rad; O-C Dec is the residual_rad directly.
OBS_FOLDER = 'Observations/AllModernJ2000'

# Folders containing the raw NSDC text files, used to read the observation
# type (ABS / REL) from the first word of each file header.
#   RawRelativeObservations/ — holds REL datasets (nm0002, nm0003, nm0004, …)
#   NeptuneObservations/     — holds ABS datasets (nm0007, nm0013, nm0015, …)
RAW_OBS_FOLDERS = [
    'Observations/RawRelativeObservations',
    'Observations/NeptuneObservations',
]

# ── Observation type override ─────────────────────────────────────────────────
# Only needed for IDs whose raw NSDC file is not available in RAW_OBS_FOLDERS.
# The code reads ABS/REL directly from the raw files, so this dict can stay
# empty unless you need to override an auto-detected value.
OBS_TYPES_OVERRIDE = {}

# ── LaTeX table settings ──────────────────────────────────────────────────────
# Passed to generate_obs_dataset_table() via main().
# Remove or set to None to skip table generation.
OBS_DATASET_TABLE = {
    'obs_folder':     OBS_FOLDER,
    'raw_obs_folder': RAW_OBS_FOLDERS,   # list — searched in order
    'obs_types':      OBS_TYPES_OVERRIDE,
    'sim_for_initial': _SIM_INITIAL,
    # Caption and label for the LaTeX table.
    'caption': (
        r'Summary of astrometric observation datasets used in this work. '
        r'$N$ is the number of observations per dataset. '
        r'Type indicates whether the original data are relative (Rel.) or absolute (Abs.) '
        r'observations. '
        r'RMS$_{\mathrm{RA}}^{\mathrm{NEP097}}$ and RMS$_{\mathrm{Dec}}^{\mathrm{NEP097}}$ '
        r'are the root-mean-square residuals against the NEP097 ephemeris. '
        r'RMS$_{\mathrm{RA}}^{\mathrm{init}}$ and RMS$_{\mathrm{Dec}}^{\mathrm{init}}$ '
        r'are the residuals against the initial propagation before estimation.'
    ),
    'label': 'tab:obs-dataset-summary',
}

# ── Figures to export ─────────────────────────────────────────────────────────
FIGURES_TO_EXPORT = [
    # ── Residual time series vs NEP097 ────────────────────────────────────────
    ('obs_spice_timeseries', {
        'obs_folder': OBS_FOLDER,
        'title':      r'O$-$C Residuals vs NEP097',
    }),
    # ── Residual time series vs initial propagation ───────────────────────────
    ('obs_initial_timeseries', {
        'sim_name': _SIM_INITIAL,
        'title':    'Residuals vs Initial Propagation',
    }),
    # ── Per-ID residual histograms (both SPICE and initial propagation) ────────
    # Returns one figure per observation ID → saved as separate PDF files.
    ('obs_histogram_per_id', {
        'sim_name':   _SIM_INITIAL,
        'obs_folder': OBS_FOLDER,
        'source':     'both',    # 'spice', 'initial', or 'both'
        'bins':       30,
        'fit_gauss':  True,
        'title':      'Residual Histograms',
    }),
]

SINGLE_SIM_TABLES = []

# No per-sim colors needed (single-sim config).
SIM_COLORS    = {_SIM_INITIAL: '#1f77b4'}
SIM_MARKERS   = {_SIM_INITIAL: 'o'}
SIM_LINESTYLE = {_SIM_INITIAL: '-'}
