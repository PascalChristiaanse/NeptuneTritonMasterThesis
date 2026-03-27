"""Export configuration: WeightComparison
Cross-dataset comparison of three weight-scheme analyses:
  • WeightAnalysis      — initial_state only, current dataset
  • WeightAnalysis_Old  — initial_state only, old dataset
  • WeightAnalysis_Pole — initial_state + pole (pole_lib_cov), current dataset

Each figure has 3 panels (one per dataset) so results can be compared
side-by-side at a glance.
"""

# WeightComparison is a meta-config: it does not load a single dataset but
# drives multi-dataset figure functions.  DATASET_LABEL must still point to a
# valid loaded dataset (used as a fallback if a figure function falls back to
# the single-dataset path).
DATASET_LABEL = 'WeightAnalysis'

# SELECTED_SIMS / SIM_LABELS are not used here — each multi-figure function
# loads them directly from the child configs.
SELECTED_SIMS = None
SIM_LABELS    = None

OUTPUT_DIR    = 'ThesisFigures'

# The three datasets and their display titles.
# Order: WA-IAU (old, no pole) → WA-FIT (current, no pole) → WA-FIT-PL (current, with pole)
_DATASETS       = ['WeightAnalysis_Old', 'WeightAnalysis', 'WeightAnalysis_Pole']
_DATASET_TITLES = ['WA-IAU', 'WA-FIT', 'WA-FIT-PL']

FIGURES_TO_EXPORT = [
    # ── RMS vs NEP097 — 3 panels ──────────────────────────────────────────────
    ('rms_compare_multi', {
        'datasets':       _DATASETS,
        'dataset_titles': _DATASET_TITLES,
        'title':          'RMS vs NEP097 — Weight Scheme Comparison',
    }),
    # ── Goodness of fit — one figure per metric, N panels stacked ────────────
    ('gof_metric_multi', {
        'fig_label':      'wrms',
        'datasets':       _DATASETS,
        'dataset_titles': _DATASET_TITLES,
        'metric':         'wrms',
        'show_initial':   True,
        'title':          'WRMS — Weight Scheme Comparison',
    }),
    ('gof_metric_multi', {
        'fig_label':      'rms',
        'datasets':       _DATASETS,
        'dataset_titles': _DATASET_TITLES,
        'metric':         'rms',
        'show_initial':   True,
        'title':          'RMS — Weight Scheme Comparison',
    }),
    ('gof_metric_multi', {
        'fig_label':      'cost',
        'datasets':       _DATASETS,
        'dataset_titles': _DATASET_TITLES,
        'metric':         'cost',
        'show_initial':   True,
        'title':          'Cost Function — Weight Scheme Comparison',
    }),
    # ── Formal error RMS — 3 panels ───────────────────────────────────────────
    ('formal_rms_multi', {
        'datasets':       _DATASETS,
        'dataset_titles': _DATASET_TITLES,
        'title':          'Formal Error RMS — Weight Scheme Comparison',
    }),
    # ── RMS / Formal error RMS ratio — 3 panels ───────────────────────────────
    ('rms_ratio_multi', {
        'datasets':       _DATASETS,
        'dataset_titles': _DATASET_TITLES,
        'title':          'RMS / Formal Error RMS — Weight Scheme Comparison',
    }),
]

SINGLE_SIM_TABLES = []

# No per-sim colors/markers needed here — each multi-figure function pulls
# them from the child config modules.
SIM_COLORS    = {}
SIM_LINESTYLE = {}
SIM_MARKERS   = {}
