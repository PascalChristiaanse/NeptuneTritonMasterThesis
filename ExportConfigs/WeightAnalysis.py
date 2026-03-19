"""Export configuration: WeightAnalysis
Comparison of weight schemes — all variants estimate initial_state only,
using SimPole starting conditions with manual bias applied.
"""

DATASET_LABEL = 'WeightAnalysis'  # TODO: set to matching key in DashInteractivePlotFull DATA_FILES

SELECTED_SIMS = [
    'id_new_1_weights',
    'id_new_2_weights',
    'id_weights',
    'tf_weights',
    'tf_weights_no_limit',
    'hybrid_weights',
    'hybrid_old_weights',
    'hybrid_new_id_weights',
    'hybrid_old_new_id_weights',
]

SIM_LABELS = [
    'scaled_per_file_alt',
    'scaled_per_file',
    'per_file',
    'per_timeframe',
    'per_timeframe_no_limit',
    'hybrid_g',
    'hybrid_a',
    'hybrid_g_scaled',
    'hybrid_a_scaled',
]

OUTPUT_DIR = 'ThesisFigures'

FIGURES_TO_EXPORT = [
    # ── Goodness of fit (3-panel: WRMS / RMS / Cost) ─────────────────────────
    ('gof_combined', {'show_initial': True, 'title': 'Goodness of Fit — Weight Scheme Comparison'}),
    # ── RMS vs NEP097 [km] ────────────────────────────────────────────────────
    ('rms_compare', {'title': 'Total RMS vs NEP097 — Weight Scheme Comparison'}),
    # ── RSW Statistics vs NEP097 — all weight schemes ─────────────────────────
    ('rsw_stats', {
        'fig_label': 'all',
        'title': 'RSW Statistics vs NEP097 — Weight Scheme Comparison',
    }),
    # ── RSW difference with zoom ──────────────────────────────────────────────
    ('rsw_with_zoom', {
        'fig_label': 'subset',
        'sim_subset': ['id_weights', 'id_new_2_weights', 'tf_weights', 'hybrid_old_new_id_weights'],
        'title': 'RSW Difference vs NEP097 — Weight Scheme Comparison',
    }),
    # ── Formal errors with zoom ───────────────────────────────────────────────
    ('formal_with_zoom', {
        'fig_label': 'subset',
        'sim_subset': ['id_weights', 'id_new_2_weights', 'tf_weights', 'hybrid_old_new_id_weights'],
        'title': 'Formal Errors RSW — Weight Scheme Comparison',
    }),
]

SINGLE_SIM_TABLES = []

# Per-simulation colors — Paul Tol "muted" palette (colorblind-safe, 9 colors).
# Distinct from the Wong palette used in CASE1_Manual_Bias.
SIM_COLORS = {
    'id_weights':              '#332288',  # indigo
    'id_new_2_weights':        '#117733',  # green
    'id_new_1_weights':        '#44AA99',  # teal
    'tf_weights':              '#88CCEE',  # cyan
    'tf_weights_no_limit':     '#DDCC77',  # sand
    'hybrid_weights':          '#CC6677',  # rose
    'hybrid_old_weights':      '#882255',  # wine
    'hybrid_new_id_weights':   '#AA4499',  # purple
    'hybrid_old_new_id_weights': '#999933', # olive
}
SIM_LINESTYLE = {sn: '-' for sn in SIM_COLORS}
