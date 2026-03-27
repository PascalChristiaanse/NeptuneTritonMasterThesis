"""Export configuration: WeightAnalysis_Pole
Comparison of weight schemes where all variants estimate initial_state + pole
(pole_lib_cov), using SimPole starting conditions with manual bias applied.
Includes a reference run (ref_SimPole_pole_lib_cov) that uses no custom
weight scheme for direct comparison.
"""

DATASET_LABEL = 'WeightAnalysis_Pole'  # TODO: set to matching key in DashInteractivePlotFull DATA_FILES

SELECTED_SIMS = [
    'ref_SimPole_pole_lib_cov',
    'id_new_1_weights',
    'id_new_2_weights',
    'id_weights',
    'tf_weights',
    'tf_weights_no_limit',
    'hybrid_weights',
    'hybrid_old_weights',
    'hybrid_new_id_weights',
    'hybrid_old_new_id_weights',
    'id_new_2_weights_no_cov',
]

SIM_LABELS = [
    'ref',                    # ref_SimPole_pole_lib_cov
    'ID v1',                  # id_new_1_weights
    'ID v2',                  # id_new_2_weights
    'ID',                     # id_weights
    'TF',                     # tf_weights
    'TF no cap',              # tf_weights_no_limit
    'hybrid G',               # hybrid_weights
    'hybrid A',               # hybrid_old_weights
    'hybrid G+v2',            # hybrid_new_id_weights
    'hybrid A+v2',            # hybrid_old_new_id_weights
    'ID v2 no cov',           # id_new_2_weights_no_cov
]

OUTPUT_DIR = 'ThesisFigures'

FIGURES_TO_EXPORT = [
    # ── Standalone legend ─────────────────────────────────────────────────────
    ('legend', {
        'title': 'Weight Scheme Legend (Pole Estimation)',
        'ncols': 2,
        'shape_labels': {
            '*': 'Reference run  (SimPole + lib, no custom weights)',
            'o': 'Weight scheme variants  (SimPole + lib)',
        },
        'notes': [
            'Abbreviations:',
            '  ID            per-observation-file RMSE weights',
            '  TF            per-timeframe RMSE weights',
            '  v1 / v2       two scaling variants of ID weights',
            '  G / A         two hybrid combination variants (G: new ID, A: old ID)',
            '  no cap        no upper limit applied to computed weights',
            '  no cov        no a-priori covariance on pole parameters',
        ],
    }),
    # ── Goodness of fit (3-panel: WRMS / RMS / Cost) ─────────────────────────
    ('gof_combined', {'show_initial': True, 'title': 'Goodness of Fit — Weight Scheme Comparison (Pole Estimation)'}),
    # ── RMS vs NEP097 [km] ────────────────────────────────────────────────────
    ('rms_compare', {'title': 'Total RMS vs NEP097 — Weight Scheme Comparison (Pole Estimation)'}),
    # ── RMS / formal error RMS ratio (total and per-direction) ───────────────
    ('rms_ratio', {'title': 'RMS / Formal Error RMS — Weight Scheme Comparison (Pole Estimation)'}),
    ('rsw_ratio', {'title': 'RSW RMS / Formal σ RMS — Weight Scheme Comparison (Pole Estimation)'}),
    # ── RSW Statistics vs NEP097 — all weight schemes ─────────────────────────
    ('rsw_stats', {
        'fig_label': 'all',
        'title': 'RSW Statistics vs NEP097 — Weight Scheme Comparison (Pole Estimation)',
    }),
    # ── RSW difference with zoom — ref vs ID v2 ──────────────────────────────
    ('rsw_with_zoom', {
        'fig_label': 'ref_vs_idv2',
        'sim_subset': ['ref_SimPole_pole_lib_cov', 'id_new_2_weights'],
        'title': 'RSW Difference vs NEP097 — ref vs ID v2',
    }),
    # ── Formal errors with zoom — ref vs ID v2 ───────────────────────────────
    ('formal_with_zoom', {
        'fig_label': 'ref_vs_idv2',
        'sim_subset': ['ref_SimPole_pole_lib_cov', 'id_new_2_weights'],
        'title': 'Formal Errors RSW — ref vs ID v2',
    }),
    # ── RSW difference with zoom — ref vs TF ─────────────────────────────────
    ('rsw_with_zoom', {
        'fig_label': 'ref_vs_tf',
        'sim_subset': ['ref_SimPole_pole_lib_cov', 'tf_weights'],
        'title': 'RSW Difference vs NEP097 — ref vs TF',
    }),
    # ── Formal errors with zoom — ref vs TF ──────────────────────────────────
    ('formal_with_zoom', {
        'fig_label': 'ref_vs_tf',
        'sim_subset': ['ref_SimPole_pole_lib_cov', 'tf_weights'],
        'title': 'Formal Errors RSW — ref vs TF',
    }),
    # ── RSW difference with zoom — ref vs hybrid A ────────────────────────────
    ('rsw_with_zoom', {
        'fig_label': 'ref_vs_hybrid_a',
        'sim_subset': ['ref_SimPole_pole_lib_cov', 'hybrid_old_weights'],
        'title': 'RSW Difference vs NEP097 — ref vs hybrid A',
    }),
    # ── Formal errors with zoom — ref vs hybrid A ─────────────────────────────
    ('formal_with_zoom', {
        'fig_label': 'ref_vs_hybrid_a',
        'sim_subset': ['ref_SimPole_pole_lib_cov', 'hybrid_old_weights'],
        'title': 'Formal Errors RSW — ref vs hybrid A',
    }),
    # ── Pole parameter updates ────────────────────────────────────────────────
    ('param_pole_lib', {'variant': 'combined'}),
]

SINGLE_SIM_TABLES = []

# Data for the weight-scheme overview table (Label, Weighting Scheme, Comments).
# Each entry corresponds to one row in the LaTeX table.
SIM_TABLE_ROWS = [
    ('ref',           'none (unit weights)',              r'SimPole + pole lib, reference run'),
    ('ID v1',         'per file (scaled, variant 1)',     r'—'),
    ('ID v2',         'per file (scaled, variant 2)',     r'—'),
    ('ID',            'per file',                         r'—'),
    ('TF',            'per timeframe',                    r'$v_{\min} = 10$\,mas'),
    ('TF no cap',     'per timeframe',                    r'no $v_{\min}$'),
    ('hybrid G',      'per timeframe + per file',         r'geometric mean'),
    ('hybrid A',      'per timeframe + per file',         r'arithmetic mean'),
    ('hybrid G+v2',   'per timeframe + scaled per file',  r'geometric mean'),
    ('hybrid A+v2',   'per timeframe + scaled per file',  r'arithmetic mean'),
    ('ID v2 no cov',  'per file (scaled, variant 2)',     r'no a-priori pole covariance'),
]

# Human-readable descriptions for the naming-convention LaTeX table.
SIM_DESCRIPTIONS = {
    'ref_SimPole_pole_lib_cov':    r'Reference run: SimPole rotation model, estimating state + pole librations, no custom weight scheme',
    'id_new_1_weights':            r'Per-file RMSE weights, scaled by $\sigma_{\mathrm{global}} / \sigma_{\mathrm{file}}$ (variant 1)',
    'id_new_2_weights':            r'Per-file RMSE weights, scaled by $\sigma_{\mathrm{global}} / \sigma_{\mathrm{file}}$ (variant 2)',
    'id_weights':                  r'Per-observation-file RMSE weights ($1/\sigma_{\mathrm{file}}^2$)',
    'tf_weights':                  r'Per-timeframe RMSE weights ($1/\sigma_{\mathrm{tf}}^2$)',
    'tf_weights_no_limit':         r'Per-timeframe RMSE weights, no upper limit on weight magnitude',
    'hybrid_weights':              r'Hybrid G: geometric mean of per-file and per-timeframe weights',
    'hybrid_old_weights':          r'Hybrid A: arithmetic mean of per-file and per-timeframe weights',
    'hybrid_new_id_weights':       r'Hybrid G with ID\,v2 scaling applied to per-file component',
    'hybrid_old_new_id_weights':   r'Hybrid A with ID\,v2 scaling applied to per-file component',
    'id_new_2_weights_no_cov':     r'ID\,v2 weights without a-priori covariance on pole parameters',
}

# Per-simulation colors — Paul Tol "muted" palette (colorblind-safe).
# Reference run uses a black star to distinguish it from weight-scheme variants.
# All weight-scheme sims use circle markers.
SIM_MARKERS = {
    'ref_SimPole_pole_lib_cov':    '*',
    'id_new_1_weights':            'o',
    'id_new_2_weights':            'o',
    'id_weights':                  'o',
    'tf_weights':                  'o',
    'tf_weights_no_limit':         'o',
    'hybrid_weights':              'o',
    'hybrid_old_weights':          'o',
    'hybrid_new_id_weights':       'o',
    'hybrid_old_new_id_weights':   'o',
    'id_new_2_weights_no_cov':     'o',
}

SIM_COLORS = {
    'ref_SimPole_pole_lib_cov':    '#000000',  # black  — reference
    'id_weights':                  '#332288',  # indigo
    'id_new_2_weights':            '#117733',  # green
    'id_new_1_weights':            '#44AA99',  # teal
    'tf_weights':                  '#88CCEE',  # cyan
    'tf_weights_no_limit':         '#DDCC77',  # sand
    'hybrid_weights':              '#CC6677',  # rose
    'hybrid_old_weights':          '#882255',  # wine
    'hybrid_new_id_weights':       '#AA4499',  # purple
    'hybrid_old_new_id_weights':   '#999933',  # olive
    'id_new_2_weights_no_cov':     '#BBBBBB',  # light grey
}

SIM_LINESTYLE = {
    'ref_SimPole_pole_lib_cov':    '--',  # dashed to distinguish reference
    'id_new_1_weights':            '-',
    'id_new_2_weights':            '-',
    'id_weights':                  '-',
    'tf_weights':                  '-',
    'tf_weights_no_limit':         '-',
    'hybrid_weights':              '-',
    'hybrid_old_weights':          '-',
    'hybrid_new_id_weights':       '-',
    'hybrid_old_new_id_weights':   '-',
    'id_new_2_weights_no_cov':     '-',
}
