"""Export configuration: CASE1_Manual_Bias
Pole estimation with manual bias correction — IAUPole and SimPole variants.
"""

DATASET_LABEL = 'PoleEst_MB'

SELECTED_SIMS = [
    'IAUPole_initial_state',
    'IAUPole_pole_pos_cov',
    'IAUPole_pole_lib_cov',
    'IAUPole_pole_pos_cov_pole_lib_cov',
    'SimPole_initial_state',
    'SimPole_pole_pos_cov',
    'SimPole_pole_lib_cov',
    'SimPole_pole_pos_cov_pole_lib_cov',
]

SIM_LABELS = [s.replace('_cov', '') for s in SELECTED_SIMS]

OUTPUT_DIR = 'ThesisFigures'

FIGURES_TO_EXPORT = [
    # ── Initial propagation (pre-estimation) RSW diff ────────────────────────
    ('rsw_initial', {
        'sim_subset': ['IAUPole_initial_state', 'SimPole_initial_state'],
        'title': 'Initial RSW Difference vs NEP097 (pre-estimation)',
    }),
    ('rsw_initial', {
        'fig_label': 'simpole_m',
        'sim_subset': ['SimPole_initial_state'],
        'unit': 'm',
        'title': 'SimPole Initial RSW Difference vs NEP097 (pre-estimation)',
    }),
    # ── Goodness of fit (3-panel: WRMS / RMS / Cost) ─────────────────────────
    ('gof_combined', {'show_initial': True, 'title': 'Goodness of Fit — WRMS, RMS, Cost Function'}),
    # ── RMS vs NEP097 [km] ────────────────────────────────────────────────────
    ('rms_compare', {'title': 'Total RMS vs NEP097'}),
    # ── RSW Statistics vs NEP097 — all simulations ────────────────────────────
    ('rsw_stats', {
        'fig_label': 'all',
        'title': 'RSW Statistics vs NEP097 — All Simulations',
    }),
    # ── RSW Statistics vs NEP097 — prominent solutions only ──────────────────
    ('rsw_stats', {
        'fig_label': 'prominent',
        'sim_subset': [
            'IAUPole_pole_lib_cov',
            'IAUPole_pole_pos_cov_pole_lib_cov',
            'SimPole_pole_lib_cov',
            'SimPole_pole_pos_cov_pole_lib_cov',
        ],
        'title': 'RSW Statistics vs NEP097 — Prominent Solutions',
    }),
    # ── Parameter update bar charts ──────────────────────────────────────────
    ('param_state',    {'variant': 'iau'}),
    ('param_state',    {'variant': 'simpole'}),
    ('param_state',    {'variant': 'combined'}),
    ('param_rsw',      {'variant': 'iau'}),
    ('param_rsw',      {'variant': 'simpole'}),
    ('param_rsw',      {'variant': 'combined'}),
    ('param_pole_pos', {'variant': 'iau'}),
    ('param_pole_pos', {'variant': 'simpole'}),
    ('param_pole_pos', {'variant': 'combined'}),
    ('param_pole_lib', {'variant': 'iau'}),
    ('param_pole_lib', {'variant': 'simpole'}),
    ('param_pole_lib', {'variant': 'combined'}),
    # ── RSW difference with zoom (full + zoomed subfigure) ───────────────────
    ('rsw_with_zoom', {
        'fig_label':  'prominent',
        'sim_subset': [
            'IAUPole_pole_pos_cov_pole_lib_cov', 'IAUPole_pole_lib_cov',
            'SimPole_pole_pos_cov_pole_lib_cov', 'SimPole_pole_lib_cov',
        ],
        'title': 'RSW Difference vs NEP097 — Prominent Pole Estimations',
    }),
    ('rsw_with_zoom', {
        'fig_label':  'pole_lib',
        'sim_subset': [
            'IAUPole_pole_pos_cov', 'SimPole_pole_pos_cov',
            'IAUPole_pole_lib_cov', 'SimPole_pole_lib_cov',
        ],
        'title': 'RSW Difference vs NEP097 — Pole Position vs Pole Libration Estimations',
    }),
    # ── Formal errors with zoom ──────────────────────────────────────────────
    ('formal_with_zoom', {
        'fig_label':  'prominent',
        'sim_subset': [
            'IAUPole_pole_pos_cov_pole_lib_cov', 'IAUPole_pole_lib_cov',
            'SimPole_pole_pos_cov_pole_lib_cov', 'SimPole_pole_lib_cov',
        ],
        'title': 'Formal Errors RSW — Prominent Pole Estimations',
    }),
    ('formal_with_zoom', {
        'fig_label':  'pole_lib',
        'sim_subset': [
            'IAUPole_pole_pos_cov', 'SimPole_pole_pos_cov',
            'IAUPole_pole_lib_cov', 'SimPole_pole_lib_cov',
        ],
        'title': 'Formal Errors RSW — Pole Position vs Pole Libration Estimations',
    }),
]

SINGLE_SIM_TABLES = [
    'SimPole_pole_lib_cov',
]

# Per-simulation colors (Wong colorblind-safe palette).
# IAU sims use cool tones; SimPole sims use warm tones.
SIM_COLORS = {
    'IAUPole_pole_pos_cov_pole_lib_cov': '#0072B2',  # blue
    'IAUPole_pole_lib_cov':              '#009E73',  # green
    'IAUPole_pole_pos_cov':              '#56B4E9',  # sky blue
    'IAUPole_initial_state':             '#4477AA',  # muted blue
    'SimPole_pole_pos_cov_pole_lib_cov': '#D55E00',  # vermillion
    'SimPole_pole_lib_cov':              '#E69F00',  # amber
    'SimPole_pole_pos_cov':              '#CC79A7',  # rose
    'SimPole_initial_state':             '#AA3377',  # purple
}
SIM_LINESTYLE = {
    'IAUPole_pole_pos_cov_pole_lib_cov': '-',
    'IAUPole_pole_lib_cov':              '-',
    'IAUPole_pole_pos_cov':              '-',
    'IAUPole_initial_state':             '-',
    'SimPole_pole_pos_cov_pole_lib_cov': '--',
    'SimPole_pole_lib_cov':              '--',
    'SimPole_pole_pos_cov':              '--',
    'SimPole_initial_state':             '--',
}
