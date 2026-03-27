"""Export configuration: SimObs_ParameterAnalysis
Systematic parameter analysis using simulated observations — compares which
parameter sets can be meaningfully estimated, using IAU 2015 and Jacobson 2009
Neptune rotation models.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NAMING CONVENTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pattern: <estimated_params>_<rotation_model>

<rotation_model> suffix
  _IAU        IAU 2015 Neptune rotation model
  _Jacobson   Jacobson 2009 Neptune rotation model

<estimated_params> prefix — what is estimated (always includes initial_state)
  no_est          pure propagation only (no estimation performed)
  initial_state   Triton initial state (6 elements) only
  GM_Triton       initial_state + GM of Triton
  GM_Neptune      initial_state + GM of Neptune
  GM_Both         initial_state + GM of Triton and Neptune
  sh              initial_state + spherical harmonics (C20, C40 / J2, J4)
  GM_SH           initial_state + GM (both) + spherical harmonics
  pole_pos        initial_state + pole position (α₀, δ₀)
  pole_rot        initial_state + pole rotation rate (α̇₀, δ̇₀)
  pole_pos_rot    initial_state + pole position + rotation rate
  pole_lib        initial_state + pole librations (α₁, δ₁)   [IAU lib. terms]
  pole_lib1       initial_state + pole librations, degree-1 Jacobson terms
  pole_lib2       initial_state + pole librations, degree-2 Jacobson terms
  pole_pos_lib    initial_state + pole position + librations
  pole_pos_lib1   initial_state + pole pos. + lib. (degree-1 Jacobson)
  pole_pos_lib2   initial_state + pole pos. + lib. (degree-2 Jacobson)
  pole_full       initial_state + all pole parameters (pos. + rot. + lib.)
  SH_pole_full    initial_state + SH. + all pole parameters
  all             initial_state + GM (both) + SH. + all pole parameters

Marker shape encodes the rotation model (consistent with CASE1_Manual_Bias):
  'o'  →  IAU 2015 rotation model    (solid line)
  'D'  →  Jacobson 2009 rotation model  (dashed line)

NOTE: no_est_IAU and no_est_Jacobson are standalone propagation-only runs
that must be added to SimObs_ParameterAnalysis() VARIANTS dict in
EstimationAnalysisTemplates.py before the first two figures can be produced.
"""

DATASET_LABEL = 'SimObs_ParameterAnalysis'  # must match key in DashInteractivePlotFull DATA_FILES

# ─── Complete ordered list of all simulations ─────────────────────────────────
SELECTED_SIMS = [
    # ── No estimation (pure propagation) ──────────────────────────────────
    'no_est_IAU',
    'no_est_Jacobson',
    # ── State only ────────────────────────────────────────────────────────
    'initial_state_IAU',
    'initial_state_Jacobson',
    # ── Gravitational parameters (IAU) ────────────────────────────────────
    'GM_Triton_IAU',
    'GM_Neptune_IAU',
    'GM_Both_IAU',
    # ── Spherical harmonics (IAU) ──────────────────────────────────────────
    'sh_IAU',
    'GM_SH_IAU',
    # ── Pole parameters — IAU 2015 ─────────────────────────────────────────
    'pole_pos_IAU',
    'pole_rot_IAU',
    'pole_pos_rot_IAU',
    'pole_lib_IAU',
    'pole_pos_lib_IAU',
    'pole_full_IAU',
    # ── Pole parameters — Jacobson 2009 ───────────────────────────────────
    'pole_pos_Jacobson',
    'pole_rot_Jacobson',
    'pole_lib1_Jacobson',
    'pole_lib2_Jacobson',
    'pole_pos_lib1_Jacobson',
    'pole_pos_lib2_Jacobson',
    'pole_full_Jacobson',
    # ── Combined: SH. + full pole ──────────────────────────────────────────
    'SH_pole_full_IAU',
    # ── All parameters ────────────────────────────────────────────────────
    'all_IAU',
]

# Short labels for figures — rotation model encoded by marker shape.
# Abbreviations end with a period (e.g. pos., lib., Jac.) to follow
# scientific notation conventions.
SIM_LABELS = [
    # ── No estimation ──────────────────────────────────────────────────────
    'prop. IAU',              # no_est_IAU
    'prop. Jac.',             # no_est_Jacobson
    # ── State only ────────────────────────────────────────────────────────
    'state',                  # initial_state_IAU
    'state (Jac.)',           # initial_state_Jacobson
    # ── GM (IAU) ──────────────────────────────────────────────────────────
    'state+GM_T',             # GM_Triton_IAU
    'state+GM_N',             # GM_Neptune_IAU
    'state+GM_TN',            # GM_Both_IAU
    # ── SH. (IAU) ──────────────────────────────────────────────────────────
    'state+SH.',              # sh_IAU
    'state+GM+SH.',           # GM_SH_IAU
    # ── Pole IAU ──────────────────────────────────────────────────────────
    'state+pos.',             # pole_pos_IAU
    'state+rot.',             # pole_rot_IAU
    'state+pos.+rot.',        # pole_pos_rot_IAU
    'state+lib.',             # pole_lib_IAU
    'state+pos.+lib.',        # pole_pos_lib_IAU
    'state+full.',            # pole_full_IAU
    # ── Pole Jacobson ─────────────────────────────────────────────────────
    'state+pos. (Jac.)',      # pole_pos_Jacobson
    'state+rot. (Jac.)',      # pole_rot_Jacobson
    'state+lib1. (Jac.)',     # pole_lib1_Jacobson
    'state+lib2. (Jac.)',     # pole_lib2_Jacobson
    'state+pos.+lib1. (Jac.)',  # pole_pos_lib1_Jacobson
    'state+pos.+lib2. (Jac.)',  # pole_pos_lib2_Jacobson
    'state+full. (Jac.)',     # pole_full_Jacobson
    # ── SH. + pole ─────────────────────────────────────────────────────────
    'state+SH.+full.',        # SH_pole_full_IAU
    # ── All ───────────────────────────────────────────────────────────────
    'all',                    # all_IAU
]

OUTPUT_DIR = 'ThesisFigures'

_POLE_IAU = [
    'initial_state_IAU',
    'pole_pos_IAU',
    'pole_rot_IAU',
    'pole_pos_rot_IAU',
    'pole_lib_IAU',
    'pole_pos_lib_IAU',
    'pole_full_IAU',
]
_POLE_ALL_WITH_SH = [
    # ── Spherical harmonics (IAU) — provide SH. context ───────────────────
    'sh_IAU',
    'GM_SH_IAU',
    'SH_pole_full_IAU',
    # ── Pole IAU ──────────────────────────────────────────────────────────
    'pole_pos_IAU',
    'pole_rot_IAU',
    'pole_pos_rot_IAU',
    'pole_lib_IAU',
    'pole_pos_lib_IAU',
    'pole_full_IAU',
    # ── Pole Jacobson ─────────────────────────────────────────────────────
    'pole_pos_Jacobson',
    'pole_rot_Jacobson',
    'pole_lib1_Jacobson',
    'pole_lib2_Jacobson',
    'pole_pos_lib1_Jacobson',
    'pole_pos_lib2_Jacobson',
    'pole_full_Jacobson',
]
_SH_POLE_OVERVIEW = [
    'initial_state_IAU',
    'initial_state_Jacobson',
    'sh_IAU',
    'GM_SH_IAU',
    'pole_pos_IAU',
    'pole_rot_IAU',
    'pole_pos_rot_IAU',
    'pole_lib_IAU',
    'pole_pos_lib_IAU',
    'pole_full_IAU',
]

FIGURES_TO_EXPORT = [
    # ════════════════════════════════════════════════════════════════════
    # 00  Legend
    # ════════════════════════════════════════════════════════════════════
    ('legend', {
        'subfolder': '00_legend',
        'title': 'SimObs Parameter Analysis Legend',
        'ncols': 2,
        'shape_labels': {
            'o': 'IAU 2015 rotation model  (solid line)',
            'D': 'Jacobson 2009 rotation model  (dashed line)',
        },
        'notes': [
            'Abbreviations (period = abbreviated word):',
            '  prop.         pure propagation, no estimation',
            '  state         Triton initial state (6 elements, not abbreviated)',
            '  GM_T / GM_N   gravitational parameter of Triton / Neptune',
            '  GM_TN         GM of both Triton and Neptune',
            '  SH.           spherical harmonics (C20, C40 / J2, J4)',
            '  pos.          pole position (alpha_0, delta_0)',
            '  rot.          pole rotation rate',
            '  lib.          pole librations (alpha_1, delta_1)',
            '  lib1. / lib2. Jacobson degree-1 / degree-2 libration terms',
            '  full.         all pole parameters (pos. + rot. + lib.)',
            '  Jac.          Jacobson 2009 Neptune rotation model',
        ],
    }),

    # ════════════════════════════════════════════════════════════════════
    # 01  Propagation (no estimation) — IAU vs Jacobson
    # ════════════════════════════════════════════════════════════════════
    # no_est sims have diff_SPICE_RSW (not diff_SPICE_RSW_initial).
    ('rsw_with_zoom', {
        'subfolder': '01_propagation',
        'fig_label': 'no_est',
        'sim_subset': ['no_est_IAU', 'no_est_Jacobson'],
        'title': 'RSW Difference vs NEP097 — IAU vs Jacobson (no estimation)',
    }),
    # IAU − Jacobson component-wise difference (RSW frame defined by NEP097).
    ('rsw_initial_diff', {
        'subfolder': '01_propagation',
        'fig_label': 'no_est_diff',
        'sim_subset': ['no_est_IAU', 'no_est_Jacobson'],
        'title': 'IAU − Jacobson Propagation Difference (NEP097 RSW frame)',
    }),

    # ════════════════════════════════════════════════════════════════════
    # 02  State-only estimation — IAU vs Jacobson
    # ════════════════════════════════════════════════════════════════════
    ('rsw_with_zoom', {
        'subfolder': '02_state',
        'fig_label': 'state_iau_vs_jac',
        'sim_subset': ['initial_state_IAU', 'initial_state_Jacobson'],
        'title': 'RSW Difference vs NEP097 — State Only (IAU vs Jacobson)',
    }),
    # variant='combined': no sim starts with 'SimPole', each sim uses its own
    # initial value as reference — correct for SimObs.
    ('param_state', {
        'subfolder': '02_state',
        'fig_label': 'state_iau_vs_jac',
        'variant':   'combined',
        'sim_subset': ['initial_state_IAU', 'initial_state_Jacobson'],
        'title': 'State Update (|Δpos.|, |Δvel.|) — IAU vs Jacobson',
    }),
    ('param_rsw', {
        'subfolder': '02_state',
        'fig_label': 'state_iau_vs_jac',
        'variant':   'combined',
        'sim_subset': ['initial_state_IAU', 'initial_state_Jacobson'],
        'title': 'RSW State Update — IAU vs Jacobson',
    }),
    ('corr_heatmap', {
        'subfolder': '02_state',
        'fig_label': 'initial_state_IAU',
        'sim_name':  'initial_state_IAU',
        'title':     '|Correlations| — State Only (IAU 2015)',
        'cmap':      'YlOrRd',
    }),
    ('corr_heatmap', {
        'subfolder': '02_state',
        'fig_label': 'initial_state_Jacobson',
        'sim_name':  'initial_state_Jacobson',
        'title':     '|Correlations| — State Only (Jacobson 2009)',
        'cmap':      'YlOrRd',
    }),

    # ════════════════════════════════════════════════════════════════════
    # 03  Gravitational parameters (IAU)
    # ════════════════════════════════════════════════════════════════════
    ('rms_compare', {
        'subfolder': '03_gm',
        'fig_label': 'gm_variants',
        'sim_subset': [
            'initial_state_IAU',
            'GM_Triton_IAU',
            'GM_Neptune_IAU',
            'GM_Both_IAU',
        ],
        'title': 'Total RMS vs NEP097 — State vs GM Variants',
    }),
    ('rsw_stats', {
        'subfolder':   '03_gm',
        'fig_label':   'gm_variants',
        'sim_subset': [
            'initial_state_IAU',
            'GM_Triton_IAU',
            'GM_Neptune_IAU',
            'GM_Both_IAU',
        ],
        'show_formal': True,
        'title': 'RSW Statistics vs NEP097 — State vs GM Variants',
    }),
    ('param_state', {
        'subfolder': '03_gm',
        'fig_label': 'gm_variants',
        'variant':   'combined',
        'sim_subset': [
            'initial_state_IAU',
            'GM_Triton_IAU',
            'GM_Neptune_IAU',
            'GM_Both_IAU',
        ],
        'title': 'State Update (|Δpos.|, |Δvel.|) — State vs GM Variants',
    }),
    ('param_gm', {
        'subfolder': '03_gm',
        'fig_label': 'gm_variants',
        'sim_subset': ['GM_Triton_IAU', 'GM_Neptune_IAU', 'GM_Both_IAU'],
        'title': 'Gravitational Parameter Update — GM Variants',
    }),
    ('corr_heatmap', {
        'subfolder': '03_gm',
        'fig_label': 'GM_Triton_IAU',
        'sim_name':  'GM_Triton_IAU',
        'title':     '|Correlations| — State + GM Triton (IAU)',
        'cmap':      'YlOrRd',
    }),
    ('corr_heatmap', {
        'subfolder': '03_gm',
        'fig_label': 'GM_Neptune_IAU',
        'sim_name':  'GM_Neptune_IAU',
        'title':     '|Correlations| — State + GM Neptune (IAU)',
        'cmap':      'YlOrRd',
    }),
    ('corr_heatmap', {
        'subfolder': '03_gm',
        'fig_label': 'GM_Both_IAU',
        'sim_name':  'GM_Both_IAU',
        'title':     '|Correlations| — State + GM Triton + GM Neptune (IAU)',
        'cmap':      'YlOrRd',
    }),

    # ════════════════════════════════════════════════════════════════════
    # 04  Overview — state, SH., and all pole estimations
    # ════════════════════════════════════════════════════════════════════
    ('rms_compare', {
        'subfolder': '04_overview',
        'fig_label': 'sh_pole_iau',
        'sim_subset': _SH_POLE_OVERVIEW,
        'title': 'Total RMS vs NEP097 — State, SH., and Pole Estimations',
    }),
    ('param_state', {
        'subfolder': '04_overview',
        'fig_label': 'sh_pole_iau',
        'variant':   'combined',
        'sim_subset': _SH_POLE_OVERVIEW,
        'title': 'State Update (|Δpos.|, |Δvel.|) — State, SH., and Pole Estimations',
    }),

    # ════════════════════════════════════════════════════════════════════
    # 04b  All pole variants (IAU + Jacobson) + SH. — colored by rotation model
    # ════════════════════════════════════════════════════════════════════
    ('rms_compare', {
        'subfolder': '04_overview',
        'fig_label': 'pole_all_models_with_sh',
        'sim_subset': _POLE_ALL_WITH_SH,
        'use_group_colors': True,
        'title': 'Total RMS vs NEP097 — All Pole Variants + SH. (IAU vs Jacobson)',
    }),

    # ════════════════════════════════════════════════════════════════════
    # 05  Pole estimation variants — IAU 2015
    # show_formal=False: SimObs pole sims do not yield meaningful formal errors.
    # ════════════════════════════════════════════════════════════════════
    ('rsw_stats', {
        'subfolder':   '05_pole_iau',
        'fig_label':   'pole_iau',
        'sim_subset':  _POLE_IAU,
        'show_formal': False,
        'title': 'RSW Statistics vs NEP097 — Pole Estimation Variants (IAU 2015)',
    }),
    ('param_state', {
        'subfolder': '05_pole_iau',
        'fig_label': 'pole_iau',
        'variant':   'combined',
        'sim_subset': _POLE_IAU,
        'title': 'State Update (|Δpos.|, |Δvel.|) — Pole Estimation Variants (IAU 2015)',
    }),
    ('param_rsw', {
        'subfolder': '05_pole_iau',
        'fig_label': 'pole_iau',
        'variant':   'combined',
        'sim_subset': _POLE_IAU,
        'title': 'RSW Position Update — Pole Estimation Variants (IAU 2015)',
    }),
    ('param_pole_pos', {
        'subfolder': '05_pole_iau',
        'fig_label': 'pole_iau',
        'variant':   'combined',
        'sim_subset': [
            'pole_pos_IAU',
            'pole_pos_rot_IAU',
            'pole_pos_lib_IAU',
            'pole_full_IAU',
        ],
        'title': 'Pole Position Update (α₀, δ₀) — IAU 2015',
    }),
    ('param_pole_lib', {
        'subfolder': '05_pole_iau',
        'fig_label': 'pole_iau',
        'variant':   'combined',
        'sim_subset': [
            'pole_lib_IAU',
            'pole_pos_lib_IAU',
            'pole_full_IAU',
        ],
        'title': 'Pole Libration Update (α₁, δ₁) — IAU 2015',
    }),
    ('param_pole_rate', {
        'subfolder': '05_pole_iau',
        'fig_label': 'pole_iau',
        'variant':   'combined',
        'sim_subset': [
            'pole_rot_IAU',
            'pole_pos_rot_IAU',
            'pole_full_IAU',
        ],
        'title': 'Pole Rotation Rate Update (α̇₀, δ̇₀) — IAU 2015',
    }),

    # ════════════════════════════════════════════════════════════════════
    # 06  Spherical harmonics update
    # ════════════════════════════════════════════════════════════════════
    ('param_sh', {
        'subfolder': '06_sh',
        'fig_label': 'sh_iau',
        'sim_subset': ['sh_IAU', 'GM_SH_IAU', 'SH_pole_full_IAU', 'all_IAU'],
        'title': 'Spherical Harmonics Update (C₂₀, C₄₀) — IAU 2015',
    }),
]

# Single-simulation detail tables (LaTeX) written to the output directory.
# NOTE: generate_single_sim_table() uses _get_iau_reference() which looks for
# 'IAUPole_pole_pos_cov_pole_lib_cov' — not present in SimObs naming.  The IAU/SimPole
# split columns will show '---'; the Final and Δ columns for parameters estimated
# by all_IAU (state, GM, SH., pole) are correct.
SINGLE_SIM_TABLES = [
    'all_IAU',
]

# Pole comparison table: compares pole (+ position state) parameter estimates
# across these three variants side-by-side.  Columns: Initial | Est. | Δ | %
# for each sim listed here.  Rows: α₀, δ₀, α₁, δ₁ (where estimated), X, Y, Z.
# Generates: table_pole_comparison.tex in the timestamped output directory.
POLE_TABLE_SIMS = [
    'pole_pos_IAU',
    'pole_lib_IAU',
    'pole_pos_lib_IAU',
]

# ─── Human-readable descriptions for the naming-convention LaTeX table ────────
SIM_DESCRIPTIONS = {
    'no_est_IAU':             r'Pure propagation, IAU 2015 rotation model, no estimation',
    'no_est_Jacobson':        r'Pure propagation, Jacobson 2009 rotation model, no estimation',
    'initial_state_IAU':      r'Triton initial state only ($x, y, z, \dot{x}, \dot{y}, \dot{z}$), IAU 2015',
    'initial_state_Jacobson': r'Triton initial state only, Jacobson 2009 rotation model',
    'GM_Triton_IAU':          r'Initial state + $GM_{\mathrm{Triton}}$, IAU 2015',
    'GM_Neptune_IAU':         r'Initial state + $GM_{\mathrm{Neptune}}$, IAU 2015',
    'GM_Both_IAU':            r'Initial state + $GM_{\mathrm{Triton}}$ + $GM_{\mathrm{Neptune}}$, IAU 2015',
    'sh_IAU':                 r'Initial state + spherical harmonics ($C_{20}, C_{40}$), IAU 2015',
    'GM_SH_IAU':              r'Initial state + GM (both) + spherical harmonics, IAU 2015',
    'pole_pos_IAU':           r'Initial state + pole pos.\ ($\alpha_0, \delta_0$), IAU 2015',
    'pole_rot_IAU':           r'Initial state + pole rot.\ rate ($\dot{\alpha}_0, \dot{\delta}_0$), IAU 2015',
    'pole_pos_rot_IAU':       r'Initial state + pole pos.\ + rot.\ rate, IAU 2015',
    'pole_lib_IAU':           r'Initial state + pole lib.\ ($\alpha_1, \delta_1$), IAU 2015',
    'pole_pos_lib_IAU':       r'Initial state + pole pos.\ + lib., IAU 2015',
    'pole_full_IAU':          r'Initial state + all pole parameters (pos.\ + rot.\ + lib.), IAU 2015',
    'pole_pos_Jacobson':      r'Initial state + pole pos.\ ($\alpha_0, \delta_0$), Jacobson 2009',
    'pole_rot_Jacobson':      r'Initial state + pole rot.\ rate, Jacobson 2009',
    'pole_lib1_Jacobson':     r'Initial state + pole lib., degree-1 Jacobson terms',
    'pole_lib2_Jacobson':     r'Initial state + pole lib., degree-2 Jacobson terms',
    'pole_pos_lib1_Jacobson': r'Initial state + pole pos.\ + lib.\ (degree-1), Jacobson 2009',
    'pole_pos_lib2_Jacobson': r'Initial state + pole pos.\ + lib.\ (degree-2), Jacobson 2009',
    'pole_full_Jacobson':     r'Initial state + all pole parameters, degree-2 Jacobson terms',
    'SH_pole_full_IAU':       r'Initial state + SH.\ ($C_{20}, C_{40}$) + all pole parameters, IAU 2015',
    'all_IAU':                r'All parameters: state + GM (both) + SH.\ + all pole parameters, IAU 2015',
}

# ─── Group colors (suffix-based, used when use_group_colors=True) ────────────
# Keys starting with '_' are matched as suffixes by MatplotlibExport._group_color.
# This allows coloring all IAU sims with one color and all Jacobson sims with
# another, regardless of the estimated-parameter prefix.
SIM_GROUP_COLORS = {
    '_IAU':      '#0072B2',  # blue     — IAU 2015 rotation model
    '_Jacobson': '#D55E00',  # vermillion — Jacobson 2009 rotation model
}

# ─── Per-simulation colors ────────────────────────────────────────────────────
# IAU 2015 sims:    cool tones (blues, greens, teals).
# Jacobson sims:    warm tones (reds, ambers, roses).
# Propagation only: neutral grays.
# all_IAU:          black (distinguishes the maximal parameter set).
#
# Paul Tol "muted" + Wong colorblind-safe palettes.
SIM_COLORS = {
    # ── Propagation only (same hue as corresponding rotation model) ───────
    'no_est_IAU':             '#0072B2',  # blue  — same as initial_state_IAU
    'no_est_Jacobson':        '#D55E00',  # vermillion — same as initial_state_Jacobson
    # ── State only ────────────────────────────────────────────────────────
    'initial_state_IAU':      '#0072B2',  # blue     (Wong)
    'initial_state_Jacobson': '#D55E00',  # vermillion (Wong)
    # ── GM variants (IAU) ─────────────────────────────────────────────────
    'GM_Triton_IAU':          '#332288',  # indigo
    'GM_Neptune_IAU':         '#009E73',  # teal green (Wong)
    'GM_Both_IAU':            '#117733',  # dark green
    # ── SH. variants (IAU) ─────────────────────────────────────────────────
    'sh_IAU':                 '#44AA99',  # muted teal
    'GM_SH_IAU':              '#88CCEE',  # light blue
    # ── Pole IAU — blue / green spectrum ──────────────────────────────────
    'pole_pos_IAU':           '#56B4E9',  # sky blue (Wong)
    'pole_rot_IAU':           '#4477AA',  # medium blue
    'pole_pos_rot_IAU':       '#66CCEE',  # cyan
    'pole_lib_IAU':           '#CCBB44',  # yellow-green
    'pole_pos_lib_IAU':       '#228833',  # green
    'pole_full_IAU':          '#EE8866',  # salmon
    # ── Pole Jacobson — warm spectrum ─────────────────────────────────────
    'pole_pos_Jacobson':      '#E69F00',  # amber (Wong)
    'pole_rot_Jacobson':      '#CC79A7',  # rose (Wong)
    'pole_lib1_Jacobson':     '#DDCC77',  # sand
    'pole_lib2_Jacobson':     '#999933',  # olive
    'pole_pos_lib1_Jacobson': '#CC6677',  # coral
    'pole_pos_lib2_Jacobson': '#882255',  # wine
    'pole_full_Jacobson':     '#AA3377',  # purple-rose
    # ── SH. + full pole ────────────────────────────────────────────────────
    'SH_pole_full_IAU':       '#BBBBBB',  # light gray
    # ── All parameters ────────────────────────────────────────────────────
    'all_IAU':                '#000000',  # black
}

# ─── Per-simulation markers ───────────────────────────────────────────────────
# 'o'  IAU 2015 rotation model
# 'D'  Jacobson 2009 rotation model
SIM_MARKERS = {
    # ── IAU 2015 ──────────────────────────────────────────────────────────
    'no_est_IAU':             'o',
    'initial_state_IAU':      'o',
    'GM_Triton_IAU':          'o',
    'GM_Neptune_IAU':         'o',
    'GM_Both_IAU':            'o',
    'sh_IAU':                 'o',
    'GM_SH_IAU':              'o',
    'pole_pos_IAU':           'o',
    'pole_rot_IAU':           'o',
    'pole_pos_rot_IAU':       'o',
    'pole_lib_IAU':           'o',
    'pole_pos_lib_IAU':       'o',
    'pole_full_IAU':          'o',
    'SH_pole_full_IAU':       'o',
    'all_IAU':                'o',
    # ── Jacobson 2009 ─────────────────────────────────────────────────────
    'no_est_Jacobson':        'D',
    'initial_state_Jacobson': 'D',
    'pole_pos_Jacobson':      'D',
    'pole_rot_Jacobson':      'D',
    'pole_lib1_Jacobson':     'D',
    'pole_lib2_Jacobson':     'D',
    'pole_pos_lib1_Jacobson': 'D',
    'pole_pos_lib2_Jacobson': 'D',
    'pole_full_Jacobson':     'D',
}

# ─── Per-simulation linestyles ────────────────────────────────────────────────
# IAU 2015 sims: solid lines.
# Jacobson sims: dashed lines (mirrors CASE1_Manual_Bias convention for SimPole).
SIM_LINESTYLE = {
    # ── IAU 2015 ──────────────────────────────────────────────────────────
    'no_est_IAU':             '-',
    'initial_state_IAU':      '-',
    'GM_Triton_IAU':          '-',
    'GM_Neptune_IAU':         '-',
    'GM_Both_IAU':            '-',
    'sh_IAU':                 '-',
    'GM_SH_IAU':              '-',
    'pole_pos_IAU':           '-',
    'pole_rot_IAU':           '-',
    'pole_pos_rot_IAU':       '-',
    'pole_lib_IAU':           '-',
    'pole_pos_lib_IAU':       '-',
    'pole_full_IAU':          '-',
    'SH_pole_full_IAU':       '-',
    'all_IAU':                '-',
    # ── Jacobson 2009 ─────────────────────────────────────────────────────
    'no_est_Jacobson':        '--',
    'initial_state_Jacobson': '--',
    'pole_pos_Jacobson':      '--',
    'pole_rot_Jacobson':      '--',
    'pole_lib1_Jacobson':     '--',
    'pole_lib2_Jacobson':     '--',
    'pole_pos_lib1_Jacobson': '--',
    'pole_pos_lib2_Jacobson': '--',
    'pole_full_Jacobson':     '--',
}
