import dash
from dash import dcc, html, callback, Output, Input, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pickle
from scipy import stats as sp_stats

# ============================================================================
# SECTION 1: HELPER FUNCTIONS
# ============================================================================

def epoch_to_datetime(epoch_seconds):
    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    return j2000 + timedelta(seconds=float(epoch_seconds))

def rad_to_mas(radians):
    return radians * 206264806.0

def convert_time_array_to_datetime(time_array):
    return [epoch_to_datetime(float(t)) for t in time_array.flatten()]

# ============================================================================
# SECTION 2: DATA PROCESSING FUNCTIONS
# ============================================================================

def process_weight_info(df):
    df = df.copy()
    df['datetime'] = df['time'].apply(epoch_to_datetime)
    df['ra_residual_mas'] = rad_to_mas(df['ra_residual'])
    df['dec_residual_mas'] = rad_to_mas(df['dec_residual'])
    df['ra_rmse_id_mas'] = rad_to_mas(df['ra_rmse_id'])
    df['dec_rmse_id_mas'] = rad_to_mas(df['dec_rmse_id'])
    df['ra_rmse_tf'] = 1.0 / np.sqrt(df['weight_ra'])
    df['dec_rmse_tf'] = 1.0 / np.sqrt(df['weight_dec'])
    df['ra_rmse_tf_mas'] = rad_to_mas(df['ra_rmse_tf'])
    df['dec_rmse_tf_mas'] = rad_to_mas(df['dec_rmse_tf'])
    return df

def compute_rsw_statistics(diff_rsw):
    result = {}
    for i, comp in enumerate(['R', 'S', 'W']):
        result[comp] = {
            'mean': np.mean(diff_rsw[:, i]),
            'std': np.std(diff_rsw[:, i]),
            'rms': np.sqrt(np.mean(diff_rsw[:, i]**2)),
            'max': np.max(np.abs(diff_rsw[:, i]))
        }
    return result


def get_rsw_times(sim_data, n_points=None):
    """Get datetime array for RSW plots, with multiple fallbacks."""
    if 'time_column' in sim_data:
        return convert_time_array_to_datetime(sim_data['time_column'])
    # Fallback: derive from states_SPICE_with_time (subsampled to match diff_SPICE_RSW length)
    if 'states_SPICE_with_time' in sim_data:
        spice_times = sim_data['states_SPICE_with_time'][:, 0]
        if n_points and len(spice_times) != n_points:
            step = max(1, len(spice_times) // n_points)
            spice_times = spice_times[::step][:n_points]
        return convert_time_array_to_datetime(spice_times.reshape(-1, 1))
    # Fallback: derive from state_history_array
    if 'state_history_array' in sim_data:
        times_arr = sim_data['state_history_array'][:, 0:1]
        if n_points and len(times_arr) != n_points:
            step = max(1, len(times_arr) // n_points)
            times_arr = times_arr[::step][:n_points]
        return convert_time_array_to_datetime(times_arr)
    return list(range(n_points or 0))


def compute_formal_error_statistics(formal_errors_rsw):
    result = {}
    for i, comp in enumerate(['R', 'S', 'W']):
        result[comp] = {
            'mean': np.mean(formal_errors_rsw[:, i]),
            'rms': np.sqrt(np.mean(formal_errors_rsw[:, i]**2)),
            'max': np.max(formal_errors_rsw[:, i])
        }
    return result

def compute_state_updates(simulations, sim_names):
    updates = {}
    for sim_name in sim_names:
        if 'parameter_history' in simulations[sim_name] and 'initial_paramaters' in simulations[sim_name]:
            final = simulations[sim_name]['parameter_history'][:, -1]
            initial = np.array(simulations[sim_name]['initial_paramaters'])
            diff = final - initial
            updates[sim_name] = {
                'pos': diff[:3] / 1000, 'vel': diff[3:6] / 1000,
                'pos_mag': np.linalg.norm(diff[:3]) / 1000,
                'vel_mag': np.linalg.norm(diff[3:6]) / 1000
            }
        else:
            updates[sim_name] = {'pos': np.zeros(3), 'vel': np.zeros(3), 'pos_mag': 0, 'vel_mag': 0}
    return updates

def compute_weight_averages(simulations, sim_names, refpoint_filter=None):
    averages = {}
    for sim_name in sim_names:
        if 'weight_info' not in simulations[sim_name]:
            continue
        df = simulations[sim_name]['weight_info']
        if refpoint_filter:
            df = df[df['ref_point_id'].isin(refpoint_filter)]
        if len(df) == 0:
            continue
        averages[sim_name] = {
            'weight_ra': df['weight_ra'].mean(), 'weight_dec': df['weight_dec'].mean(),
            'ra_rmse_id_mas': df['ra_rmse_id_mas'].mean(), 'dec_rmse_id_mas': df['dec_rmse_id_mas'].mean(),
            'ra_rmse_tf_mas': df['ra_rmse_tf_mas'].mean(), 'dec_rmse_tf_mas': df['dec_rmse_tf_mas'].mean(),
            'ra_residual_mas': df['ra_residual_mas'].mean(), 'dec_residual_mas': df['dec_residual_mas'].mean(),
        }
    return averages

def compute_weight_averages_per_file(simulations, sim_names, refpoint_filter=None):
    averages_per_file = {}
    for sim_name in sim_names:
        if 'weight_info' not in simulations[sim_name]:
            continue
        df = simulations[sim_name]['weight_info']
        if refpoint_filter:
            df = df[df['ref_point_id'].isin(refpoint_filter)]
        averages_per_file[sim_name] = {}
        for ref_id in df['ref_point_id'].unique():
            df_ref = df[df['ref_point_id'] == ref_id]
            averages_per_file[sim_name][ref_id] = {
                'weight_ra': df_ref['weight_ra'].mean(), 'weight_dec': df_ref['weight_dec'].mean(),
                'ra_rmse_id_mas': df_ref['ra_rmse_id_mas'].mean(), 'dec_rmse_id_mas': df_ref['dec_rmse_id_mas'].mean(),
                'ra_rmse_tf_mas': df_ref['ra_rmse_tf_mas'].mean(), 'dec_rmse_tf_mas': df_ref['dec_rmse_tf_mas'].mean(),
                'ra_residual_mas': df_ref['ra_residual_mas'].mean(), 'dec_residual_mas': df_ref['dec_residual_mas'].mean(),
            }
    return averages_per_file

def get_timeframe_stats_for_sim(simulations, sim_name):
    if 'weight_info' not in simulations.get(sim_name, {}):
        return {}
    df = simulations[sim_name]['weight_info']
    result = {}
    for ref_id in df['ref_point_id'].unique():
        df_ref = df[df['ref_point_id'] == ref_id]
        result[ref_id] = []
        for tf in df_ref['timeframe'].unique():
            df_tf = df_ref[df_ref['timeframe'] == tf]
            result[ref_id].append({
                'timeframe': tf, 'n_obs': len(df_tf),
                'datetime_start': df_tf['datetime'].min(), 'datetime_end': df_tf['datetime'].max(),
                'datetime_mid': df_tf['datetime'].min() + (df_tf['datetime'].max() - df_tf['datetime'].min()) / 2,
                'ra_rmse_tf_mas': df_tf['ra_rmse_tf_mas'].iloc[0], 'dec_rmse_tf_mas': df_tf['dec_rmse_tf_mas'].iloc[0],
                'ra_rmse_id_mas': df_tf['ra_rmse_id_mas'].iloc[0], 'dec_rmse_id_mas': df_tf['dec_rmse_id_mas'].iloc[0],
            })
    return result

def compute_wrms_and_cost(simulations, sim_names):
    arcsec_to_rad = np.pi / (180 * 3600)
    metrics = {}
    for sim_name in sim_names:
        if 'residual_history_arcseconds' not in simulations[sim_name]:
            continue
        if 'weight_info' not in simulations[sim_name]:
            continue
        residual_history = simulations[sim_name]['residual_history_arcseconds']
        weights_ra = simulations[sim_name]['weight_info']['weight_ra'].values
        weights_dec = simulations[sim_name]['weight_info']['weight_dec'].values
        metrics[sim_name] = {}
        for iteration_name, iteration_idx in [('initial', 0), ('final', -1)]:
            residuals_arcsec = residual_history[iteration_idx]
            residuals_ra_rad = residuals_arcsec[:, 1] * arcsec_to_rad
            residuals_dec_rad = residuals_arcsec[:, 2] * arcsec_to_rad
            wrms = np.sqrt(
                (np.sum(weights_ra * residuals_ra_rad**2) + np.sum(weights_dec * residuals_dec_rad**2)) /
                (np.sum(weights_ra) + np.sum(weights_dec))
            )
            cost = np.sum(weights_ra * residuals_ra_rad**2) + np.sum(weights_dec * residuals_dec_rad**2)
            metrics[sim_name][f'{iteration_name}_wrms_combined_method1_mas'] = rad_to_mas(wrms)
            metrics[sim_name][f'{iteration_name}_cost_function'] = cost
        initial_wrms = metrics[sim_name]['initial_wrms_combined_method1_mas']
        final_wrms = metrics[sim_name]['final_wrms_combined_method1_mas']
        metrics[sim_name]['wrms_improvement_percent'] = ((initial_wrms - final_wrms) / initial_wrms) * 100
        initial_cost = metrics[sim_name]['initial_cost_function']
        final_cost = metrics[sim_name]['final_cost_function']
        metrics[sim_name]['cost_improvement_percent'] = ((initial_cost - final_cost) / initial_cost) * 100
    return metrics

def get_parameter_labels(est_parameters):
    labels = []
    for param_type in est_parameters:
        if param_type == 'initial_state':
            labels.extend(['X', 'Y', 'Z', 'VX', 'VY', 'VZ'])
        elif param_type == 'iau_rotation_model_pole':
            labels.extend(['α₀', 'δ₀'])
        elif param_type == 'iau_rotation_model_pole_librations':
            labels.extend(['α₁', 'δ₁'])
    return labels


def build_residual_df(simulations, sim_name):
    """
    Build a merged residual DataFrame for a simulation by combining
    residual_history_arcseconds (initial + final iteration) with weight_info metadata.
    
    Returns a DataFrame with columns:
        ref_point_id, obs_index, global_obs_index, timeframe, time, datetime,
        weight_ra, weight_dec, weight_type,
        ra_residual_initial_mas, dec_residual_initial_mas,
        ra_residual_final_mas, dec_residual_final_mas,
        ra_weighted_residual_initial, dec_weighted_residual_initial,  (in σ units)
        ra_weighted_residual_final, dec_weighted_residual_final,      (in σ units)
    """
    sim = simulations[sim_name]
    if 'residual_history_arcseconds' not in sim or 'weight_info' not in sim:
        return None
    
    rh = sim['residual_history_arcseconds']
    wi = sim['weight_info']
    
    if len(rh) < 1:
        return None
    
    first_iter = rh[0]   # shape (n_obs, 3): [time_j2000, ra_arcsec, dec_arcsec]
    last_iter = rh[-1]
    
    # arcseconds -> mas
    arcsec_to_mas = 1000.0
    # arcseconds -> radians (for weighted residuals)
    arcsec_to_rad = np.pi / (180.0 * 3600.0)
    
    # Build the DataFrame
    df = wi[['ref_point_id', 'obs_index', 'global_obs_index', 'timeframe',
             'time', 'datetime', 'weight_ra', 'weight_dec', 'weight_type']].copy()
    
    # Residuals in mas
    df['ra_residual_initial_mas'] = first_iter[:, 1] * arcsec_to_mas
    df['dec_residual_initial_mas'] = first_iter[:, 2] * arcsec_to_mas
    df['ra_residual_final_mas'] = last_iter[:, 1] * arcsec_to_mas
    df['dec_residual_final_mas'] = last_iter[:, 2] * arcsec_to_mas
    
    # Weighted residuals: residual_rad * sqrt(weight) → dimensionless (σ units)
    sqrt_wra = np.sqrt(df['weight_ra'].values)
    sqrt_wdc = np.sqrt(df['weight_dec'].values)
    df['ra_weighted_res_initial'] = first_iter[:, 1] * arcsec_to_rad * sqrt_wra
    df['dec_weighted_res_initial'] = first_iter[:, 2] * arcsec_to_rad * sqrt_wdc
    df['ra_weighted_res_final'] = last_iter[:, 1] * arcsec_to_rad * sqrt_wra
    df['dec_weighted_res_final'] = last_iter[:, 2] * arcsec_to_rad * sqrt_wdc
    
    # Uncertainty: 1/sqrt(weight) in radians → converted to mas
    # weight is in 1/rad^2, so 1/sqrt(weight) is in radians
    rad_to_mas_factor = (180.0 * 3600.0 * 1000.0) / np.pi
    with np.errstate(divide='ignore', invalid='ignore'):
        df['uncertainty_ra_mas'] = np.where(df['weight_ra'].values > 0,
            (1.0 / sqrt_wra) * rad_to_mas_factor, np.nan)
        df['uncertainty_dec_mas'] = np.where(df['weight_dec'].values > 0,
            (1.0 / sqrt_wdc) * rad_to_mas_factor, np.nan)
    
    # Number of iterations stored as metadata
    df.attrs['n_iterations'] = len(rh)
    
    return df


# ============================================================================
# SECTION 3: MULTI-FILE DATA LOADING
# ============================================================================

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONFIGURE YOUR DATA FILES HERE                                 ║
# ║  Add/remove entries as needed. Each tuple: (label, file_path)   ║
# ╚══════════════════════════════════════════════════════════════════╝
DATA_FILES = [
    #('SimObs', 'Results/EstimatedParametersSimulatedObservations/NewFinal/Analysis/simulations.pkl'),
    ('WeightAnalysis', 'Results/EstimationTemplatesTest/WeightScheme/Analysis/simulations.pkl'),
    ('UltimateCASE1', 'Results/PoleEstimationRealObservations/UltimateCASE1/Analysis/simulations_with_weights.pkl'),
    #('UltimateCASE1_ManualBias', 'Results/ManualBias/CASE1_Manual_Bias/Analysis/simulations.pkl'),
    # ('PoleInitCASE2', 'Results/PoleEstimationRealObservations/PoleInitCASE2/Analysis/simulations_with_weights.pkl'),
    # ('WeightLoop', 'Results/PoleEstimationRealObservations/EstimationWeightLoop/Analysis/simulations_with_weights.pkl'),
]

all_datasets = {}

# Map of sim_name -> custom label for promoting initial iterations to standalone entries.
# If a sim listed here has 'diff_SPICE_RSW_initial', a synthetic sim is created with the
# given label, using the initial iteration data as if it were the "final" result.
# Edit this dict to add/remove initial-iteration entries.
INITIAL_ITER_AS_SIM = {
    'initial_state_IAU': 'no_est_IAU',
    'initial_state_Jacobson': 'no_est_Jacobson',
}

for dataset_label, file_path in DATA_FILES:
    try:
        with open(file_path, 'rb') as f:
            sims = pickle.load(f)
    except FileNotFoundError:
        print(f"WARNING: File not found: {file_path} — skipping '{dataset_label}'")
        continue

    for sim_name in sims.keys():
        if 'weight_info' in sims[sim_name]:
            sims[sim_name]['weight_info'] = process_weight_info(sims[sim_name]['weight_info'])

    s_names = list(sims.keys())
    
    # Convert formal_errors_RSW to formal_errors_RSW_km if needed
    for sim_name in s_names:
        sim = sims.get(sim_name, {})
        if 'formal_errors_RSW' in sim and 'formal_errors_RSW_km' not in sim:
            # Convert from meters to kilometers
            sim['formal_errors_RSW_km'] = sim['formal_errors_RSW'] #/ 1000
    
    # Build residual DataFrames for each simulation
    for sim_name in s_names:
        rdf = build_residual_df(sims, sim_name)
        if rdf is not None:
            sims[sim_name]['residual_df'] = rdf
        # Compute RMS of initial RSW diff if available
        if 'diff_SPICE_RSW_initial' in sims[sim_name]:
            d_init = sims[sim_name]['diff_SPICE_RSW_initial']
            sims[sim_name]['rms_SPICE_initial'] = float(np.sqrt(np.mean(d_init**2)))

    # Create synthetic sims from initial iterations
    synth_names = []
    for orig_name, synth_label in INITIAL_ITER_AS_SIM.items():
        if orig_name not in sims:
            print(f"  INFO: '{orig_name}' not found in dataset '{dataset_label}' — skipping '{synth_label}'")
            continue
        
        sim = sims[orig_name]
        # Try multiple keys for the initial RSW diff
        init_rsw = None
        init_time = None
        for key in ('diff_SPICE_RSW_initial', 'initial_state_diff_SPICE'):
            if key in sim:
                init_rsw = sim[key]
                print(f"  INFO: Using '{key}' from '{orig_name}' for '{synth_label}' (shape={init_rsw.shape})")
                break
        
        if init_rsw is None:
            print(f"  INFO: No initial RSW diff found for '{orig_name}' — skipping '{synth_label}'")
            print(f"        Available keys: {list(sim.keys())[:15]}...")
            continue

        synth = {}
        synth['diff_SPICE_RSW'] = init_rsw
        synth['rms_SPICE'] = float(np.sqrt(np.mean(init_rsw**2)))
        # Time data
        for tkey in ('time_column_initial', 'time_column'):
            if tkey in sim:
                synth['time_column'] = sim[tkey]
                break
        if 'states_SPICE_with_time' in sim:
            synth['states_SPICE_with_time'] = sim['states_SPICE_with_time']
        
        sims[synth_label] = synth
        synth_names.append(synth_label)
        print(f"  CREATED: '{synth_label}' from '{orig_name}' — RMS={synth['rms_SPICE']:.3f} km")
    
    # Put synthetic names first, then original names
    s_names = synth_names + [n for n in s_names if n not in synth_names]
    
    all_datasets[dataset_label] = {
        'simulations': sims,
        'sim_names': s_names,
        'sims_with_weights': [n for n in s_names if 'weight_info' in sims.get(n, {})],
        'sims_with_formal_errors': [n for n in s_names if 'formal_errors_RSW_km' in sims.get(n, {})],
        'sims_with_correlations': [n for n in s_names if 'correlations' in sims.get(n, {})],
        'sims_with_residuals': [n for n in s_names if 'residual_history_arcseconds' in sims.get(n, {})],
    }

dataset_labels = list(all_datasets.keys())
ALL_LABEL = '── ALL DATASETS ──'

def get_active_data(dataset_choice):
    """Return (simulations_dict, sim_names_list) for the chosen dataset."""
    if dataset_choice == ALL_LABEL:
        merged = {}
        for dl in dataset_labels:
            for sn in all_datasets[dl]['sim_names']:
                merged[f"{dl} :: {sn}"] = all_datasets[dl]['simulations'][sn]
        return merged, list(merged.keys())
    elif dataset_choice in all_datasets:
        ds = all_datasets[dataset_choice]
        return ds['simulations'], ds['sim_names']
    elif dataset_labels:
        ds = all_datasets[dataset_labels[0]]
        return ds['simulations'], ds['sim_names']
    return {}, []

color_palette = (
    px.colors.qualitative.Dark24 +
    px.colors.qualitative.Light24 +
    px.colors.qualitative.Alphabet
)

def get_color_map(names):
    return {name: color_palette[i % len(color_palette)] for i, name in enumerate(names)}

# ============================================================================
# SECTION 4: STYLE CONFIGURATION
# ============================================================================

TITLE_FONT_SIZE = 36
SUBPLOT_TITLE_FONT_SIZE = 28
AXIS_TITLE_FONT_SIZE = 30
TICK_FONT_SIZE = 30
LEGEND_FONT_SIZE = 30
TEXT_FONT_SIZE = 28



# ============================================================================
# SECTION 5: DASH APP LAYOUT
# ============================================================================

app = dash.Dash(__name__, suppress_callback_exceptions=True)

dataset_dropdown_options = (
    [{'label': ALL_LABEL, 'value': ALL_LABEL}] +
    [{'label': dl, 'value': dl} for dl in dataset_labels]
)

app.layout = html.Div([
    html.H1("Simulation Analysis Dashboard", style={'textAlign': 'center', 'fontSize': '28px'}),

    # Font size stores (shared across all tabs)
    dcc.Store(id='font-store', data={
        'title': TITLE_FONT_SIZE, 'subplot': SUBPLOT_TITLE_FONT_SIZE,
        'axis': AXIS_TITLE_FONT_SIZE, 'tick': TICK_FONT_SIZE,
        'legend': LEGEND_FONT_SIZE, 'text': TEXT_FONT_SIZE
    }),

    # Global dataset selector bar
    html.Div([
        html.Label("Dataset:", style={'fontSize': '14px', 'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.Dropdown(
            id='global-dataset-dropdown',
            options=dataset_dropdown_options,
            value=dataset_labels[0] if dataset_labels else ALL_LABEL,
            clearable=False,
            style={'width': '400px', 'fontSize': '13px', 'display': 'inline-block', 'verticalAlign': 'middle'}
        ),
        html.Button("⚙ Font Settings", id='font-toggle-btn', n_clicks=0,
            style={'marginLeft': '20px', 'fontSize': '12px', 'padding': '4px 12px'}),
    ], style={'padding': '10px 20px', 'backgroundColor': '#d4e6f1', 'marginBottom': '10px',
              'display': 'flex', 'alignItems': 'center'}),

    # Collapsible font settings panel
    html.Div([
        html.Div([
            html.Div([
                html.Label(f"{lbl}:", style={'fontSize': '12px', 'width': '100px', 'display': 'inline-block'}),
                dcc.Input(id=f'font-{key}', type='number', value=default, min=6, max=60, step=1,
                    style={'width': '60px', 'fontSize': '12px'}),
            ], style={'display': 'inline-block', 'marginRight': '15px'})
            for key, lbl, default in [
                ('title', 'Title', TITLE_FONT_SIZE), ('subplot', 'Subplot', SUBPLOT_TITLE_FONT_SIZE),
                ('axis', 'Axis', AXIS_TITLE_FONT_SIZE), ('tick', 'Tick', TICK_FONT_SIZE),
                ('legend', 'Legend', LEGEND_FONT_SIZE), ('text', 'Text', TEXT_FONT_SIZE),
            ]
        ] + [html.Button("Apply", id='font-apply-btn', n_clicks=0,
            style={'fontSize': '12px', 'padding': '4px 12px', 'marginLeft': '15px'})],
        style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap'}),
    ], id='font-settings-panel', style={'display': 'none', 'padding': '10px 20px',
        'backgroundColor': '#eaf2f8', 'marginBottom': '10px'}),

    dcc.Tabs(id='main-tabs', value='tab-rsw', children=[
        dcc.Tab(label='RSW Plots', value='tab-rsw'),
        dcc.Tab(label='Observation Analysis', value='tab-obs'),
        dcc.Tab(label='Statistical Comparison', value='tab-stats'),
        dcc.Tab(label='Correlation Analysis', value='tab-corr'),
    ], style={'fontSize': '16px'}),

    html.Div(id='tab-content', style={'padding': '20px'})
])


@callback(Output('font-settings-panel', 'style'), Input('font-toggle-btn', 'n_clicks'))
def toggle_font_panel(n):
    if n and n % 2 == 1:
        return {'display': 'block', 'padding': '10px 20px', 'backgroundColor': '#eaf2f8', 'marginBottom': '10px'}
    return {'display': 'none'}


@callback(Output('font-store', 'data'),
    Input('font-apply-btn', 'n_clicks'),
    State('font-title', 'value'), State('font-subplot', 'value'),
    State('font-axis', 'value'), State('font-tick', 'value'),
    State('font-legend', 'value'), State('font-text', 'value'),
    prevent_initial_call=True)
def update_font_store(n, title, subplot, axis, tick, legend, text):
    return {'title': title or TITLE_FONT_SIZE, 'subplot': subplot or SUBPLOT_TITLE_FONT_SIZE,
            'axis': axis or AXIS_TITLE_FONT_SIZE, 'tick': tick or TICK_FONT_SIZE,
            'legend': legend or LEGEND_FONT_SIZE, 'text': text or TEXT_FONT_SIZE}


def get_fonts(font_data=None):
    """Return font sizes dict, using font_data if provided, else defaults."""
    if font_data:
        return font_data
    return {'title': TITLE_FONT_SIZE, 'subplot': SUBPLOT_TITLE_FONT_SIZE,
            'axis': AXIS_TITLE_FONT_SIZE, 'tick': TICK_FONT_SIZE,
            'legend': LEGEND_FONT_SIZE, 'text': TEXT_FONT_SIZE}


@callback(Output('tab-content', 'children'),
          Input('main-tabs', 'value'),
          Input('global-dataset-dropdown', 'value'))
def render_tab_content(tab, dataset_choice):
    sims, names = get_active_data(dataset_choice)
    if tab == 'tab-rsw':
        return render_rsw_tab(sims, names, dataset_choice)
    elif tab == 'tab-obs':
        return render_observation_tab(sims, names, dataset_choice)
    elif tab == 'tab-stats':
        return render_stats_tab(sims, names, dataset_choice)
    elif tab == 'tab-corr':
        return render_correlation_tab(sims, names, dataset_choice)


# ============================================================================
# SECTION 6: COMBINED RSW PLOTS TAB (Formal Errors + Diff vs SPICE)
# ============================================================================

def render_rsw_tab(simulations, sim_names, dataset_choice):
    sims_fe = [n for n in sim_names if 'formal_errors_RSW_km' in simulations.get(n, {})]
    sims_diff = [n for n in sim_names if 'diff_SPICE_RSW' in simulations.get(n, {})]
    all_avail = sorted(set(sims_fe + sims_diff))

    default = []
    if len(all_avail) >= 2:
        default = [all_avail[0], all_avail[-1]]
    elif all_avail:
        default = [all_avail[0]]

    return html.Div([
        html.Div([
            # Row 1 — Plot mode
            html.Div([
                html.Label("Plot Mode:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                dcc.RadioItems(id='rsw-plot-mode', options=[
                    {'label': 'Formal Errors RSW', 'value': 'formal'},
                    {'label': 'RSW Difference vs SPICE', 'value': 'diff_spice'},
                    {'label': 'Side-by-Side (Formal + Difference)', 'value': 'side_by_side'},
                ], value='formal', inline=True, style={'fontSize': '14px'}),
            ], style={'marginBottom': '15px'}),

            # Row 2 — Sim selection
            html.Div([
                html.Div([
                    html.Label("Select Simulations:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='rsw-sim-dropdown',
                        options=[{'label': n, 'value': n} for n in all_avail],
                        value=default, multi=True,
                        style={'width': '600px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                html.Div([
                    dcc.Checklist(id='rsw-show-stats',
                        options=[{'label': ' Show stats in legend', 'value': 'show'}],
                        value=['show'], style={'fontSize': '14px'}),
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                html.Div([
                    html.Label("Initial Iteration:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.RadioItems(id='rsw-show-initial', options=[
                        {'label': 'Off', 'value': 'off'},
                        {'label': 'Overlay (Final + Initial)', 'value': 'overlay'},
                        {'label': 'Initial Only', 'value': 'initial_only'},
                    ], value='off', inline=True, style={'fontSize': '14px'}),
                ], style={'display': 'inline-block'}),
            ], style={'marginBottom': '15px'}),

            # Row 3 — Diff mode checkbox
            html.Div([
                dcc.Checklist(id='rsw-diff-mode',
                    options=[{'label': ' Show difference between two simulations (sim1 − sim2)', 'value': 'diff'}],
                    value=[], style={'fontSize': '14px'}),
            ], style={'marginBottom': '15px'}),

            # Row 4 — Diff sim selectors (hidden by default)
            html.Div([
                html.Div([
                    html.Label("Diff Data:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.RadioItems(id='rsw-diff-data', options=[
                        {'label': 'Final − Final', 'value': 'final_final'},
                        {'label': 'Initial − Initial', 'value': 'initial_initial'},
                        {'label': 'Final − Initial (same sim)', 'value': 'final_initial'},
                    ], value='final_final', inline=True, style={'fontSize': '14px'}),
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.Label("Simulation 1:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                        dcc.Dropdown(id='rsw-diff-sim1',
                            options=[{'label': n, 'value': n} for n in all_avail],
                            value=all_avail[0] if all_avail else None,
                            clearable=False, style={'width': '280px', 'fontSize': '13px'}),
                    ], style={'display': 'inline-block', 'marginRight': '30px'}),
                    html.Div([
                        html.Label("Simulation 2:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                        dcc.Dropdown(id='rsw-diff-sim2',
                            options=[{'label': n, 'value': n} for n in all_avail],
                            value=all_avail[-1] if len(all_avail) > 1 else (all_avail[0] if all_avail else None),
                            clearable=False, style={'width': '280px', 'fontSize': '13px'}),
                    ], style={'display': 'inline-block'}),
                ]),
            ], id='rsw-diff-controls', style={'display': 'none'}),
        ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginBottom': '20px'}),

        dcc.Store(id='rsw-dataset-store', data=dataset_choice),
        dcc.Graph(id='rsw-graph', style={'height': '900px'},
            config={'toImageButtonOptions': {'format': 'svg', 'filename': 'rsw_plot', 'scale': 2}}),
        html.Div(id='rsw-info-panel',
                 style={'padding': '20px', 'backgroundColor': '#e9ecef', 'marginTop': '10px', 'fontSize': '14px'})
    ])


@callback(Output('rsw-diff-controls', 'style'), Input('rsw-diff-mode', 'value'))
def toggle_rsw_diff(diff_mode):
    return {'display': 'block', 'marginTop': '15px'} if 'diff' in diff_mode else {'display': 'none'}


@callback(
    Output('rsw-graph', 'figure'), Output('rsw-info-panel', 'children'),
    Input('rsw-plot-mode', 'value'), Input('rsw-sim-dropdown', 'value'),
    Input('rsw-show-stats', 'value'), Input('rsw-show-initial', 'value'),
    Input('rsw-diff-mode', 'value'),
    Input('rsw-diff-sim1', 'value'), Input('rsw-diff-sim2', 'value'),
    Input('rsw-diff-data', 'value'),
    Input('rsw-dataset-store', 'data'))
def update_rsw_graph(mode, sel_sims, show_stats, show_initial, diff_mode, d1, d2, diff_data, ds_choice):
    sims, names = get_active_data(ds_choice)
    is_diff = 'diff' in diff_mode
    do_initial = show_initial in ('overlay', 'initial_only')
    initial_only = show_initial == 'initial_only'

    if is_diff:
        if mode == 'formal':
            return _formal_diff(sims, d1, d2)
        elif mode == 'diff_spice':
            return _spice_diff(sims, d1, d2, diff_data)
        else:
            return _side_by_side_diff(sims, d1, d2)
    else:
        if mode == 'formal':
            return _formal_compare(sims, sel_sims, show_stats)
        elif mode == 'diff_spice':
            return _spice_compare(sims, sel_sims, show_stats, do_initial, initial_only)
        else:
            return _side_by_side_compare(sims, sel_sims, show_stats)


# ---------- Formal Errors helpers ----------

def _formal_diff(sims, s1, s2):
    for s in (s1, s2):
        if 'formal_errors_RSW_km' not in sims.get(s, {}):
            return go.Figure(), f"No formal error data for {s}"
    f1, f2 = sims[s1]['formal_errors_RSW_km'], sims[s2]['formal_errors_RSW_km']
    if f1.shape != f2.shape:
        return go.Figure(), f"Shape mismatch: {f1.shape} vs {f2.shape}"
    diff = f1 - f2
    times = convert_time_array_to_datetime(sims[s1]['state_history_array'][:, 0]) if 'state_history_array' in sims[s1] else list(range(len(diff)))

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=[f'R: {s1} − {s2}', f'S: {s1} − {s2}', f'W: {s1} − {s2}'])
    for i, (comp, col) in enumerate(zip(['R','S','W'], ['steelblue','coral','green'])):
        fig.add_trace(go.Scattergl(x=times, y=diff[:, i], mode='lines', name=f'Δσ_{comp}', line=dict(color=col, width=1)), row=i+1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=i+1, col=1)
        fig.update_yaxes(title_text=f"Δσ_{comp} [km]", title_font_size=AXIS_TITLE_FONT_SIZE, row=i+1, col=1)
    fig.update_xaxes(title_text="DateTime", title_font_size=AXIS_TITLE_FONT_SIZE, row=3, col=1)
    fig.update_layout(title=dict(text=f'Formal Error Difference: {s1} − {s2}', x=0.5, font=dict(size=TITLE_FONT_SIZE)),
                      hovermode='x unified', showlegend=False, font=dict(size=TICK_FONT_SIZE))
    fig.update_annotations(font_size=SUBPLOT_TITLE_FONT_SIZE)
    st = compute_rsw_statistics(diff)
    info = [html.B(f"Difference Statistics ({s1} − {s2}):"), html.Br()]
    for c in ['R','S','W']:
        info += [f"{c}: Mean={st[c]['mean']:.6f} km, RMS={st[c]['rms']:.6f} km", html.Br()]
    return fig, info


def _formal_compare(sims, selected, show_stats):
    if not selected:
        return go.Figure(), "Select at least one simulation"
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=['Radial (R) Formal Error', 'Along-track (S) Formal Error', 'Cross-track (W) Formal Error'])
    cmap = get_color_map(selected)
    stats_info = []
    for sn in selected:
        if 'formal_errors_RSW_km' not in sims.get(sn, {}):
            continue
        fe = sims[sn]['formal_errors_RSW_km']
        times = convert_time_array_to_datetime(sims[sn]['state_history_array'][:, 0]) if 'state_history_array' in sims[sn] else list(range(len(fe)))
        st = compute_formal_error_statistics(fe)
        mx = max(st[c]['max'] for c in ['R','S','W'])
        legend = f"{sn} (Max: {mx:.2f} km)" if 'show' in show_stats else sn
        stats_info.append({'name': sn, **{f'{c}_max': st[c]['max'] for c in ['R','S','W']}})
        for i, comp in enumerate(['R','S','W']):
            fig.add_trace(go.Scattergl(x=times, y=fe[:, i], mode='lines', name=legend,
                line=dict(color=cmap[sn], width=1), legendgroup=sn, showlegend=(i==0)), row=i+1, col=1)
    for i, c in enumerate(['R','S','W']):
        fig.update_yaxes(title_text=f"σ_{c} [km]", title_font_size=AXIS_TITLE_FONT_SIZE, row=i+1, col=1)
    fig.update_xaxes(title_text="DateTime", title_font_size=AXIS_TITLE_FONT_SIZE, row=3, col=1)
    fig.update_layout(title=dict(text='Formal Errors RSW', x=0.5, font=dict(size=TITLE_FONT_SIZE)),
        hovermode='x unified', legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=LEGEND_FONT_SIZE)),
        margin=dict(r=200), font=dict(size=TICK_FONT_SIZE))
    fig.update_annotations(font_size=SUBPLOT_TITLE_FONT_SIZE)
    info = [html.B("Formal Error Statistics:"), html.Br()]
    for s in sorted(stats_info, key=lambda x: x['R_max']):
        info += [f"{s['name']}: R={s['R_max']:.4f}, S={s['S_max']:.4f}, W={s['W_max']:.4f} km", html.Br()]
    return fig, info


# ---------- RSW Diff vs SPICE helpers ----------

def _spice_diff(sims, s1, s2, diff_data='final_final'):
    """RSW difference between sims. Modes:
      final_final: diff_SPICE_RSW(s1) - diff_SPICE_RSW(s2)
      initial_initial: diff_SPICE_RSW_initial(s1) - diff_SPICE_RSW_initial(s2)
      final_initial: diff_SPICE_RSW(s1) - diff_SPICE_RSW_initial(s1)  (same sim, s2 ignored)
    """
    if diff_data == 'final_final':
        key = 'diff_SPICE_RSW'
        label1, label2 = f'{s1} Final', f'{s2} Final'
        for s in (s1, s2):
            if key not in sims.get(s, {}):
                return go.Figure(), f"No final RSW diff data for {s}"
        d = sims[s1][key] - sims[s2][key]
        times = get_rsw_times(sims[s1], n_points=len(d))
        title = f'RSW Diff: {s1} Final − {s2} Final'

    elif diff_data == 'initial_initial':
        key = 'diff_SPICE_RSW_initial'
        for s in (s1, s2):
            if key not in sims.get(s, {}):
                return go.Figure(), f"No initial RSW diff data for {s}"
        d1_data = sims[s1][key]
        d2_data = sims[s2][key]
        min_len = min(len(d1_data), len(d2_data))
        d = d1_data[:min_len] - d2_data[:min_len]
        times = get_rsw_times(sims[s1], n_points=min_len)
        title = f'RSW Diff: {s1} Initial − {s2} Initial'

    elif diff_data == 'final_initial':
        k_final = 'diff_SPICE_RSW'
        k_init = 'diff_SPICE_RSW_initial'
        if k_final not in sims.get(s1, {}):
            return go.Figure(), f"No final RSW diff data for {s1}"
        if k_init not in sims.get(s1, {}):
            return go.Figure(), f"No initial RSW diff data for {s1}"
        d_final = sims[s1][k_final]
        d_init = sims[s1][k_init]
        min_len = min(len(d_final), len(d_init))
        d = d_final[:min_len] - d_init[:min_len]
        times = get_rsw_times(sims[s1], n_points=min_len)
        title = f'RSW Diff: {s1} Final − {s1} Initial'
    else:
        return go.Figure(), f"Unknown diff_data mode: {diff_data}"

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=[f'ΔR', f'ΔS', f'ΔW'])
    for i, (comp, col) in enumerate(zip(['R','S','W'], ['steelblue','coral','green'])):
        fig.add_trace(go.Scattergl(x=times, y=d[:, i], mode='lines+markers', name=f'Δ{comp}',
            line=dict(color=col), marker=dict(size=4)), row=i+1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=i+1, col=1)
        fig.update_yaxes(title_text=f"Δ{comp} [km]", title_font_size=AXIS_TITLE_FONT_SIZE, row=i+1, col=1)
    fig.update_xaxes(title_text="DateTime", title_font_size=AXIS_TITLE_FONT_SIZE, row=3, col=1)
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=TITLE_FONT_SIZE)),
        hovermode='x unified', showlegend=False, font=dict(size=TICK_FONT_SIZE))
    fig.update_annotations(font_size=SUBPLOT_TITLE_FONT_SIZE)
    st = compute_rsw_statistics(d)
    info = [html.B(f"{title}:"), html.Br()]
    for c in ['R','S','W']:
        info += [f"{c}: Mean={st[c]['mean']:.3f}, RMS={st[c]['rms']:.3f}, Max={st[c]['max']:.3f} km", html.Br()]
    return fig, info


def _spice_compare(sims, selected, show_stats, show_initial=False, initial_only=False):
    if not selected:
        return go.Figure(), "Select at least one simulation"
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=['Radial (R) Difference', 'Along-track (S) Difference', 'Cross-track (W) Difference'])
    cmap = get_color_map(selected)
    rms_info = []

    for sn in selected:
        col = cmap[sn]

        # Plot final (unless initial_only)
        if not initial_only and 'diff_SPICE_RSW' in sims.get(sn, {}):
            dr = sims[sn]['diff_SPICE_RSW']
            rms_spice = sims[sn].get('rms_SPICE', None)
            times = get_rsw_times(sims[sn], n_points=len(dr))
            st = compute_rsw_statistics(dr)
            legend = f"{sn} (RMS: {rms_spice:.2f} km)" if 'show' in show_stats and rms_spice else sn
            rms_info.append({'name': sn, 'total': rms_spice, **{f'{c}_rms': st[c]['rms'] for c in ['R','S','W']}})
            for i, comp in enumerate(['R','S','W']):
                fig.add_trace(go.Scattergl(x=times, y=dr[:, i], mode='lines+markers', name=legend,
                    line=dict(color=col), marker=dict(size=4), legendgroup=sn, showlegend=(i==0)), row=i+1, col=1)

        # Plot initial iteration
        if (show_initial or initial_only) and 'diff_SPICE_RSW_initial' in sims.get(sn, {}):
            dr_init = sims[sn]['diff_SPICE_RSW_initial']
            t_init = convert_time_array_to_datetime(sims[sn]['time_column_initial'].reshape(-1, 1))
            st_init = compute_rsw_statistics(dr_init)
            rms_init = np.sqrt(np.mean(np.sum(dr_init**2, axis=1)))

            if initial_only:
                init_legend = f"{sn} Init (RMS: {rms_init:.2f} km)" if 'show' in show_stats else f"{sn} Init"
                line_style = dict(color=col, width=2)
                mode_str = 'lines+markers'
                marker_dict = dict(size=4)
                opacity = 1.0
            else:
                init_legend = f"{sn} Init (RMS: {rms_init:.2f} km)" if 'show' in show_stats else f"{sn} Init"
                line_style = dict(color=col, width=1, dash='dot')
                mode_str = 'lines'
                marker_dict = dict(size=3)
                opacity = 0.6

            rms_info.append({'name': f'{sn} (Init)', 'total': rms_init,
                **{f'{c}_rms': st_init[c]['rms'] for c in ['R','S','W']}})
            for i, comp in enumerate(['R','S','W']):
                fig.add_trace(go.Scattergl(x=t_init, y=dr_init[:, i], mode=mode_str,
                    name=init_legend, line=line_style, marker=marker_dict,
                    legendgroup=f'{sn}_init', showlegend=(i==0), opacity=opacity), row=i+1, col=1)

    for i, c in enumerate(['R','S','W']):
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=i+1, col=1)
        fig.update_yaxes(title_text=f"Δ{c} [km]", title_font_size=AXIS_TITLE_FONT_SIZE, row=i+1, col=1)
    fig.update_xaxes(title_text="DateTime", title_font_size=AXIS_TITLE_FONT_SIZE, row=3, col=1)

    title_suffix = ' (Initial)' if initial_only else (' (Final + Initial)' if show_initial else '')
    fig.update_layout(title=dict(text=f'RSW Difference vs SPICE{title_suffix}', x=0.5, font=dict(size=TITLE_FONT_SIZE)),
        hovermode='x unified', legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=LEGEND_FONT_SIZE)),
        margin=dict(r=200), font=dict(size=TICK_FONT_SIZE))
    fig.update_annotations(font_size=SUBPLOT_TITLE_FONT_SIZE)
    info = [html.B("RMS (sorted):"), html.Br()]
    for r in sorted(rms_info, key=lambda x: x['total'] if x['total'] else float('inf')):
        t = f"{r['total']:.2f}" if r['total'] else "N/A"
        info += [f"{r['name']}: Total={t} | R={r['R_rms']:.2f} | S={r['S_rms']:.2f} | W={r['W_rms']:.2f} km", html.Br()]
    return fig, info


# ---------- Side-by-side helpers ----------

def _side_by_side_compare(sims, selected, show_stats):
    if not selected:
        return go.Figure(), "Select at least one simulation"
    fig = make_subplots(rows=3, cols=2, shared_xaxes='columns', vertical_spacing=0.08, horizontal_spacing=0.08,
        subplot_titles=['R Formal Error','R Diff vs SPICE','S Formal Error','S Diff vs SPICE','W Formal Error','W Diff vs SPICE'])
    cmap = get_color_map(selected)
    for sn in selected:
        col = cmap[sn]
        if 'formal_errors_RSW_km' in sims.get(sn, {}):
            fe = sims[sn]['formal_errors_RSW_km']
            t = convert_time_array_to_datetime(sims[sn]['state_history_array'][:, 0]) if 'state_history_array' in sims[sn] else list(range(len(fe)))
            for i in range(3):
                fig.add_trace(go.Scattergl(x=t, y=fe[:, i], mode='lines', name=sn,
                    line=dict(color=col, width=1), legendgroup=sn, showlegend=(i==0)), row=i+1, col=1)
        if 'diff_SPICE_RSW' in sims.get(sn, {}):
            dr = sims[sn]['diff_SPICE_RSW']
            t2 = get_rsw_times(sims[sn], n_points=len(dr))
            for i in range(3):
                fig.add_trace(go.Scattergl(x=t2, y=dr[:, i], mode='lines+markers', name=sn,
                    line=dict(color=col), marker=dict(size=3), legendgroup=sn, showlegend=False), row=i+1, col=2)
    for i, c in enumerate(['R','S','W']):
        fig.update_yaxes(title_text=f"σ_{c} [km]", title_font_size=AXIS_TITLE_FONT_SIZE, row=i+1, col=1)
        fig.update_yaxes(title_text=f"Δ{c} [km]", title_font_size=AXIS_TITLE_FONT_SIZE, row=i+1, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=i+1, col=2)
    fig.update_xaxes(title_text="DateTime", title_font_size=AXIS_TITLE_FONT_SIZE, row=3, col=1)
    fig.update_xaxes(title_text="DateTime", title_font_size=AXIS_TITLE_FONT_SIZE, row=3, col=2)
    fig.update_layout(title=dict(text='Formal Errors (left) vs RSW Diff (right)', x=0.5, font=dict(size=TITLE_FONT_SIZE)),
        hovermode='x unified', legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=LEGEND_FONT_SIZE)),
        margin=dict(r=200), font=dict(size=TICK_FONT_SIZE))
    fig.update_annotations(font_size=SUBPLOT_TITLE_FONT_SIZE)
    return fig, [html.B("Side-by-Side: "), "Left = Formal Errors, Right = Diff vs SPICE"]


def _side_by_side_diff(sims, s1, s2):
    fig = make_subplots(rows=3, cols=2, shared_xaxes='columns', vertical_spacing=0.08, horizontal_spacing=0.08,
        subplot_titles=[f'FE R: {s1}−{s2}', f'Diff R: {s1}−{s2}', f'FE S: {s1}−{s2}', f'Diff S: {s1}−{s2}',
                        f'FE W: {s1}−{s2}', f'Diff W: {s1}−{s2}'])
    colors = ['steelblue','coral','green']
    info = []
    # Left: formal error diff
    if all('formal_errors_RSW_km' in sims.get(s, {}) for s in (s1, s2)):
        fd = sims[s1]['formal_errors_RSW_km'] - sims[s2]['formal_errors_RSW_km']
        t = convert_time_array_to_datetime(sims[s1]['state_history_array'][:, 0]) if 'state_history_array' in sims[s1] else list(range(len(fd)))
        for i in range(3):
            fig.add_trace(go.Scattergl(x=t, y=fd[:, i], mode='lines', line=dict(color=colors[i], width=1), showlegend=False), row=i+1, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=i+1, col=1)
        st = compute_rsw_statistics(fd)
        info += [html.B("Formal Error Diff:"), html.Br()]
        for c in ['R','S','W']:
            info += [f"{c}: RMS={st[c]['rms']:.6f} km", html.Br()]
    # Right: SPICE diff
    if all('diff_SPICE_RSW' in sims.get(s, {}) for s in (s1, s2)):
        sd = sims[s1]['diff_SPICE_RSW'] - sims[s2]['diff_SPICE_RSW']
        t2 = get_rsw_times(sims[s1], n_points=len(sd))
        for i in range(3):
            fig.add_trace(go.Scattergl(x=t2, y=sd[:, i], mode='lines+markers', line=dict(color=colors[i]),
                marker=dict(size=3), showlegend=False), row=i+1, col=2)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=i+1, col=2)
        st2 = compute_rsw_statistics(sd)
        info += [html.Br(), html.B("SPICE Diff:"), html.Br()]
        for c in ['R','S','W']:
            info += [f"{c}: RMS={st2[c]['rms']:.3f} km", html.Br()]
    for i, c in enumerate(['R','S','W']):
        fig.update_yaxes(title_text=f"Δσ_{c} [km]", title_font_size=AXIS_TITLE_FONT_SIZE, row=i+1, col=1)
        fig.update_yaxes(title_text=f"Δ{c} [km]", title_font_size=AXIS_TITLE_FONT_SIZE, row=i+1, col=2)
    fig.update_xaxes(title_text="DateTime", title_font_size=AXIS_TITLE_FONT_SIZE, row=3, col=1)
    fig.update_xaxes(title_text="DateTime", title_font_size=AXIS_TITLE_FONT_SIZE, row=3, col=2)
    fig.update_layout(title=dict(text=f'Side-by-Side Diff: {s1} − {s2}', x=0.5, font=dict(size=TITLE_FONT_SIZE)),
        hovermode='x unified', showlegend=False, font=dict(size=TICK_FONT_SIZE))
    fig.update_annotations(font_size=SUBPLOT_TITLE_FONT_SIZE)
    return fig, info

# ============================================================================
# SECTION 7: WEIGHT ANALYSIS TAB
# ============================================================================

def render_weights_tab(simulations, sim_names, dataset_choice):
    sww = [n for n in sim_names if 'weight_info' in simulations.get(n, {})]
    if not sww:
        return html.Div("No weight information available in any simulation")

    all_refs = sorted(simulations[sww[0]]['weight_info']['ref_point_id'].unique().tolist()) if sww else []

    y_opts_ra = [
        {'label': 'RA Residual [mas]', 'value': 'ra_residual_mas'},
        {'label': 'RA RMSE ID [mas]', 'value': 'ra_rmse_id_mas'},
        {'label': 'RA RMSE TF [mas]', 'value': 'ra_rmse_tf_mas'},
        {'label': 'RA Residual [rad]', 'value': 'ra_residual'},
        {'label': 'RA RMSE ID [rad]', 'value': 'ra_rmse_id'},
        {'label': 'Weight RA', 'value': 'weight_ra'},
    ]
    y_opts_dec = [
        {'label': 'DEC Residual [mas]', 'value': 'dec_residual_mas'},
        {'label': 'DEC RMSE ID [mas]', 'value': 'dec_rmse_id_mas'},
        {'label': 'DEC RMSE TF [mas]', 'value': 'dec_rmse_tf_mas'},
        {'label': 'DEC Residual [rad]', 'value': 'dec_residual'},
        {'label': 'DEC RMSE ID [rad]', 'value': 'dec_rmse_id'},
        {'label': 'Weight DEC', 'value': 'weight_dec'},
    ]

    return html.Div([
        html.Div([
            html.Div([
                html.Label("Plot Type:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                dcc.RadioItems(id='weight-plot-type', options=[
                    {'label': 'Time Series', 'value': 'timeseries'},
                    {'label': 'N_obs per TF vs DateTime', 'value': 'nobs_datetime'},
                    {'label': 'RMSE vs N_obs', 'value': 'rmse_nobs'},
                    {'label': 'Difference Between Sims', 'value': 'diff'},
                ], value='timeseries', inline=True, style={'fontSize': '14px'}),
            ], style={'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.Label("Simulation:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='weight-sim-dropdown', options=[{'label': n, 'value': n} for n in sww],
                        value=sww[0], clearable=False, style={'width': '280px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                html.Div([
                    html.Label("Reference Point:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='weight-refpoint-dropdown',
                        options=[{'label': 'ALL FILES', 'value': 'ALL FILES'}] + [{'label': r, 'value': r} for r in all_refs],
                        value=all_refs[0] if all_refs else 'ALL FILES',
                        clearable=False, style={'width': '250px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block'}),
            ], style={'marginBottom': '15px'}, id='weight-sim-controls'),
            html.Div([
                html.Div([
                    html.Label("Sim 1:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='weight-diff-sim1', options=[{'label': n, 'value': n} for n in sww],
                        value=sww[0], clearable=False, style={'width': '280px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                html.Div([
                    html.Label("Sim 2:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='weight-diff-sim2', options=[{'label': n, 'value': n} for n in sww],
                        value=sww[-1] if len(sww) > 1 else sww[0], clearable=False,
                        style={'width': '280px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block'}),
            ], id='weight-diff-controls', style={'display': 'none', 'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.Label("RA Y-Axis:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='weight-y-ra-dropdown', options=y_opts_ra, value='ra_residual_mas',
                        clearable=False, style={'width': '250px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                html.Div([
                    html.Label("DEC Y-Axis:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='weight-y-dec-dropdown', options=y_opts_dec, value='dec_residual_mas',
                        clearable=False, style={'width': '250px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block'}),
            ], style={'marginBottom': '15px'}, id='weight-yaxis-controls'),
            html.Div([
                dcc.Checklist(id='weight-show-tf-lines', options=[{'label': ' Show TF boundaries', 'value': 'show'}],
                    value=[], style={'display': 'inline-block', 'marginRight': '30px', 'fontSize': '14px'}),
                dcc.Checklist(id='weight-log-y', options=[{'label': ' Log Y', 'value': 'log'}],
                    value=[], style={'display': 'inline-block', 'fontSize': '14px'}),
            ], id='weight-options-controls'),
        ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginBottom': '20px'}),
        dcc.Store(id='weight-dataset-store', data=dataset_choice),
        dcc.Graph(id='weight-graph', style={'height': '800px'},
            config={'toImageButtonOptions': {'format': 'svg', 'filename': 'weight_plot', 'scale': 2}}),
        html.Div(id='weight-info-panel', style={'padding': '20px', 'backgroundColor': '#e9ecef', 'marginTop': '10px', 'fontSize': '14px'})
    ])


@callback(
    Output('weight-sim-controls', 'style'), Output('weight-diff-controls', 'style'),
    Output('weight-yaxis-controls', 'style'), Output('weight-options-controls', 'style'),
    Input('weight-plot-type', 'value'))
def toggle_weight_ctrls(pt):
    ss = {'marginBottom': '15px'}
    ds = {'display': 'none', 'marginBottom': '15px'}
    ys = {'marginBottom': '15px'}
    os = {}
    if pt == 'diff':
        ss = {'display': 'none'}; ds = {'display': 'block', 'marginBottom': '15px'}
    elif pt in ('nobs_datetime', 'rmse_nobs'):
        ys = {'display': 'none'}
    return ss, ds, ys, os


@callback(Output('weight-log-y', 'value'), Input('weight-y-ra-dropdown', 'value'), Input('weight-y-dec-dropdown', 'value'))
def auto_log_w(yra, ydec):
    if (yra and 'weight' in yra) or (ydec and 'weight' in ydec):
        return ['log']
    return []


@callback(
    Output('weight-graph', 'figure'), Output('weight-info-panel', 'children'),
    Input('weight-plot-type', 'value'), Input('weight-sim-dropdown', 'value'),
    Input('weight-refpoint-dropdown', 'value'), Input('weight-y-ra-dropdown', 'value'),
    Input('weight-y-dec-dropdown', 'value'), Input('weight-show-tf-lines', 'value'),
    Input('weight-log-y', 'value'), Input('weight-diff-sim1', 'value'),
    Input('weight-diff-sim2', 'value'), Input('weight-dataset-store', 'data'))
def update_weight(pt, sn, rp, yra, ydec, tf_lines, logy, d1, d2, ds_choice):
    sims, names = get_active_data(ds_choice)
    yra = yra or 'ra_residual_mas'; ydec = ydec or 'dec_residual_mas'
    if pt == 'timeseries':
        return _weight_ts(sims, sn, rp, yra, ydec, tf_lines, logy)
    elif pt == 'nobs_datetime':
        return _nobs_dt(sims, sn, rp)
    elif pt == 'rmse_nobs':
        return _rmse_nobs(sims, sn, rp)
    elif pt == 'diff':
        return _weight_diff(sims, d1, d2, rp, yra, ydec, logy)
    return go.Figure(), ""


def _weight_ts(sims, sn, rp, yra, ydec, tf_lines, logy):
    if not sn or 'weight_info' not in sims.get(sn, {}):
        return go.Figure(), "No weight data"
    df = sims[sn]['weight_info'].copy()
    all_mode = rp == 'ALL FILES'
    if not all_mode:
        df = df[df['ref_point_id'] == rp]
    if len(df) == 0:
        return go.Figure(), "No data for selected ref"
    fig = make_subplots(rows=2, cols=1, subplot_titles=[f'RA ({yra})', f'DEC ({ydec})'],
        shared_xaxes=True, vertical_spacing=0.08)
    if all_mode:
        refs = df['ref_point_id'].unique()
        rcm = {r: color_palette[i % len(color_palette)] for i, r in enumerate(refs)}
        for r in refs:
            dr = df[df['ref_point_id'] == r]
            c = rcm[r]
            fig.add_trace(go.Scattergl(x=dr['datetime'], y=dr[yra], mode='markers', name=r,
                marker=dict(color=c, size=5, opacity=0.7), legendgroup=r), row=1, col=1)
            fig.add_trace(go.Scattergl(x=dr['datetime'], y=dr[ydec], mode='markers', name=r,
                marker=dict(color=c, size=5, opacity=0.7), legendgroup=r, showlegend=False), row=2, col=1)
        title = f'{sn} - All Files ({len(refs)} files)'
    else:
        tfs = sorted(df['timeframe'].unique())
        tcm = {t: color_palette[i % len(color_palette)] for i, t in enumerate(tfs)}
        for t in tfs:
            dt = df[df['timeframe'] == t]
            c = tcm[t]
            fig.add_trace(go.Scattergl(x=dt['datetime'], y=dt[yra], mode='markers', name=f'TF {t}',
                marker=dict(color=c, size=6), legendgroup=f'tf{t}'), row=1, col=1)
            fig.add_trace(go.Scattergl(x=dt['datetime'], y=dt[ydec], mode='markers', name=f'TF {t}',
                marker=dict(color=c, size=6), legendgroup=f'tf{t}', showlegend=False), row=2, col=1)
            if 'show' in tf_lines:
                for row in [1, 2]:
                    fig.add_vline(x=dt['datetime'].min(), line=dict(color=c, width=1, dash='dash'), row=row, col=1, opacity=0.7)
                    fig.add_vline(x=dt['datetime'].max(), line=dict(color=c, width=1, dash='dash'), row=row, col=1, opacity=0.7)
        title = f'{sn} - {rp} ({len(tfs)} TFs)'
    yt = 'log' if 'log' in logy else 'linear'
    def yl(c):
        if c.endswith('_mas'): return c.replace('_mas','') + ' [mas]'
        if 'weight' in c: return c
        return c + ' [rad]'
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=TITLE_FONT_SIZE)), hovermode='closest',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=LEGEND_FONT_SIZE)),
        margin=dict(r=150), yaxis=dict(type=yt, exponentformat='e'), yaxis2=dict(type=yt, exponentformat='e'),
        font=dict(size=TICK_FONT_SIZE))
    fig.update_yaxes(title_text=yl(yra), title_font_size=AXIS_TITLE_FONT_SIZE, row=1, col=1)
    fig.update_yaxes(title_text=yl(ydec), title_font_size=AXIS_TITLE_FONT_SIZE, row=2, col=1)
    fig.update_xaxes(title_text="DateTime", title_font_size=AXIS_TITLE_FONT_SIZE, row=2, col=1)
    fig.update_annotations(font_size=SUBPLOT_TITLE_FONT_SIZE)
    dr_str = f"{df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}"
    return fig, [html.B("Summary: "), f"{len(df)} points | {dr_str}"]


def _nobs_dt(sims, sn, rp):
    tfs = get_timeframe_stats_for_sim(sims, sn)
    if not tfs:
        return go.Figure(), "No TF stats"
    fig = go.Figure()
    if rp == 'ALL FILES':
        for idx, rid in enumerate(tfs.keys()):
            s = tfs[rid]
            c = color_palette[idx % len(color_palette)]
            fig.add_trace(go.Scatter(x=[x['datetime_mid'] for x in s], y=[x['n_obs'] for x in s],
                mode='lines+markers', name=rid, line=dict(color=c), marker=dict(size=8)))
    else:
        if rp not in tfs:
            return go.Figure(), "No data for ref"
        s = tfs[rp]
        fig.add_trace(go.Scatter(x=[x['datetime_mid'] for x in s], y=[x['n_obs'] for x in s],
            mode='lines+markers+text', name='N_obs', line=dict(color='steelblue', width=2),
            marker=dict(size=10), text=[f"TF {x['timeframe']}" for x in s], textposition='top center'))
    fig.update_layout(title=dict(text=f'N_obs - {sn} - {rp}', x=0.5, font=dict(size=TITLE_FONT_SIZE)),
        xaxis=dict(title='DateTime', title_font_size=AXIS_TITLE_FONT_SIZE),
        yaxis=dict(title='N_obs', title_font_size=AXIS_TITLE_FONT_SIZE), font=dict(size=TICK_FONT_SIZE))
    return fig, [html.B("N_obs per timeframe vs time")]


def _rmse_nobs(sims, sn, rp):
    tfs = get_timeframe_stats_for_sim(sims, sn)
    if not tfs:
        return go.Figure(), "No TF stats"
    fig = make_subplots(rows=1, cols=2, subplot_titles=['RA RMSE TF vs N_obs', 'DEC RMSE TF vs N_obs'])
    if rp == 'ALL FILES':
        for idx, rid in enumerate(tfs.keys()):
            s = tfs[rid]
            c = color_palette[idx % len(color_palette)]
            n = [x['n_obs'] for x in s]
            fig.add_trace(go.Scatter(x=n, y=[x['ra_rmse_tf_mas'] for x in s], mode='markers', name=rid,
                marker=dict(color=c, size=8), legendgroup=rid), row=1, col=1)
            fig.add_trace(go.Scatter(x=n, y=[x['dec_rmse_tf_mas'] for x in s], mode='markers', name=rid,
                marker=dict(color=c, size=8), legendgroup=rid, showlegend=False), row=1, col=2)
    else:
        if rp not in tfs:
            return go.Figure(), "No data"
        s = tfs[rp]; n = [x['n_obs'] for x in s]; tl = [f"TF {x['timeframe']}" for x in s]
        fig.add_trace(go.Scatter(x=n, y=[x['ra_rmse_tf_mas'] for x in s], mode='markers+text', name='RA',
            marker=dict(color='steelblue', size=10), text=tl, textposition='top center'), row=1, col=1)
        fig.add_trace(go.Scatter(x=n, y=[x['dec_rmse_tf_mas'] for x in s], mode='markers+text', name='DEC',
            marker=dict(color='coral', size=10), text=tl, textposition='top center'), row=1, col=2)
    fig.update_xaxes(title_text="N_obs", title_font_size=AXIS_TITLE_FONT_SIZE)
    fig.update_yaxes(title_text="RMSE TF [mas]", title_font_size=AXIS_TITLE_FONT_SIZE)
    fig.update_layout(title=dict(text=f'RMSE vs N_obs - {sn} - {rp}', x=0.5, font=dict(size=TITLE_FONT_SIZE)),
        font=dict(size=TICK_FONT_SIZE))
    fig.update_annotations(font_size=SUBPLOT_TITLE_FONT_SIZE)
    return fig, [html.B("RMSE per TF vs N_obs")]


def _weight_diff(sims, s1, s2, rp, yra, ydec, logy):
    if 'weight_info' not in sims.get(s1, {}) or 'weight_info' not in sims.get(s2, {}):
        return go.Figure(), "No weight data"
    d1 = sims[s1]['weight_info'].copy(); d2 = sims[s2]['weight_info'].copy()
    if rp != 'ALL FILES':
        d1 = d1[d1['ref_point_id'] == rp]; d2 = d2[d2['ref_point_id'] == rp]
    dm = pd.merge(d1, d2, on=['ref_point_id', 'obs_index', 'timeframe'], suffixes=('_1', '_2'))
    if len(dm) == 0:
        return go.Figure(), "No matching points"
    dm[f'{yra}_diff'] = dm[f'{yra}_1'] - dm[f'{yra}_2']
    dm[f'{ydec}_diff'] = dm[f'{ydec}_1'] - dm[f'{ydec}_2']
    fig = make_subplots(rows=2, cols=1, subplot_titles=[f'RA: {s1}−{s2}', f'DEC: {s1}−{s2}'],
        shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scattergl(x=dm['datetime_1'], y=dm[f'{yra}_diff'], mode='markers',
        name='RA', marker=dict(color='steelblue', size=5)), row=1, col=1)
    fig.add_trace(go.Scattergl(x=dm['datetime_1'], y=dm[f'{ydec}_diff'], mode='markers',
        name='DEC', marker=dict(color='coral', size=5)), row=2, col=1)
    for r in [1, 2]:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=r, col=1)
    fig.update_layout(title=dict(text=f'Weight Diff: {s1}−{s2}', x=0.5, font=dict(size=TITLE_FONT_SIZE)),
        hovermode='closest', showlegend=False, font=dict(size=TICK_FONT_SIZE))
    fig.update_yaxes(title_text=f"Δ{yra}", title_font_size=AXIS_TITLE_FONT_SIZE, row=1, col=1)
    fig.update_yaxes(title_text=f"Δ{ydec}", title_font_size=AXIS_TITLE_FONT_SIZE, row=2, col=1)
    fig.update_xaxes(title_text="DateTime", title_font_size=AXIS_TITLE_FONT_SIZE, row=2, col=1)
    fig.update_annotations(font_size=SUBPLOT_TITLE_FONT_SIZE)
    return fig, [html.B("Diff: "),
        f"RA Mean={dm[f'{yra}_diff'].mean():.4f}, DEC Mean={dm[f'{ydec}_diff'].mean():.4f}"]


# ============================================================================
# SECTION 8: OBSERVATION ANALYSIS TAB (Residual Explorer)
# ============================================================================

def render_observation_tab(simulations, sim_names, dataset_choice):
    sww = [n for n in sim_names if 'weight_info' in simulations.get(n, {})]
    swr = [n for n in sim_names if 'residual_df' in simulations.get(n, {})]
    all_refs = sorted(simulations[sww[0]]['weight_info']['ref_point_id'].unique().tolist()) if sww else []

    metric_options = [
        {'label': 'Weight RA', 'value': 'weight_ra'}, {'label': 'Weight DEC', 'value': 'weight_dec'},
        {'label': 'RA RMSE ID [mas]', 'value': 'ra_rmse_id_mas'}, {'label': 'DEC RMSE ID [mas]', 'value': 'dec_rmse_id_mas'},
        {'label': 'RA RMSE TF [mas]', 'value': 'ra_rmse_tf_mas'}, {'label': 'DEC RMSE TF [mas]', 'value': 'dec_rmse_tf_mas'},
        {'label': 'RA Residual [mas]', 'value': 'ra_residual_mas'}, {'label': 'DEC Residual [mas]', 'value': 'dec_residual_mas'},
    ]

    return html.Div([
        html.Div([
            # Main plot type selector
            html.Div([
                html.Label("Plot Type:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                dcc.RadioItems(id='obs-plot-type', options=[
                    {'label': 'Avg per Simulation', 'value': 'avg_sim'},
                    {'label': 'Avg per File per Simulation', 'value': 'avg_file'},
                    {'label': 'Residual Explorer', 'value': 'residual_explorer'},
                    {'label': 'Weighted Residual Explorer', 'value': 'weighted_residual_explorer'},
                    {'label': 'Residual Histogram', 'value': 'residual_histogram'},
                    {'label': 'Weighted Residual Histogram', 'value': 'weighted_residual_histogram'},
                    {'label': 'Residual Summary', 'value': 'residual_summary'},
                    {'label': 'Weighted Residual Summary', 'value': 'weighted_residual_summary'},
                ], value='avg_sim', inline=True, style={'fontSize': '14px'}),
            ], style={'marginBottom': '15px'}),
        ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginBottom': '10px'}),

        # --- Controls for avg_sim / avg_file ---
        html.Div([
            html.Div([
                html.Div([
                    html.Label("Metric:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='obs-metric-dropdown', options=metric_options, value='ra_rmse_tf_mas',
                        clearable=False, style={'width': '300px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block', 'marginRight': '20px'}),
                html.Div([
                    html.Label("2nd Metric (side-by-side):", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='obs-metric-dropdown-2',
                        options=[{'label': '— None —', 'value': 'none'}] + metric_options,
                        value='none', clearable=False, style={'width': '300px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block', 'marginRight': '20px'}),
                dcc.Checklist(id='obs-log-y', options=[{'label': ' Log Y', 'value': 'log'}], value=[],
                    style={'display': 'inline-block', 'fontSize': '14px'}),
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Div([
                    html.Label("Simulations:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='obs-avg-sim-dropdown',
                        options=[{'label': n, 'value': n} for n in swr],
                        value=swr, multi=True,
                        style={'width': '600px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block', 'marginRight': '20px'}),
                html.Div([
                    html.Label("Filter Files:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='obs-avg-refpoint',
                        options=[{'label': r, 'value': r} for r in all_refs],
                        value=[], multi=True, placeholder='All files (leave empty)',
                        style={'width': '350px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block'}),
            ]),
        ], id='obs-avg-controls', style={'padding': '10px 20px', 'backgroundColor': '#f8f9fa', 'marginBottom': '10px'}),

        # --- Controls for residual / weighted residual explorer ---
        html.Div([
            html.Div([
                html.Div([
                    html.Label("Simulations:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='obs-res-sim-dropdown',
                        options=[{'label': n, 'value': n} for n in swr],
                        value=[swr[0]] if swr else [], multi=True,
                        style={'width': '500px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                html.Div([
                    html.Label("Group By:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.RadioItems(id='obs-res-groupby', options=[
                        {'label': 'None (all data)', 'value': 'none'},
                        {'label': 'File (ref_point_id)', 'value': 'ref_point_id'},
                        {'label': 'Timeframe', 'value': 'timeframe'},
                    ], value='none', inline=True, style={'fontSize': '14px'}),
                ], style={'display': 'inline-block'}),
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Div([
                    html.Label("Display Mode:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.RadioItems(id='obs-res-mode', options=[
                        {'label': 'Overlay (Initial + Final)', 'value': 'overlay'},
                        {'label': 'Final Only', 'value': 'final_only'},
                        {'label': 'Difference (Final − Initial)', 'value': 'difference'},
                        {'label': '|Difference|', 'value': 'abs_difference'},
                    ], value='overlay', inline=True, style={'fontSize': '14px'}),
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                html.Div([
                    html.Label("Filter Files:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='obs-res-refpoint',
                        options=[{'label': r, 'value': r} for r in all_refs],
                        value=[], multi=True, placeholder='All files (leave empty)',
                        style={'width': '350px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block'}),
            ]),
        ], id='obs-res-controls', style={'display': 'none', 'padding': '10px 20px',
            'backgroundColor': '#f8f9fa', 'marginBottom': '10px'}),

        # --- Controls for histogram modes ---
        html.Div([
            html.Div([
                html.Div([
                    html.Label("Simulations:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='obs-hist-sim-dropdown',
                        options=[{'label': n, 'value': n} for n in swr],
                        value=[swr[0]] if swr else [], multi=True,
                        style={'width': '500px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                html.Div([
                    html.Label("Histogram Data:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.RadioItems(id='obs-hist-data', options=[
                        {'label': 'Final Residuals', 'value': 'final'},
                        {'label': 'Initial Residuals', 'value': 'initial'},
                        {'label': 'Difference (Final − Initial)', 'value': 'diff_iter'},
                    ], value='final', inline=True, style={'fontSize': '14px'}),
                ], style={'display': 'inline-block'}),
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Div([
                    html.Label("Group By:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.RadioItems(id='obs-hist-groupby', options=[
                        {'label': 'None (all data)', 'value': 'none'},
                        {'label': 'File (ref_point_id)', 'value': 'ref_point_id'},
                        {'label': 'Timeframe', 'value': 'timeframe'},
                    ], value='none', inline=True, style={'fontSize': '14px'}),
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                html.Div([
                    html.Label("Filter Files:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='obs-hist-refpoint',
                        options=[{'label': r, 'value': r} for r in all_refs],
                        value=[], multi=True, placeholder='All files (leave empty)',
                        style={'width': '350px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                html.Div([
                    html.Label("Filter Timeframes:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='obs-hist-timeframes',
                        options=[], value=[], multi=True, placeholder='All TFs (leave empty)',
                        style={'width': '300px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block'}),
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Div([
                    html.Label("Diff Between Sims:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.RadioItems(id='obs-hist-sim-diff', options=[
                        {'label': 'Off', 'value': 'off'},
                        {'label': 'On (Sim1 − Sim2)', 'value': 'on'},
                    ], value='off', inline=True, style={'fontSize': '14px'}),
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                html.Div([
                    html.Label("Sim 1:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='obs-hist-diff-sim1',
                        options=[{'label': n, 'value': n} for n in swr],
                        value=swr[0] if swr else None, clearable=False,
                        style={'width': '250px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block', 'marginRight': '15px'}),
                html.Div([
                    html.Label("Sim 2:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='obs-hist-diff-sim2',
                        options=[{'label': n, 'value': n} for n in swr],
                        value=swr[-1] if len(swr) > 1 else (swr[0] if swr else None), clearable=False,
                        style={'width': '250px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block'}),
            ], style={'marginBottom': '10px'}),
            html.Div([
                dcc.Checklist(id='obs-hist-fit-gauss',
                    options=[{'label': ' Fit Gaussian + Normality test', 'value': 'fit'}],
                    value=['fit'], style={'display': 'inline-block', 'marginRight': '30px', 'fontSize': '14px'}),
                html.Div([
                    html.Label("Bins:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Input(id='obs-hist-bins', type='number', value=50, min=10, max=500, step=10,
                        style={'width': '80px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block'}),
            ]),
        ], id='obs-hist-controls', style={'display': 'none', 'padding': '10px 20px',
            'backgroundColor': '#f8f9fa', 'marginBottom': '10px'}),

        # --- Controls for summary modes ---
        html.Div([
            html.Div([
                html.Div([
                    html.Label("Simulations:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='obs-summ-sim-dropdown',
                        options=[{'label': n, 'value': n} for n in swr],
                        value=[swr[0]] if swr else [], multi=True,
                        style={'width': '500px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                html.Div([
                    html.Label("Data:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.RadioItems(id='obs-summ-data', options=[
                        {'label': 'Final', 'value': 'final'},
                        {'label': 'Initial', 'value': 'initial'},
                        {'label': 'Diff (Final − Initial)', 'value': 'diff_iter'},
                        {'label': 'Compare Initial vs Final', 'value': 'compare_init_final'},
                    ], value='final', inline=True, style={'fontSize': '14px'}),
                ], style={'display': 'inline-block'}),
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Div([
                    html.Label("View Level:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.RadioItems(id='obs-summ-level', options=[
                        {'label': 'Global (all data per sim)', 'value': 'global'},
                        {'label': 'Per File (ref_point_id)', 'value': 'per_file'},
                        {'label': 'Per Timeframe (within files)', 'value': 'per_timeframe'},
                    ], value='global', inline=True, style={'fontSize': '14px'}),
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Div([
                    html.Label("Metric:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.RadioItems(id='obs-summ-metric', options=[
                        {'label': 'Mean', 'value': 'mean'},
                        {'label': 'Std', 'value': 'std'},
                        {'label': 'RMS', 'value': 'rms'},
                        {'label': 'Mean ± Std (error bars)', 'value': 'mean_std'},
                        {'label': 'N_obs (count)', 'value': 'n_obs'},
                        {'label': 'Uncertainty [mas]', 'value': 'uncertainty'},
                        {'label': 'N_timeframes', 'value': 'n_timeframes'},
                        {'label': 'N_obs / N_timeframes', 'value': 'nobs_per_tf'},
                        {'label': 'Std / Uncertainty', 'value': 'std_over_unc'},
                    ], value='mean_std', inline=True, style={'fontSize': '14px'}),
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                html.Div([
                    html.Label("Filter Files:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='obs-summ-refpoint',
                        options=[{'label': r, 'value': r} for r in all_refs],
                        value=[], multi=True, placeholder='All files (leave empty)',
                        style={'width': '350px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block'}),
            ]),
        ], id='obs-summ-controls', style={'display': 'none', 'padding': '10px 20px',
            'backgroundColor': '#f8f9fa', 'marginBottom': '10px'}),

        dcc.Store(id='obs-dataset-store', data=dataset_choice),
        dcc.Graph(id='obs-graph', style={'height': '900px'},
            config={'toImageButtonOptions': {'format': 'svg', 'filename': 'obs_plot', 'scale': 2}}),
        html.Div(id='obs-info-panel', style={'padding': '20px', 'backgroundColor': '#e9ecef',
            'marginTop': '10px', 'fontSize': '14px'})
    ])


@callback(Output('obs-avg-controls', 'style'), Output('obs-res-controls', 'style'),
          Output('obs-hist-controls', 'style'), Output('obs-summ-controls', 'style'),
          Input('obs-plot-type', 'value'))
def toggle_obs_controls(pt):
    base = {'padding': '10px 20px', 'backgroundColor': '#f8f9fa', 'marginBottom': '10px'}
    hide = lambda: {**base, 'display': 'none'}
    h = [hide(), hide(), hide(), hide()]
    idx = {'avg_sim': 0, 'avg_file': 0,
           'residual_explorer': 1, 'weighted_residual_explorer': 1,
           'residual_histogram': 2, 'weighted_residual_histogram': 2,
           'residual_summary': 3, 'weighted_residual_summary': 3}
    i = idx.get(pt, -1)
    if i >= 0:
        h[i] = base
    return tuple(h)


@callback(Output('obs-log-y', 'value'), Input('obs-metric-dropdown', 'value'))
def auto_log_o(m):
    return ['log'] if m and 'weight' in m else []


@callback(Output('obs-hist-timeframes', 'options'),
          Input('obs-hist-sim-dropdown', 'value'),
          Input('obs-dataset-store', 'data'))
def update_hist_tf_options(hist_sims, ds_choice):
    sims, _ = get_active_data(ds_choice)
    tfs = set()
    if hist_sims:
        for sn in hist_sims:
            if 'residual_df' in sims.get(sn, {}):
                tfs.update(sims[sn]['residual_df']['timeframe'].unique())
    return [{'label': f'TF {tf}', 'value': tf} for tf in sorted(tfs)]


@callback(
    Output('obs-graph', 'figure'), Output('obs-info-panel', 'children'),
    Input('obs-plot-type', 'value'), Input('obs-metric-dropdown', 'value'),
    Input('obs-metric-dropdown-2', 'value'),
    Input('obs-log-y', 'value'),
    Input('obs-avg-sim-dropdown', 'value'), Input('obs-avg-refpoint', 'value'),
    Input('obs-res-sim-dropdown', 'value'),
    Input('obs-res-groupby', 'value'), Input('obs-res-mode', 'value'),
    Input('obs-res-refpoint', 'value'),
    Input('obs-hist-sim-dropdown', 'value'), Input('obs-hist-data', 'value'),
    Input('obs-hist-groupby', 'value'), Input('obs-hist-refpoint', 'value'),
    Input('obs-hist-timeframes', 'value'),
    Input('obs-hist-sim-diff', 'value'), Input('obs-hist-diff-sim1', 'value'),
    Input('obs-hist-diff-sim2', 'value'), Input('obs-hist-fit-gauss', 'value'),
    Input('obs-hist-bins', 'value'),
    Input('obs-summ-sim-dropdown', 'value'), Input('obs-summ-data', 'value'),
    Input('obs-summ-level', 'value'), Input('obs-summ-metric', 'value'),
    Input('obs-summ-refpoint', 'value'),
    Input('obs-dataset-store', 'data'))
def update_obs(pt, metric, metric2, logy, avg_sims, avg_refpoint,
               res_sims, groupby, res_mode, res_refpoint,
               hist_sims, hist_data, hist_groupby, hist_refpoint, hist_timeframes,
               hist_sim_diff, hist_diff_s1, hist_diff_s2, hist_fit, hist_bins,
               summ_sims, summ_data, summ_level, summ_metric, summ_refpoint,
               ds_choice):
    sims, names = get_active_data(ds_choice)

    if pt == 'avg_sim':
        return _obs_avg_sim(sims, names, metric, metric2, logy, avg_sims, avg_refpoint)
    elif pt == 'avg_file':
        return _obs_avg_file(sims, names, metric, metric2, logy, avg_sims, avg_refpoint)
    elif pt == 'residual_explorer':
        return _obs_residual_explorer(sims, res_sims, groupby, res_mode, res_refpoint, weighted=False)
    elif pt == 'weighted_residual_explorer':
        return _obs_residual_explorer(sims, res_sims, groupby, res_mode, res_refpoint, weighted=True)
    elif pt == 'residual_histogram':
        return _obs_residual_histogram(sims, hist_sims, hist_data, hist_groupby, hist_refpoint, hist_timeframes,
                                       hist_sim_diff, hist_diff_s1, hist_diff_s2, hist_fit, hist_bins, weighted=False)
    elif pt == 'weighted_residual_histogram':
        return _obs_residual_histogram(sims, hist_sims, hist_data, hist_groupby, hist_refpoint, hist_timeframes,
                                       hist_sim_diff, hist_diff_s1, hist_diff_s2, hist_fit, hist_bins, weighted=True)
    elif pt == 'residual_summary':
        return _obs_residual_summary(sims, summ_sims, summ_data, summ_level, summ_metric, summ_refpoint, weighted=False)
    elif pt == 'weighted_residual_summary':
        return _obs_residual_summary(sims, summ_sims, summ_data, summ_level, summ_metric, summ_refpoint, weighted=True)
    return go.Figure(), ""


def _obs_avg_sim(sims, names, metric, metric2, logy, avg_sims, avg_refpoint):
    sel = [n for n in (avg_sims or names) if n in sims]
    if not sel:
        return go.Figure(), "No simulations selected"
    wa = compute_weight_averages(sims, sel, avg_refpoint if avg_refpoint else None)
    xl = list(wa.keys()); xp = list(range(len(xl)))
    yt = 'log' if 'log' in logy else 'linear'
    has_m2 = metric2 and metric2 != 'none'

    if has_m2:
        fig = make_subplots(rows=1, cols=2, subplot_titles=[metric, metric2], horizontal_spacing=0.12)
        for ci, m in enumerate([metric, metric2]):
            vals = [wa[s].get(m, 0) for s in xl]
            fig.add_trace(go.Scatter(x=xp, y=vals, mode='lines+markers+text',
                line=dict(color='steelblue' if ci == 0 else 'coral', width=2), marker=dict(size=10),
                text=[f'{v:.4f}' if abs(v) < 1000 else f'{v:.2e}' for v in vals],
                textposition='top center', textfont=dict(size=TEXT_FONT_SIZE), cliponaxis=False,
                showlegend=False), row=1, col=ci+1)
            fig.update_xaxes(tickvals=xp, ticktext=xl, tickangle=45, row=1, col=ci+1)
            fig.update_yaxes(title_text=m, type=yt, row=1, col=ci+1)
        fig.update_layout(title=dict(text='Avg per Sim', x=0.5, font=dict(size=TITLE_FONT_SIZE)),
            margin=dict(b=150, t=80), font=dict(size=TICK_FONT_SIZE))
    else:
        vals = [wa[s].get(metric, 0) for s in xl]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xp, y=vals, mode='lines+markers+text', name=metric,
            line=dict(color='steelblue', width=2), marker=dict(size=10),
            text=[f'{v:.4f}' if abs(v) < 1000 else f'{v:.2e}' for v in vals],
            textposition='top center', textfont=dict(size=TEXT_FONT_SIZE), cliponaxis=False))
        fig.update_layout(title=dict(text=f'Avg {metric} per Sim', x=0.5, font=dict(size=TITLE_FONT_SIZE)),
            xaxis=dict(tickvals=xp, ticktext=xl, tickangle=45),
            yaxis=dict(title=metric, type=yt, exponentformat='e'),
            margin=dict(b=150), font=dict(size=TICK_FONT_SIZE))
    fig.update_annotations(font_size=SUBPLOT_TITLE_FONT_SIZE)
    info = [html.B("Values:"), html.Br()]
    for s in xl:
        parts = [f"{metric}={wa[s].get(metric, 0):.6f}"]
        if has_m2: parts.append(f"{metric2}={wa[s].get(metric2, 0):.6f}")
        info += [f"{s}: {', '.join(parts)}", html.Br()]
    return fig, info


def _obs_avg_file(sims, names, metric, metric2, logy, avg_sims, avg_refpoint):
    sel = [n for n in (avg_sims or names) if n in sims]
    if not sel:
        return go.Figure(), "No simulations selected"
    wapf = compute_weight_averages_per_file(sims, sel, avg_refpoint if avg_refpoint else None)
    cmap = get_color_map(sel)
    yt = 'log' if 'log' in logy else 'linear'
    has_m2 = metric2 and metric2 != 'none'

    if has_m2:
        fig = make_subplots(rows=1, cols=2, subplot_titles=[metric, metric2], horizontal_spacing=0.12)
        for ci, m in enumerate([metric, metric2]):
            for sn in wapf:
                rids = list(wapf[sn].keys())
                vals = [wapf[sn][r].get(m, 0) for r in rids]
                fig.add_trace(go.Scatter(x=rids, y=vals, mode='lines+markers', name=sn,
                    line=dict(color=cmap.get(sn, 'gray')), marker=dict(size=6),
                    legendgroup=sn, showlegend=(ci == 0)), row=1, col=ci+1)
            fig.update_xaxes(title_text='Ref Point ID', tickangle=45, row=1, col=ci+1)
            fig.update_yaxes(title_text=m, type=yt, row=1, col=ci+1)
        fig.update_layout(title=dict(text='Avg per File', x=0.5, font=dict(size=TITLE_FONT_SIZE)),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=LEGEND_FONT_SIZE)),
            margin=dict(b=150, r=200, t=80), font=dict(size=TICK_FONT_SIZE))
    else:
        fig = go.Figure()
        for sn in wapf:
            rids = list(wapf[sn].keys())
            vals = [wapf[sn][r].get(metric, 0) for r in rids]
            fig.add_trace(go.Scatter(x=rids, y=vals, mode='lines+markers', name=sn,
                line=dict(color=cmap.get(sn, 'gray')), marker=dict(size=6)))
        fig.update_layout(title=dict(text=f'Avg {metric} per File', x=0.5, font=dict(size=TITLE_FONT_SIZE)),
            xaxis=dict(title='Ref Point ID', tickangle=45),
            yaxis=dict(title=metric, type=yt, exponentformat='e'),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=LEGEND_FONT_SIZE)),
            margin=dict(b=150, r=200), font=dict(size=TICK_FONT_SIZE))
    fig.update_annotations(font_size=SUBPLOT_TITLE_FONT_SIZE)
    return fig, [html.B(f"Avg per file across sims")]


def _obs_residual_explorer(sims, sel_sims, groupby, mode, refpoint, weighted=False):
    """General-purpose residual explorer using the prebuilt residual_df."""
    if not sel_sims:
        return go.Figure(), "Select at least one simulation"

    valid = [s for s in sel_sims if 'residual_df' in sims.get(s, {})]
    if not valid:
        return go.Figure(), "No residual data available for selected simulations"

    # Column name prefixes depending on weighted or not
    if weighted:
        ra_init_col = 'ra_weighted_res_initial'
        dec_init_col = 'dec_weighted_res_initial'
        ra_final_col = 'ra_weighted_res_final'
        dec_final_col = 'dec_weighted_res_final'
        unit = 'σ'
        ra_label = 'RA Weighted Residual [σ]'
        dec_label = 'DEC Weighted Residual [σ]'
        title_prefix = 'Weighted Residual'
    else:
        ra_init_col = 'ra_residual_initial_mas'
        dec_init_col = 'dec_residual_initial_mas'
        ra_final_col = 'ra_residual_final_mas'
        dec_final_col = 'dec_residual_final_mas'
        unit = 'mas'
        ra_label = 'RA Residual [mas]'
        dec_label = 'DEC Residual [mas]'
        title_prefix = 'Residual'

    fig = make_subplots(rows=2, cols=1,
        subplot_titles=[f'RA {title_prefix}s', f'DEC {title_prefix}s'],
        shared_xaxes=True, vertical_spacing=0.10)

    info = [html.B(f"{title_prefix} Statistics:"), html.Br(), html.Br()]
    trace_idx = 0  # for legend dedup

    for sim_idx, sn in enumerate(valid):
        df = sims[sn]['residual_df'].copy()

        # Filter by ref_point_id if requested (multi-select: empty = all)
        if refpoint and len(refpoint) > 0:
            df = df[df['ref_point_id'].isin(refpoint)]
        if len(df) == 0:
            continue

        n_iters = df.attrs.get('n_iterations', '?')

        # Determine grouping
        if groupby == 'none':
            groups = [('all', df)]
        elif groupby == 'ref_point_id':
            groups = [(rid, df[df['ref_point_id'] == rid]) for rid in sorted(df['ref_point_id'].unique())]
        elif groupby == 'timeframe':
            groups = [(f'TF {tf}', df[df['timeframe'] == tf]) for tf in sorted(df['timeframe'].unique())]
        else:
            groups = [('all', df)]

        for grp_name, grp_df in groups:
            if len(grp_df) == 0:
                continue

            # Label for legend
            if groupby == 'none':
                legend_base = sn
            else:
                legend_base = f"{sn} | {grp_name}"

            # Pick a color
            color_idx = trace_idx
            col = color_palette[color_idx % len(color_palette)]
            col_light = color_palette[(color_idx + len(color_palette) // 3) % len(color_palette)]

            datetimes = grp_df['datetime']
            ra_init = grp_df[ra_init_col].values
            dec_init = grp_df[dec_init_col].values
            ra_final = grp_df[ra_final_col].values
            dec_final = grp_df[dec_final_col].values

            if mode == 'overlay':
                # Initial
                fig.add_trace(go.Scattergl(x=datetimes, y=ra_init, mode='markers',
                    name=f'{legend_base} Init', marker=dict(size=5, color=col_light, opacity=0.5),
                    legendgroup=f'{sn}_{grp_name}'), row=1, col=1)
                # Final
                fig.add_trace(go.Scattergl(x=datetimes, y=ra_final, mode='markers',
                    name=f'{legend_base} Final', marker=dict(size=5, color=col, opacity=0.8),
                    legendgroup=f'{sn}_{grp_name}'), row=1, col=1)
                # DEC
                fig.add_trace(go.Scattergl(x=datetimes, y=dec_init, mode='markers',
                    marker=dict(size=5, color=col_light, opacity=0.5),
                    legendgroup=f'{sn}_{grp_name}', showlegend=False), row=2, col=1)
                fig.add_trace(go.Scattergl(x=datetimes, y=dec_final, mode='markers',
                    marker=dict(size=5, color=col, opacity=0.8),
                    legendgroup=f'{sn}_{grp_name}', showlegend=False), row=2, col=1)

                rms_ra_i = np.sqrt(np.mean(ra_init**2))
                rms_ra_f = np.sqrt(np.mean(ra_final**2))
                rms_dec_i = np.sqrt(np.mean(dec_init**2))
                rms_dec_f = np.sqrt(np.mean(dec_final**2))
                ra_improv = (rms_ra_i - rms_ra_f) / rms_ra_i * 100 if rms_ra_i != 0 else 0
                dec_improv = (rms_dec_i - rms_dec_f) / rms_dec_i * 100 if rms_dec_i != 0 else 0
                info += [html.B(f"{legend_base} ({len(grp_df)} obs, {n_iters} iters):"), html.Br(),
                    f"  Init RMS — RA: {rms_ra_i:.2f} {unit}, DEC: {rms_dec_i:.2f} {unit}", html.Br(),
                    f"  Final RMS — RA: {rms_ra_f:.2f} {unit}, DEC: {rms_dec_f:.2f} {unit}", html.Br(),
                    f"  Improvement — RA: {ra_improv:.1f}%, DEC: {dec_improv:.1f}%", html.Br(), html.Br()]

            elif mode == 'final_only':
                fig.add_trace(go.Scattergl(x=datetimes, y=ra_final, mode='markers',
                    name=legend_base, marker=dict(size=5, color=col, opacity=0.8),
                    legendgroup=f'{sn}_{grp_name}'), row=1, col=1)
                fig.add_trace(go.Scattergl(x=datetimes, y=dec_final, mode='markers',
                    marker=dict(size=5, color=col, opacity=0.8),
                    legendgroup=f'{sn}_{grp_name}', showlegend=False), row=2, col=1)

                rms_ra = np.sqrt(np.mean(ra_final**2))
                rms_dec = np.sqrt(np.mean(dec_final**2))
                info += [html.B(f"{legend_base} ({len(grp_df)} obs):"), html.Br(),
                    f"  Final RMS — RA: {rms_ra:.2f} {unit}, DEC: {rms_dec:.2f} {unit}", html.Br(), html.Br()]

            else:  # difference or abs_difference
                dra = ra_final - ra_init
                ddc = dec_final - dec_init
                if mode == 'abs_difference':
                    dra = np.abs(dra); ddc = np.abs(ddc)
                fig.add_trace(go.Scattergl(x=datetimes, y=dra, mode='markers',
                    name=legend_base, marker=dict(size=5, color=col),
                    legendgroup=f'{sn}_{grp_name}'), row=1, col=1)
                fig.add_trace(go.Scattergl(x=datetimes, y=ddc, mode='markers',
                    marker=dict(size=5, color=col),
                    legendgroup=f'{sn}_{grp_name}', showlegend=False), row=2, col=1)

                info += [html.B(f"{legend_base} ({len(grp_df)} obs):"), html.Br(),
                    f"  RA Diff — Mean: {np.mean(dra):.2f}, Std: {np.std(dra):.2f} {unit}", html.Br(),
                    f"  DEC Diff — Mean: {np.mean(ddc):.2f}, Std: {np.std(ddc):.2f} {unit}", html.Br(), html.Br()]

            trace_idx += 1

    # Zero lines
    if mode not in ('abs_difference',):
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

    # σ reference lines for weighted residuals in overlay/final mode
    if weighted and mode in ('overlay', 'final_only'):
        for sigma in [1, 2, 3]:
            for r in [1, 2]:
                fig.add_hline(y=sigma, line_dash="dot", line_color="red", opacity=0.3, row=r, col=1)
                fig.add_hline(y=-sigma, line_dash="dot", line_color="red", opacity=0.3, row=r, col=1)

    fig.update_yaxes(title_text=ra_label, title_font_size=AXIS_TITLE_FONT_SIZE, row=1, col=1)
    fig.update_yaxes(title_text=dec_label, title_font_size=AXIS_TITLE_FONT_SIZE, row=2, col=1)
    fig.update_xaxes(title_text="DateTime", title_font_size=AXIS_TITLE_FONT_SIZE, row=2, col=1)

    mode_titles = {
        'overlay': f'{title_prefix} Overlay: Initial vs Final',
        'final_only': f'{title_prefix}: Final Iteration',
        'difference': f'{title_prefix} Diff: Final − Initial',
        'abs_difference': f'{title_prefix} |Final − Initial|',
    }
    fig.update_layout(
        title=dict(text=mode_titles.get(mode, title_prefix), x=0.5, font=dict(size=TITLE_FONT_SIZE)),
        hovermode='closest', height=900, font=dict(size=TICK_FONT_SIZE),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=LEGEND_FONT_SIZE)),
        margin=dict(r=250))
    fig.update_annotations(font_size=SUBPLOT_TITLE_FONT_SIZE)

    return fig, info


def _obs_residual_histogram(sims, sel_sims, hist_data, groupby, refpoint, timeframes,
                             sim_diff_mode, diff_s1, diff_s2, fit_gauss, n_bins,
                             weighted=False):
    """Histogram of residuals with optional Gaussian fit and Shapiro-Wilk normality test."""
    n_bins = int(n_bins) if n_bins else 50
    do_fit = 'fit' in fit_gauss if fit_gauss else False
    do_sim_diff = sim_diff_mode == 'on'
    # Normalize refpoint: empty list or None means all files
    if not refpoint:
        refpoint = []
    # Normalize timeframes: empty list or None means all
    if not timeframes:
        timeframes = []

    # Column setup
    if weighted:
        cols = {'initial': ('ra_weighted_res_initial', 'dec_weighted_res_initial'),
                'final': ('ra_weighted_res_final', 'dec_weighted_res_final')}
        unit = 'σ'
        title_prefix = 'Weighted Residual'
    else:
        cols = {'initial': ('ra_residual_initial_mas', 'dec_residual_initial_mas'),
                'final': ('ra_residual_final_mas', 'dec_residual_final_mas')}
        unit = 'mas'
        title_prefix = 'Residual'

    # --- Sim diff mode: histogram of (Sim1 - Sim2) residuals ---
    if do_sim_diff:
        for s in (diff_s1, diff_s2):
            if 'residual_df' not in sims.get(s, {}):
                return go.Figure(), f"No residual data for {s}"
        df1 = sims[diff_s1]['residual_df'].copy()
        df2 = sims[diff_s2]['residual_df'].copy()
        if refpoint:
            df1 = df1[df1['ref_point_id'].isin(refpoint)]
            df2 = df2[df2['ref_point_id'].isin(refpoint)]
        if timeframes:
            df1 = df1[df1['timeframe'].isin(timeframes)]
            df2 = df2[df2['timeframe'].isin(timeframes)]
        # Align on global_obs_index
        merged = pd.merge(df1, df2, on=['ref_point_id', 'obs_index', 'timeframe'], suffixes=('_1', '_2'))
        if len(merged) == 0:
            return go.Figure(), "No matching observations between simulations"

        ra_col1, dec_col1 = cols.get(hist_data, cols['final'])
        ra_col2, dec_col2 = ra_col1, dec_col1
        ra_data = merged[f'{ra_col1}_1'].values - merged[f'{ra_col2}_2'].values
        dec_data = merged[f'{dec_col1}_1'].values - merged[f'{dec_col2}_2'].values

        label = f'{diff_s1} − {diff_s2}'
        data_label = hist_data.capitalize()
        all_series = [('all', label, ra_data, dec_data)]

        if groupby == 'ref_point_id':
            all_series = []
            for rid in sorted(merged['ref_point_id'].unique()):
                m = merged[merged['ref_point_id'] == rid]
                ra_d = m[f'{ra_col1}_1'].values - m[f'{ra_col2}_2'].values
                dec_d = m[f'{dec_col1}_1'].values - m[f'{dec_col2}_2'].values
                all_series.append((rid, f'{label} | {rid}', ra_d, dec_d))
        elif groupby == 'timeframe':
            all_series = []
            for tf in sorted(merged['timeframe_1'].unique() if 'timeframe_1' in merged.columns else merged['timeframe'].unique()):
                tf_col = 'timeframe_1' if 'timeframe_1' in merged.columns else 'timeframe'
                m = merged[merged[tf_col] == tf]
                ra_d = m[f'{ra_col1}_1'].values - m[f'{ra_col2}_2'].values
                dec_d = m[f'{dec_col1}_1'].values - m[f'{dec_col2}_2'].values
                all_series.append((f'TF {tf}', f'{label} | TF {tf}', ra_d, dec_d))

    # --- Normal mode: histogram per selected sim, optionally grouped ---
    else:
        if not sel_sims:
            return go.Figure(), "Select at least one simulation"
        valid = [s for s in sel_sims if 'residual_df' in sims.get(s, {})]
        if not valid:
            return go.Figure(), "No residual data"

        ra_col, dec_col = cols.get(hist_data, cols['final'])
        data_label = hist_data.capitalize()

        all_series = []  # list of (group_key, legend_label, ra_array, dec_array)
        for sn in valid:
            df = sims[sn]['residual_df'].copy()
            if refpoint:
                df = df[df['ref_point_id'].isin(refpoint)]
            if timeframes:
                df = df[df['timeframe'].isin(timeframes)]
            if len(df) == 0:
                continue

            if hist_data == 'diff_iter':
                ra_init_c, dec_init_c = cols['initial']
                ra_final_c, dec_final_c = cols['final']
                ra_vals = df[ra_final_c].values - df[ra_init_c].values
                dec_vals = df[dec_final_c].values - df[dec_init_c].values
            else:
                ra_vals = df[ra_col].values
                dec_vals = df[dec_col].values

            if groupby == 'none':
                all_series.append(('all', sn, ra_vals, dec_vals))
            elif groupby == 'ref_point_id':
                for rid in sorted(df['ref_point_id'].unique()):
                    sub = df[df['ref_point_id'] == rid]
                    if hist_data == 'diff_iter':
                        rv = sub[ra_final_c].values - sub[ra_init_c].values
                        dv = sub[dec_final_c].values - sub[dec_init_c].values
                    else:
                        rv = sub[ra_col].values
                        dv = sub[dec_col].values
                    all_series.append((rid, f'{sn} | {rid}', rv, dv))
            elif groupby == 'timeframe':
                for tf in sorted(df['timeframe'].unique()):
                    sub = df[df['timeframe'] == tf]
                    if hist_data == 'diff_iter':
                        rv = sub[ra_final_c].values - sub[ra_init_c].values
                        dv = sub[dec_final_c].values - sub[dec_init_c].values
                    else:
                        rv = sub[ra_col].values
                        dv = sub[dec_col].values
                    all_series.append((f'TF {tf}', f'{sn} | TF {tf}', rv, dv))

    if not all_series:
        return go.Figure(), "No data to plot"

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=[f'RA {title_prefix} [{unit}]', f'DEC {title_prefix} [{unit}]'],
        horizontal_spacing=0.10)

    info = [html.B(f"{title_prefix} Histogram — {data_label}:"), html.Br(), html.Br()]
    trace_idx = 0

    for grp_key, legend_label, ra_arr, dec_arr in all_series:
        col = color_palette[trace_idx % len(color_palette)]

        for ci, (arr, comp) in enumerate([(ra_arr, 'RA'), (dec_arr, 'DEC')]):
            col_idx = ci + 1
            show_leg = (ci == 0)

            fig.add_trace(go.Histogram(
                x=arr, nbinsx=n_bins, name=legend_label,
                marker_color=col, opacity=0.6,
                legendgroup=f'{legend_label}', showlegend=show_leg,
            ), row=1, col=col_idx)

            # Gaussian fit overlay
            mu, std = np.mean(arr), np.std(arr)
            n_pts = len(arr)

            if do_fit and n_pts >= 8:
                # Shapiro-Wilk test (subsample if > 5000 for performance)
                test_data = arr if n_pts <= 5000 else np.random.choice(arr, 5000, replace=False)
                try:
                    sw_stat, sw_p = sp_stats.shapiro(test_data)
                except Exception:
                    sw_stat, sw_p = 0, 0

                # Also compute D'Agostino-Pearson for larger samples
                try:
                    dp_stat, dp_p = sp_stats.normaltest(arr)
                except Exception:
                    dp_stat, dp_p = 0, 0

                # Use the more conservative p-value
                p_val = min(sw_p, dp_p) if n_pts > 20 else sw_p
                is_gaussian = p_val > 0.05

                # Generate fitted curve
                x_fit = np.linspace(arr.min(), arr.max(), 200)
                # Scale PDF to match histogram area (n * bin_width)
                bin_width = (arr.max() - arr.min()) / n_bins if n_bins > 0 else 1
                y_fit = sp_stats.norm.pdf(x_fit, mu, std) * n_pts * bin_width

                gauss_label = "Gaussian" if is_gaussian else "NOT Gaussian"
                p_str = f'{p_val:.2e}' if p_val < 0.01 else f'{p_val:.3f}'

                fig.add_trace(go.Scatter(
                    x=x_fit, y=y_fit, mode='lines',
                    name=f'{legend_label} fit',
                    line=dict(color=col, width=2, dash='dash'),
                    legendgroup=f'{legend_label}', showlegend=False,
                ), row=1, col=col_idx)

                # Add annotation with normality result
                fig.add_annotation(
                    x=mu + 2*std, y=max(y_fit) * 0.9,
                    text=f"<b>{gauss_label}</b><br>μ={mu:.2f}, σ={std:.2f}<br>p={p_str} (n={n_pts})",
                    showarrow=False, font=dict(size=11, color='green' if is_gaussian else 'red'),
                    bgcolor='rgba(255,255,255,0.8)', bordercolor=col,
                    row=1, col=col_idx,
                    xshift=0, yshift=-30*trace_idx,  # offset for multiple series
                )

                info += [html.B(f"{legend_label} — {comp}:"), html.Br(),
                    f"  μ={mu:.4f}, σ={std:.4f} {unit}, n={n_pts}", html.Br(),
                    f"  Shapiro p={sw_p:.4f}, D'Agostino p={dp_p:.4f} → ",
                    html.Span(gauss_label, style={'color': 'green' if is_gaussian else 'red', 'fontWeight': 'bold'}),
                    html.Br()]
            else:
                info += [html.B(f"{legend_label} — {comp}:"), html.Br(),
                    f"  μ={mu:.4f}, σ={std:.4f} {unit}, n={n_pts}", html.Br()]

        info.append(html.Br())
        trace_idx += 1

    fig.update_xaxes(title_text=f'RA [{unit}]', title_font_size=AXIS_TITLE_FONT_SIZE, row=1, col=1)
    fig.update_xaxes(title_text=f'DEC [{unit}]', title_font_size=AXIS_TITLE_FONT_SIZE, row=1, col=2)
    fig.update_yaxes(title_text='Count', title_font_size=AXIS_TITLE_FONT_SIZE, row=1, col=1)
    fig.update_yaxes(title_text='Count', title_font_size=AXIS_TITLE_FONT_SIZE, row=1, col=2)

    title_data_label = {'final': 'Final', 'initial': 'Initial', 'diff_iter': 'Final−Initial'}
    if do_sim_diff:
        title_text = f'{title_prefix} Histogram: {diff_s1} − {diff_s2} ({title_data_label.get(hist_data, "")})'
    else:
        title_text = f'{title_prefix} Histogram ({title_data_label.get(hist_data, "")})'

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=TITLE_FONT_SIZE)),
        barmode='overlay', height=700, font=dict(size=TICK_FONT_SIZE),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=LEGEND_FONT_SIZE)),
        margin=dict(r=250))
    fig.update_annotations(font_size=SUBPLOT_TITLE_FONT_SIZE)

    return fig, info


def _obs_residual_summary(sims, sel_sims, data_source, level, metric_type, refpoint, weighted=False):
    """
    Summary bar/line plots of residual statistics.
    Levels:
      - global: one value per simulation (mean/std/rms across all obs)
      - per_file: one value per ref_point_id per simulation
      - per_timeframe: one value per timeframe per file per simulation
    """
    if not sel_sims:
        return go.Figure(), "Select at least one simulation"

    valid = [s for s in sel_sims if 'residual_df' in sims.get(s, {})]
    if not valid:
        return go.Figure(), "No residual data"

    if not refpoint:
        refpoint = []

    # Column setup
    if weighted:
        cols = {'initial': ('ra_weighted_res_initial', 'dec_weighted_res_initial'),
                'final': ('ra_weighted_res_final', 'dec_weighted_res_final')}
        unit = 'σ'
        title_prefix = 'Weighted Residual'
    else:
        cols = {'initial': ('ra_residual_initial_mas', 'dec_residual_initial_mas'),
                'final': ('ra_residual_final_mas', 'dec_residual_final_mas')}
        unit = 'mas'
        title_prefix = 'Residual'

    def _get_arrays(df, src):
        ra_init_c, dec_init_c = cols['initial']
        ra_final_c, dec_final_c = cols['final']
        if src == 'initial':
            return df[ra_init_c].values, df[dec_init_c].values
        elif src == 'final':
            return df[ra_final_c].values, df[dec_final_c].values
        else:  # diff_iter — return final arrays (we handle the diff at stat level)
            return df[ra_final_c].values, df[dec_final_c].values

    def _calc(arr):
        return {'mean': np.mean(arr), 'std': np.std(arr),
                'rms': np.sqrt(np.mean(arr**2)), 'n': len(arr)}

    def _calc_diff(df):
        """Compute stat(final) - stat(initial) for each metric."""
        ra_init_c, dec_init_c = cols['initial']
        ra_final_c, dec_final_c = cols['final']
        ra_f = _calc(df[ra_final_c].values)
        ra_i = _calc(df[ra_init_c].values)
        dec_f = _calc(df[dec_final_c].values)
        dec_i = _calc(df[dec_init_c].values)
        ra_diff = {k: ra_f[k] - ra_i[k] for k in ('mean', 'std', 'rms')}
        ra_diff['n'] = ra_f['n']
        dec_diff = {k: dec_f[k] - dec_i[k] for k in ('mean', 'std', 'rms')}
        dec_diff['n'] = dec_f['n']
        return ra_diff, dec_diff

    def _calc_special(df, dsrc, metric_type):
        """Compute special metrics that don't follow the standard _calc pattern.
        Returns (ra_val, dec_val) or a single value for scalar metrics."""
        if metric_type == 'uncertainty':
            ra_unc = df['uncertainty_ra_mas'].mean() if 'uncertainty_ra_mas' in df.columns else 0
            dec_unc = df['uncertainty_dec_mas'].mean() if 'uncertainty_dec_mas' in df.columns else 0
            return ra_unc, dec_unc
        elif metric_type == 'n_timeframes':
            return len(df['timeframe'].unique()), len(df['timeframe'].unique())
        elif metric_type == 'nobs_per_tf':
            n_tf = max(1, len(df['timeframe'].unique()))
            return len(df) / n_tf, len(df) / n_tf
        elif metric_type == 'std_over_unc':
            src = 'final' if dsrc != 'initial' else 'initial'
            ra, dec = _get_arrays(df, src)
            ra_std, dec_std = np.std(ra), np.std(dec)
            ra_unc = df['uncertainty_ra_mas'].mean() if 'uncertainty_ra_mas' in df.columns else np.nan
            dec_unc = df['uncertainty_dec_mas'].mean() if 'uncertainty_dec_mas' in df.columns else np.nan
            return (ra_std / ra_unc if ra_unc > 0 else np.nan,
                    dec_std / dec_unc if dec_unc > 0 else np.nan)
        return 0, 0

    # Check if this is a "special" metric that doesn't use the standard _calc
    special_metrics = {'uncertainty', 'n_timeframes', 'nobs_per_tf', 'std_over_unc'}
    is_special = metric_type in special_metrics
    # For scalar metrics (same value for RA and DEC), use single column
    scalar_metrics = {'n_timeframes', 'nobs_per_tf'}
    is_scalar = metric_type in scalar_metrics

    unit_label = {'uncertainty': 'mas', 'n_timeframes': '', 'nobs_per_tf': '',
                  'std_over_unc': 'ratio'}.get(metric_type, unit)

    data_label = {'final': 'Final', 'initial': 'Initial', 'diff_iter': 'Final−Initial',
                  'compare_init_final': 'Init vs Final'}.get(data_source, data_source)
    compare_mode = data_source == 'compare_init_final'
    info = [html.B(f"{title_prefix} Summary — {data_label} — {level}:"), html.Br(), html.Br()]

    # For compare mode, we duplicate each sim into two series (initial + final)
    if compare_mode:
        expanded_sims = []
        for sn in valid:
            expanded_sims.append((sn, f'{sn} (Init)', 'initial'))
            expanded_sims.append((sn, f'{sn} (Final)', 'final'))
    else:
        expanded_sims = [(sn, sn, data_source) for sn in valid]

    # =====================================================================
    # GLOBAL level: one bar per simulation
    # =====================================================================
    if level == 'global':
        sim_labels, ra_stats, dec_stats, ra_special, dec_special = [], [], [], [], []
        colors = []
        for cfg_idx, (sn, label, dsrc) in enumerate(expanded_sims):
            df = sims[sn]['residual_df'].copy()
            if refpoint:
                df = df[df['ref_point_id'].isin(refpoint)]
            if len(df) == 0:
                continue
            if is_special:
                rv, dv = _calc_special(df, dsrc, metric_type)
                ra_special.append(rv); dec_special.append(dv)
                # Still need n for n_obs
                ra_stats.append({'n': len(df)}); dec_stats.append({'n': len(df)})
            elif dsrc == 'diff_iter':
                rs, ds = _calc_diff(df)
                ra_stats.append(rs); dec_stats.append(ds)
            else:
                ra, dec = _get_arrays(df, dsrc)
                ra_stats.append(_calc(ra)); dec_stats.append(_calc(dec))
            sim_labels.append(label)
            colors.append(color_palette[cfg_idx % len(color_palette)])

        if not sim_labels:
            return go.Figure(), "No data"

        if metric_type == 'n_obs':
            fig = go.Figure()
            nobs = [s['n'] for s in ra_stats]
            fig.add_trace(go.Bar(x=sim_labels, y=nobs, marker_color=colors,
                text=[str(n) for n in nobs], textposition='outside', textfont=dict(size=TEXT_FONT_SIZE), cliponaxis=False))
            fig.update_layout(title=dict(text=f'{title_prefix}: N_obs Global ({data_label})', x=0.5,
                font=dict(size=TITLE_FONT_SIZE)), height=500, margin=dict(b=150), font=dict(size=TICK_FONT_SIZE),
                xaxis=dict(tickangle=45), yaxis=dict(title='N_obs'))
            for sn, rs in zip(sim_labels, ra_stats):
                info += [f"{sn}: n={rs['n']}", html.Br()]
            return fig, info

        if is_special:
            xp = list(range(len(sim_labels)))
            if is_scalar:
                fig = go.Figure()
                vals = ra_special  # same for RA/DEC
                fig.add_trace(go.Bar(x=sim_labels, y=vals, marker_color=colors,
                    text=[f'{v:.2f}' for v in vals], textposition='outside',
                    textfont=dict(size=TEXT_FONT_SIZE), cliponaxis=False))
                fig.update_layout(title=dict(text=f'{title_prefix}: {metric_type} ({data_label})', x=0.5,
                    font=dict(size=TITLE_FONT_SIZE)), height=500, margin=dict(b=150), font=dict(size=TICK_FONT_SIZE),
                    xaxis=dict(tickangle=45), yaxis=dict(title=f'{metric_type} [{unit_label}]' if unit_label else metric_type))
            else:
                fig = make_subplots(rows=1, cols=2,
                    subplot_titles=[f'RA {metric_type} [{unit_label}]', f'DEC {metric_type} [{unit_label}]'],
                    horizontal_spacing=0.12)
                for ci, vals in enumerate([ra_special, dec_special]):
                    fig.add_trace(go.Bar(x=sim_labels, y=vals, marker_color=colors,
                        text=[f'{v:.4f}' for v in vals], textposition='outside',
                        textfont=dict(size=TEXT_FONT_SIZE), cliponaxis=False, showlegend=False), row=1, col=ci+1)
                    fig.update_yaxes(title_text=f'{metric_type} [{unit_label}]', row=1, col=ci+1)
                fig.update_layout(title=dict(text=f'{title_prefix}: {metric_type} ({data_label})', x=0.5,
                    font=dict(size=TITLE_FONT_SIZE)), height=500, margin=dict(b=150), font=dict(size=TICK_FONT_SIZE))
                fig.update_xaxes(tickangle=45)
            fig.update_annotations(font_size=SUBPLOT_TITLE_FONT_SIZE)
            return fig, info

        fig = make_subplots(rows=1, cols=2,
            subplot_titles=[f'RA [{unit}]', f'DEC [{unit}]'], horizontal_spacing=0.12)
        xp = list(range(len(sim_labels)))

        for ci, (stats_list, comp) in enumerate([(ra_stats, 'RA'), (dec_stats, 'DEC')]):
            col_idx = ci + 1
            if metric_type == 'mean_std':
                means = [s['mean'] for s in stats_list]
                stds = [s['std'] for s in stats_list]
                fig.add_trace(go.Bar(x=xp, y=means, error_y=dict(type='data', array=stds, visible=True),
                    name=comp, marker_color=colors,
                    text=[f'{m:.3f}±{s:.3f}' for m, s in zip(means, stds)],
                    textposition='outside', textfont=dict(size=TEXT_FONT_SIZE-2),
                    showlegend=False), row=1, col=col_idx)
            else:
                vals = [s[metric_type] for s in stats_list]
                fig.add_trace(go.Bar(x=xp, y=vals, name=comp, marker_color=colors,
                    text=[f'{v:.4f}' for v in vals], textposition='outside',
                    textfont=dict(size=TEXT_FONT_SIZE-2), showlegend=False), row=1, col=col_idx)

            fig.update_xaxes(tickvals=xp, ticktext=sim_labels, tickangle=45, row=1, col=col_idx)
            fig.update_yaxes(title_text=f'{metric_type} [{unit}]', title_font_size=AXIS_TITLE_FONT_SIZE, row=1, col=col_idx)

        for sn, rs, ds in zip(sim_labels, ra_stats, dec_stats):
            info += [html.B(f"{sn} (n={rs['n']}):"), html.Br(),
                f"  RA — mean={rs['mean']:.4f}, std={rs['std']:.4f}, rms={rs['rms']:.4f} {unit}", html.Br(),
                f"  DEC — mean={ds['mean']:.4f}, std={ds['std']:.4f}, rms={ds['rms']:.4f} {unit}", html.Br(), html.Br()]

        fig.update_layout(title=dict(text=f'{title_prefix} Summary: Global ({data_label})', x=0.5,
            font=dict(size=TITLE_FONT_SIZE)), height=600, margin=dict(b=150), font=dict(size=TICK_FONT_SIZE))
        fig.update_annotations(font_size=SUBPLOT_TITLE_FONT_SIZE)
        return fig, info

    # =====================================================================
    # PER FILE level: x-axis = ref_point_id, one line/bar per sim
    # =====================================================================
    elif level == 'per_file':
        if metric_type == 'n_obs' or is_scalar:
            fig = go.Figure()
        elif is_special and not is_scalar:
            fig = make_subplots(rows=2, cols=1,
                subplot_titles=[f'RA {metric_type} [{unit_label}]', f'DEC {metric_type} [{unit_label}]'],
                shared_xaxes=True, vertical_spacing=0.12)
        else:
            fig = make_subplots(rows=2, cols=1,
                subplot_titles=[f'RA {metric_type} [{unit}]', f'DEC {metric_type} [{unit}]'],
                shared_xaxes=True, vertical_spacing=0.12)

        for cfg_idx, (sn, label, dsrc) in enumerate(expanded_sims):
            df = sims[sn]['residual_df'].copy()
            if refpoint:
                df = df[df['ref_point_id'].isin(refpoint)]
            if len(df) == 0:
                continue
            file_ids = sorted(df['ref_point_id'].unique())
            col = color_palette[cfg_idx % len(color_palette)]
            show_leg = True

            if metric_type == 'n_obs':
                nobs = [len(df[df['ref_point_id'] == rid]) for rid in file_ids]
                fig.add_trace(go.Bar(x=[str(f) for f in file_ids], y=nobs, name=label,
                    marker_color=col, opacity=0.7,
                    text=[str(n) for n in nobs], textposition='outside',
                    textfont=dict(size=TEXT_FONT_SIZE-2)))
            elif is_special:
                ra_vals, dec_vals = [], []
                for rid in file_ids:
                    sub = df[df['ref_point_id'] == rid]
                    rv, dv = _calc_special(sub, dsrc, metric_type)
                    ra_vals.append(rv); dec_vals.append(dv)
                if is_scalar:
                    fig.add_trace(go.Scatter(x=file_ids, y=ra_vals,
                        mode='lines+markers', name=label, line=dict(color=col), marker=dict(size=6)))
                else:
                    fig.add_trace(go.Scatter(x=file_ids, y=ra_vals,
                        mode='lines+markers', name=label, line=dict(color=col), marker=dict(size=6),
                        legendgroup=label), row=1, col=1)
                    fig.add_trace(go.Scatter(x=file_ids, y=dec_vals,
                        mode='lines+markers', name=label, line=dict(color=col), marker=dict(size=6),
                        legendgroup=label, showlegend=False), row=2, col=1)
            else:
                ra_vals, dec_vals, ra_errs, dec_errs = [], [], [], []
                for rid in file_ids:
                    sub = df[df['ref_point_id'] == rid]
                    if dsrc == 'diff_iter':
                        rs, ds = _calc_diff(sub)
                    else:
                        ra, dec = _get_arrays(sub, dsrc)
                        rs, ds = _calc(ra), _calc(dec)
                    if metric_type == 'mean_std':
                        ra_vals.append(rs['mean']); dec_vals.append(ds['mean'])
                        ra_errs.append(rs['std']); dec_errs.append(ds['std'])
                    else:
                        ra_vals.append(rs[metric_type]); dec_vals.append(ds[metric_type])

                if metric_type == 'mean_std':
                    fig.add_trace(go.Scatter(x=file_ids, y=ra_vals,
                        error_y=dict(type='data', array=ra_errs, visible=True),
                        mode='lines+markers', name=label, line=dict(color=col), marker=dict(size=6),
                        legendgroup=label), row=1, col=1)
                    fig.add_trace(go.Scatter(x=file_ids, y=dec_vals,
                        error_y=dict(type='data', array=dec_errs, visible=True),
                        mode='lines+markers', name=label, line=dict(color=col), marker=dict(size=6),
                        legendgroup=label, showlegend=False), row=2, col=1)
                else:
                    fig.add_trace(go.Scatter(x=file_ids, y=ra_vals,
                        mode='lines+markers', name=label, line=dict(color=col), marker=dict(size=6),
                        legendgroup=label), row=1, col=1)
                    fig.add_trace(go.Scatter(x=file_ids, y=dec_vals,
                        mode='lines+markers', name=label, line=dict(color=col), marker=dict(size=6),
                        legendgroup=label, showlegend=False), row=2, col=1)

            info += [html.B(f"{label}: {len(file_ids)} files"), html.Br()]

        if metric_type == 'n_obs' or is_scalar:
            yl = 'N_obs' if metric_type == 'n_obs' else (f'{metric_type} [{unit_label}]' if unit_label else metric_type)
            fig.update_layout(title=dict(text=f'{title_prefix}: {metric_type} Per File ({data_label})', x=0.5,
                font=dict(size=TITLE_FONT_SIZE)), barmode='group' if metric_type == 'n_obs' else 'relative',
                height=600, margin=dict(b=150),
                font=dict(size=TICK_FONT_SIZE), xaxis=dict(tickangle=45, title='Ref Point ID'),
                yaxis=dict(title=yl),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=LEGEND_FONT_SIZE)))
        else:
            fig.update_xaxes(title_text='Ref Point ID', tickangle=45, title_font_size=AXIS_TITLE_FONT_SIZE, row=2, col=1)
            fig.update_yaxes(title_text=f'RA {metric_type} [{unit}]', title_font_size=AXIS_TITLE_FONT_SIZE, row=1, col=1)
            fig.update_yaxes(title_text=f'DEC {metric_type} [{unit}]', title_font_size=AXIS_TITLE_FONT_SIZE, row=2, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4, row=1, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4, row=2, col=1)
            fig.update_layout(
                title=dict(text=f'{title_prefix} Summary: Per File ({data_label})', x=0.5, font=dict(size=TITLE_FONT_SIZE)),
                hovermode='x unified', height=900, font=dict(size=TICK_FONT_SIZE),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=LEGEND_FONT_SIZE)),
                margin=dict(b=150, r=200))
        fig.update_annotations(font_size=SUBPLOT_TITLE_FONT_SIZE)
        return fig, info

    # =====================================================================
    # PER TIMEFRAME level: x-axis = timeframe, subplots per file, lines per sim
    # =====================================================================
    elif level == 'per_timeframe':
        # Collect all file ids across selected sims
        all_file_ids = set()
        for sn in valid:
            df = sims[sn]['residual_df']
            if refpoint:
                df = df[df['ref_point_id'].isin(refpoint)]
            all_file_ids.update(df['ref_point_id'].unique())
        all_file_ids = sorted(all_file_ids)

        if not all_file_ids:
            return go.Figure(), "No files found"

        n_files = len(all_file_ids)
        u = unit_label if is_special else unit

        if metric_type == 'n_obs' or is_scalar:
            fig = make_subplots(rows=n_files, cols=1,
                subplot_titles=[f'{metric_type} — {fid}' for fid in all_file_ids],
                shared_xaxes=False, vertical_spacing=max(0.02, 0.4 / n_files))
        else:
            fig = make_subplots(rows=n_files, cols=2,
                subplot_titles=[item for fid in all_file_ids for item in [f'RA — {fid}', f'DEC — {fid}']],
                shared_xaxes=False, vertical_spacing=max(0.02, 0.4 / n_files), horizontal_spacing=0.10)

        for cfg_idx, (sn, label, dsrc) in enumerate(expanded_sims):
            df = sims[sn]['residual_df'].copy()
            if refpoint:
                df = df[df['ref_point_id'].isin(refpoint)]
            col = color_palette[cfg_idx % len(color_palette)]

            for file_row, fid in enumerate(all_file_ids):
                sub_file = df[df['ref_point_id'] == fid]
                if len(sub_file) == 0:
                    continue
                tfs = sorted(sub_file['timeframe'].unique())
                row = file_row + 1
                show_leg = (file_row == 0)
                tf_labels = [str(tf) for tf in tfs]

                if metric_type == 'n_obs':
                    nobs = [len(sub_file[sub_file['timeframe'] == tf]) for tf in tfs]
                    fig.add_trace(go.Bar(x=tf_labels, y=nobs, name=label,
                        marker_color=col, opacity=0.7,
                        text=[str(n) for n in nobs], textposition='outside',
                        textfont=dict(size=TEXT_FONT_SIZE-2),
                        legendgroup=label, showlegend=show_leg), row=row, col=1)
                elif is_special:
                    ra_vals, dec_vals = [], []
                    for tf in tfs:
                        sub_tf = sub_file[sub_file['timeframe'] == tf]
                        rv, dv = _calc_special(sub_tf, dsrc, metric_type)
                        ra_vals.append(rv); dec_vals.append(dv)
                    if is_scalar:
                        fig.add_trace(go.Scatter(x=tf_labels, y=ra_vals,
                            mode='lines+markers', name=label, line=dict(color=col), marker=dict(size=5),
                            legendgroup=label, showlegend=show_leg), row=row, col=1)
                    else:
                        fig.add_trace(go.Scatter(x=tf_labels, y=ra_vals,
                            mode='lines+markers', name=label, line=dict(color=col), marker=dict(size=5),
                            legendgroup=label, showlegend=show_leg), row=row, col=1)
                        fig.add_trace(go.Scatter(x=tf_labels, y=dec_vals,
                            mode='lines+markers', name=label, line=dict(color=col), marker=dict(size=5),
                            legendgroup=label, showlegend=False), row=row, col=2)
                else:
                    ra_vals, dec_vals, ra_errs, dec_errs = [], [], [], []
                    for tf in tfs:
                        sub_tf = sub_file[sub_file['timeframe'] == tf]
                        if dsrc == 'diff_iter':
                            rs, ds = _calc_diff(sub_tf)
                        else:
                            ra, dec = _get_arrays(sub_tf, dsrc)
                            rs, ds = _calc(ra), _calc(dec)
                        if metric_type == 'mean_std':
                            ra_vals.append(rs['mean']); dec_vals.append(ds['mean'])
                            ra_errs.append(rs['std']); dec_errs.append(ds['std'])
                        else:
                            ra_vals.append(rs[metric_type]); dec_vals.append(ds[metric_type])

                    if metric_type == 'mean_std':
                        fig.add_trace(go.Scatter(x=tf_labels, y=ra_vals,
                            error_y=dict(type='data', array=ra_errs, visible=True),
                            mode='lines+markers', name=label, line=dict(color=col), marker=dict(size=5),
                            legendgroup=label, showlegend=show_leg), row=row, col=1)
                        fig.add_trace(go.Scatter(x=tf_labels, y=dec_vals,
                            error_y=dict(type='data', array=dec_errs, visible=True),
                            mode='lines+markers', name=label, line=dict(color=col), marker=dict(size=5),
                            legendgroup=label, showlegend=False), row=row, col=2)
                    else:
                        fig.add_trace(go.Scatter(x=tf_labels, y=ra_vals,
                            mode='lines+markers', name=label, line=dict(color=col), marker=dict(size=5),
                            legendgroup=label, showlegend=show_leg), row=row, col=1)
                        fig.add_trace(go.Scatter(x=tf_labels, y=dec_vals,
                            mode='lines+markers', name=label, line=dict(color=col), marker=dict(size=5),
                            legendgroup=label, showlegend=False), row=row, col=2)

        # Axis labels — independent x-axes for each file, y-axis units on every row
        for r in range(1, n_files + 1):
            if metric_type == 'n_obs' or is_scalar:
                fig.update_xaxes(title_text='Timeframe', title_font_size=AXIS_TITLE_FONT_SIZE-2, row=r, col=1)
                fig.update_yaxes(title_text=f'{metric_type}' + (f' [{u}]' if u else ''),
                    title_font_size=AXIS_TITLE_FONT_SIZE-2, row=r, col=1)
            else:
                fig.update_xaxes(title_text='Timeframe', title_font_size=AXIS_TITLE_FONT_SIZE-2, row=r, col=1)
                fig.update_xaxes(title_text='Timeframe', title_font_size=AXIS_TITLE_FONT_SIZE-2, row=r, col=2)
                fig.update_yaxes(title_text=f'RA [{u}]', title_font_size=AXIS_TITLE_FONT_SIZE-2, row=r, col=1)
                fig.update_yaxes(title_text=f'DEC [{u}]', title_font_size=AXIS_TITLE_FONT_SIZE-2, row=r, col=2)
                if metric_type not in special_metrics:
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3, row=r, col=1)
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3, row=r, col=2)

        height = max(600, 250 * n_files)
        fig.update_layout(
            title=dict(text=f'{title_prefix} Summary: Per Timeframe ({data_label})', x=0.5, font=dict(size=TITLE_FONT_SIZE)),
            height=height, font=dict(size=TICK_FONT_SIZE),
            barmode='group' if metric_type == 'n_obs' else 'relative',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=LEGEND_FONT_SIZE)),
            margin=dict(r=200))
        fig.update_annotations(font_size=min(SUBPLOT_TITLE_FONT_SIZE, 14))

        info += [f"{len(all_file_ids)} files × {len(expanded_sims)} series plotted", html.Br()]
        return fig, info

    return go.Figure(), ""


# ============================================================================
# SECTION 9: STATISTICAL COMPARISON TAB
# ============================================================================

def render_stats_tab(simulations, sim_names, dataset_choice):
    # Default: exclude original sims that have synthetic replacements
    source_sims = set(INITIAL_ITER_AS_SIM.keys())
    default_selected = [n for n in sim_names if n not in source_sims]
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Plot Type:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                dcc.RadioItems(id='stats-plot-type', options=[
                    {'label': 'RSW Statistics', 'value': 'rsw_stats'},
                    {'label': 'RMS Comparison', 'value': 'rms_compare'},
                    {'label': 'Pos/Vel Magnitude', 'value': 'pos_vel_mag'},
                    {'label': 'State Components', 'value': 'state_components'},
                    {'label': 'Goodness of Fit', 'value': 'goodness_of_fit'},
                ], value='rsw_stats', inline=False, style={'fontSize': '14px'}),
            ], style={'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.Label("Simulations to include (drag to reorder):", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(id='stats-sim-dropdown',
                        options=[{'label': n, 'value': n} for n in sim_names],
                        value=default_selected, multi=True, placeholder='Select simulations...',
                        style={'width': '600px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                html.Div([
                    dcc.Checklist(id='stats-show-max-formal',
                        options=[{'label': ' Show Max Formal Errors column', 'value': 'show'}],
                        value=['show'], style={'fontSize': '14px'}),
                ], style={'display': 'inline-block', 'marginRight': '20px'}),
                html.Div([
                    dcc.Checklist(id='stats-transpose-rsw',
                        options=[{'label': ' Transpose RSW grid (R/S/W on top)', 'value': 'transpose'}],
                        value=[], style={'fontSize': '14px'}),
                ], style={'display': 'inline-block'}),
            ]),
            html.Div([
                html.Label("Reorder:", style={'fontSize': '14px', 'fontWeight': 'bold', 'marginRight': '10px'}),
                html.Button("⬅ Move Left", id='stats-move-left', n_clicks=0,
                    style={'fontSize': '12px', 'marginRight': '5px'}),
                html.Button("Move Right ➡", id='stats-move-right', n_clicks=0,
                    style={'fontSize': '12px', 'marginRight': '15px'}),
                dcc.Dropdown(id='stats-reorder-target',
                    options=[{'label': n, 'value': n} for n in sim_names],
                    value=None, placeholder='Select sim to move...',
                    style={'width': '250px', 'fontSize': '13px', 'display': 'inline-block', 'verticalAlign': 'middle'}),
            ], style={'marginTop': '10px'}),
        ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginBottom': '20px'}),
        html.Div([
            dcc.RadioItems(id='gof-metric-radio', options=[
                {'label': 'WRMS [mas]', 'value': 'wrms'}, {'label': 'Cost', 'value': 'cost'}],
                value='wrms', inline=True, style={'fontSize': '14px'}),
            html.Div([
                dcc.Checklist(id='gof-show-initial', options=[{'label': ' Initial', 'value': 'show'}], value=['show'],
                    style={'display': 'inline-block', 'marginRight': '20px', 'fontSize': '14px'}),
                dcc.Checklist(id='gof-show-improvement', options=[{'label': ' Improvement %', 'value': 'show'}], value=[],
                    style={'display': 'inline-block', 'marginRight': '20px', 'fontSize': '14px'}),
                dcc.Checklist(id='gof-log-y', options=[{'label': ' Log Y', 'value': 'log'}], value=[],
                    style={'display': 'inline-block', 'fontSize': '14px'}),
            ], style={'marginTop': '10px'}),
        ], id='gof-controls', style={'display': 'none', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginBottom': '20px'}),
        # Hidden placeholders for residual dropdowns (callbacks still reference them)
        html.Div([
            dcc.Dropdown(id='residual-sim-dropdown', value=[], multi=True, style={'display': 'none'}),
            dcc.RadioItems(id='residual-plot-mode', value='overlay', style={'display': 'none'}),
        ], style={'display': 'none'}),
        html.Div([
            dcc.Dropdown(id='weighted-residual-sim-dropdown', value=[], multi=True, style={'display': 'none'}),
            dcc.RadioItems(id='weighted-residual-plot-mode', value='overlay', style={'display': 'none'}),
        ], style={'display': 'none'}),
        dcc.Store(id='stats-dataset-store', data=dataset_choice),
        dcc.Graph(id='stats-graph', style={'height': '900px'},
            config={'toImageButtonOptions': {'format': 'svg', 'filename': 'stats_plot', 'scale': 2}}),
        html.Div(id='stats-info-panel', style={'padding': '20px', 'backgroundColor': '#e9ecef', 'marginTop': '10px', 'fontSize': '14px'})
    ])

@callback(Output('gof-controls', 'style'), Input('stats-plot-type', 'value'))
def toggle_stats_ctrls(pt):
    h = {'display': 'none'}
    s = {'display': 'block', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginBottom': '20px'}
    if pt == 'goodness_of_fit': return s
    return h


@callback(Output('stats-sim-dropdown', 'value'),
    Input('stats-move-left', 'n_clicks'), Input('stats-move-right', 'n_clicks'),
    State('stats-sim-dropdown', 'value'), State('stats-reorder-target', 'value'),
    prevent_initial_call=True)
def reorder_stats_sims(left_clicks, right_clicks, current_order, target):
    if not current_order or not target or target not in current_order:
        return current_order or []
    order = list(current_order)
    idx = order.index(target)
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id'] if ctx.triggered else ''
    if 'move-left' in triggered and idx > 0:
        order[idx], order[idx-1] = order[idx-1], order[idx]
    elif 'move-right' in triggered and idx < len(order) - 1:
        order[idx], order[idx+1] = order[idx+1], order[idx]
    return order


@callback(Output('stats-graph', 'figure'), Output('stats-info-panel', 'children'),
    Input('stats-plot-type', 'value'), Input('gof-metric-radio', 'value'),
    Input('gof-show-initial', 'value'), Input('gof-show-improvement', 'value'),
    Input('gof-log-y', 'value'), Input('residual-sim-dropdown', 'value'),
    Input('residual-plot-mode', 'value'), Input('weighted-residual-sim-dropdown', 'value'),
    Input('weighted-residual-plot-mode', 'value'),
    Input('stats-sim-dropdown', 'value'), Input('stats-show-max-formal', 'value'),
    Input('stats-transpose-rsw', 'value'),
    Input('font-store', 'data'),
    Input('stats-dataset-store', 'data'))
def update_stats(pt, gm, gsi, gimp, gly, rs, rm, wrs, wrm, sel_sims, show_mf, transpose_rsw, fonts, dsc):
    sims, all_names = get_active_data(dsc)
    f = get_fonts(fonts)
    # Use dropdown order directly (preserves user reordering)
    names = [n for n in (sel_sims or []) if n in sims]
    if not names:
        return go.Figure(), "No simulations selected"
    xp = list(range(len(names)))
    show_max_formal = 'show' in (show_mf or [])
    do_transpose = 'transpose' in (transpose_rsw or [])
    if pt == 'rsw_stats': return _rsw_stats(sims, names, xp, show_max_formal, do_transpose, f)
    elif pt == 'rms_compare': return _rms_compare(sims, names, xp, f)
    elif pt == 'pos_vel_mag': return _pos_vel(sims, names, xp, f)
    elif pt == 'state_components': return _state_comp(sims, names, xp, f)
    elif pt == 'goodness_of_fit': return _gof(sims, names, xp, gm, gsi, gimp, gly, f)
    return go.Figure(), ""

def _rsw_stats(sims, names, xp, show_max_formal=True, transpose=False, f=None):
    if f is None: f = get_fonts()
    stats_names = ['Mean','RMS','Max Diff'] + (['Max Formal'] if show_max_formal else [])
    comps = ['R','S','W']
    clr = {'R':'steelblue','S':'coral','W':'green'}

    # Compute data
    pd_ = {c: {'mean':[],'rms':[],'max_diff':[],'max_formal':[]} for c in comps}
    for sn in names:
        if 'diff_SPICE_RSW' in sims[sn]:
            ds = compute_rsw_statistics(sims[sn]['diff_SPICE_RSW'])
            for c in comps: pd_[c]['mean'].append(ds[c]['mean']); pd_[c]['rms'].append(ds[c]['rms']); pd_[c]['max_diff'].append(ds[c]['max'])
        else:
            for c in comps: pd_[c]['mean'].append(0); pd_[c]['rms'].append(0); pd_[c]['max_diff'].append(0)
        if show_max_formal:
            if 'formal_errors_RSW_km' in sims[sn]:
                fs = compute_formal_error_statistics(sims[sn]['formal_errors_RSW_km'])
                for c in comps: pd_[c]['max_formal'].append(fs[c]['max'])
            else:
                for c in comps: pd_[c]['max_formal'].append(0)

    stat_keys = ['mean','rms','max_diff'] + (['max_formal'] if show_max_formal else [])

    if transpose:
        # Transposed: rows = stats (Mean/RMS/Max Diff/Max Formal), cols = R/S/W
        n_rows = len(stats_names)
        n_cols = 3
        titles = [f'{sn} — {c}' if i == 0 else '' for c in comps for i, sn in enumerate(stats_names)]
        # Build proper title list: first row gets component names, first col gets stat names
        subplot_titles = []
        for r, sn in enumerate(stats_names):
            for ci, c in enumerate(comps):
                subplot_titles.append(f'{c}' if r == 0 else '')

        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles,
            vertical_spacing=0.12, horizontal_spacing=0.08,
            row_titles=stats_names)

        for ci, c in enumerate(comps):
            for ri, sk in enumerate(stat_keys):
                v = pd_[c][sk]
                fig.add_trace(go.Scatter(x=xp, y=v, mode='lines+markers+text', line=dict(color=clr[c], width=2),
                    marker=dict(size=8), text=[f'{x:.2f}' for x in v], textposition='top center',
                    textfont=dict(size=f['text']), showlegend=False, cliponaxis=False), row=ri+1, col=ci+1)
        for col in range(1, n_cols+1):
            fig.update_xaxes(tickvals=xp, ticktext=names, tickangle=45, row=n_rows, col=col)
        fig.update_layout(title=dict(text='RSW Statistics (Transposed)', x=0.5, font=dict(size=f['title'])),
            height=max(1100, 280*n_rows), margin=dict(b=200, t=80, l=120), font=dict(size=f['tick']))
    else:
        # Original: rows = R/S/W, cols = Mean/RMS/Max Diff/Max Formal
        n_cols = len(stats_names)
        titles = stats_names + ['']*(3*(n_cols-1))
        fig = make_subplots(rows=3, cols=n_cols, subplot_titles=titles[:3*n_cols],
            vertical_spacing=0.12, horizontal_spacing=0.08)

        cols_map = {sk: i+1 for i, sk in enumerate(stat_keys)}
        rows_map = {'R':1,'S':2,'W':3}
        for c in comps:
            for sk, col in cols_map.items():
                v = pd_[c][sk]
                fig.add_trace(go.Scatter(x=xp, y=v, mode='lines+markers+text', line=dict(color=clr[c], width=2),
                    marker=dict(size=8), text=[f'{x:.2f}' for x in v], textposition='top center',
                    textfont=dict(size=f['text']), showlegend=False, cliponaxis=False), row=rows_map[c], col=col)
        for col in range(1, n_cols+1):
            fig.update_xaxes(tickvals=xp, ticktext=names, tickangle=45, row=3, col=col)
        fig.update_layout(title=dict(text='RSW Statistics', x=0.5, font=dict(size=f['title'])),
            height=1100, margin=dict(b=200, t=80), font=dict(size=f['tick']))

    fig.update_annotations(font_size=f['subplot'])
    return fig, [html.B("RSW diff & formal error stats")]

def _rms_compare(sims, names, xp, f=None):
    if f is None: f = get_fonts()
    vals = [sims[s].get('rms_SPICE', 0) or 0 for s in names]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xp, y=vals, mode='lines+markers+text', line=dict(color='steelblue', width=2),
        marker=dict(size=10), text=[f'{v:.3f}' for v in vals], textposition='top center',
        textfont=dict(size=f['text']), cliponaxis=False))
    fig.update_layout(title=dict(text='RMS Comparison', x=0.5, font=dict(size=f['title'])),
        xaxis=dict(tickvals=xp, ticktext=names, tickangle=45), yaxis=dict(title='RMS [km]'),
        height=600, margin=dict(b=150, t=80), font=dict(size=f['tick']))
    info = [html.B("RMS values [km]:"), html.Br()]
    for sn, v in zip(names, vals):
        info += [f"  {sn}: {v:.3f} km", html.Br()]
    return fig, info

def _pos_vel(sims, names, xp, f=None):
    if f is None: f = get_fonts()
    su = compute_state_updates(sims, names)
    pm = [su[s]['pos_mag'] for s in names]; vm = [su[s]['vel_mag'] for s in names]
    fig = make_subplots(rows=1, cols=2, subplot_titles=['|Δr| [km]', '|Δv| [km/s]'])
    fig.add_trace(go.Scatter(x=xp, y=pm, mode='lines+markers+text', line=dict(color='steelblue', width=2),
        marker=dict(size=10), text=[f'{v:.3f}' for v in pm], textposition='top center',
        textfont=dict(size=f['text'])), row=1, col=1)
    fig.add_trace(go.Scatter(x=xp, y=vm, mode='lines+markers+text', line=dict(color='coral', width=2),
        marker=dict(size=10), text=[f'{v:.6f}' for v in vm], textposition='top center',
        textfont=dict(size=f['text'])), row=1, col=2)
    fig.update_xaxes(tickvals=xp, ticktext=names, tickangle=45)
    fig.update_layout(title=dict(text='Pos/Vel Updates', x=0.5, font=dict(size=f['title'])),
        height=600, margin=dict(b=150, t=80), showlegend=False, font=dict(size=f['tick']))
    fig.update_annotations(font_size=f['subplot'])
    return fig, [html.B("Magnitude updates")]

def _state_comp(sims, names, xp, f=None):
    if f is None: f = get_fonts()
    su = compute_state_updates(sims, names)
    fig = make_subplots(rows=3, cols=2, vertical_spacing=0.12, horizontal_spacing=0.12,
        subplot_titles=['ΔX [km]','ΔVx [km/s]','ΔY [km]','ΔVy [km/s]','ΔZ [km]','ΔVz [km/s]'])
    for row in range(3):
        pv = [su[s]['pos'][row] for s in names]; vv = [su[s]['vel'][row] for s in names]
        fig.add_trace(go.Scatter(x=xp, y=pv, mode='lines+markers+text', line=dict(color='steelblue', width=2),
            marker=dict(size=8), text=[f'{v:.3f}' for v in pv], textposition='top center',
            textfont=dict(size=f['text']), showlegend=False, cliponaxis=False), row=row+1, col=1)
        fig.add_trace(go.Scatter(x=xp, y=vv, mode='lines+markers+text', line=dict(color='coral', width=2),
            marker=dict(size=8), text=[f'{v:.6f}' for v in vv], textposition='top center',
            textfont=dict(size=f['text']), showlegend=False, cliponaxis=False), row=row+1, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=row+1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=row+1, col=2)
    fig.update_xaxes(tickvals=xp, ticktext=names, tickangle=45, row=3, col=1)
    fig.update_xaxes(tickvals=xp, ticktext=names, tickangle=45, row=3, col=2)
    fig.update_layout(title=dict(text='State Component Updates', x=0.5, font=dict(size=f['title'])),
        height=900, margin=dict(b=150, t=80), font=dict(size=f['tick']))
    fig.update_annotations(font_size=f['subplot'])
    return fig, [html.B("State updates")]

def _gof(sims, names, xp, metric_type, show_initial, show_improvement, log_y, f=None):
    if f is None: f = get_fonts()
    wm = compute_wrms_and_cost(sims, names)
    if not wm: return go.Figure(), "No WRMS data"
    fk = 'final_wrms_combined_method1_mas' if metric_type == 'wrms' else 'final_cost_function'
    ik = 'initial_wrms_combined_method1_mas' if metric_type == 'wrms' else 'initial_cost_function'
    yl = 'WRMS [mas]' if metric_type == 'wrms' else 'Cost Function'
    imp_k = 'wrms_improvement_percent' if metric_type == 'wrms' else 'cost_improvement_percent'
    fv, iv, impv, avs = [], [], [], []
    for s in names:
        if s in wm and fk in wm[s]:
            fv.append(wm[s][fk]); iv.append(wm[s][ik]); impv.append(wm[s][imp_k]); avs.append(s)
    if not fv: return go.Figure(), "No data"
    xps = list(range(len(avs)))
    fig = go.Figure()
    fmt = lambda v: f'{v:.2e}' if metric_type == 'cost' else f'{v:.2f}'
    if 'show' in show_initial:
        fig.add_trace(go.Scatter(x=xps, y=iv, mode='lines+markers+text', name='Initial',
            line=dict(color='lightcoral', width=2, dash='dash'), marker=dict(size=10),
            text=[fmt(v) for v in iv], textposition='top center', textfont=dict(size=f['text']), cliponaxis=False))
    fig.add_trace(go.Scatter(x=xps, y=fv, mode='lines+markers+text', name='Final',
        line=dict(color='steelblue', width=2), marker=dict(size=10),
        text=[fmt(v) for v in fv],
        textposition='bottom center' if 'show' in show_initial else 'top center',
        textfont=dict(size=f['text']), cliponaxis=False))
    yt = 'log' if 'log' in log_y else 'linear'
    fig.update_layout(title=dict(text=f'{yl} Comparison', x=0.5, font=dict(size=f['title'])),
        xaxis=dict(tickvals=xps, ticktext=avs, tickangle=45), yaxis=dict(title=yl, type=yt, exponentformat='e'),
        height=600, margin=dict(b=150), font=dict(size=f['tick']),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, font=dict(size=f['legend'])))
    info = [html.B("Values:"), html.Br()]
    for i, s in enumerate(avs):
        info += [f"{s}: Init={fmt(iv[i])}, Final={fmt(fv[i])}, Improv={impv[i]:.2f}%", html.Br()]
    return fig, info


# ============================================================================
# SECTION 10: CORRELATION ANALYSIS TAB
# ============================================================================

def render_correlation_tab(simulations, sim_names, dataset_choice):
    swc = [n for n in sim_names if 'correlations' in simulations.get(n, {})]
    if not swc:
        return html.Div("No correlation data available", style={'padding': '20px', 'fontSize': '16px'})
    return html.Div([
        html.Div([
            dcc.RadioItems(id='corr-plot-type', options=[
                {'label': 'Single Heatmap', 'value': 'single'},
                {'label': 'Compare Two', 'value': 'compare'},
                {'label': 'Difference', 'value': 'diff'},
            ], value='single', inline=False, style={'fontSize': '14px'}),
            html.Div([
                dcc.Dropdown(id='corr-sim-dropdown', options=[{'label': n, 'value': n} for n in swc],
                    value=swc[0], clearable=False, style={'width': '350px', 'fontSize': '13px'}),
            ], style={'marginTop': '15px'}, id='corr-single-controls'),
            html.Div([
                html.Div([
                    html.Label("Sim 1:"), dcc.Dropdown(id='corr-sim1-dropdown',
                        options=[{'label': n, 'value': n} for n in swc], value=swc[0],
                        clearable=False, style={'width': '280px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                html.Div([
                    html.Label("Sim 2:"), dcc.Dropdown(id='corr-sim2-dropdown',
                        options=[{'label': n, 'value': n} for n in swc],
                        value=swc[-1] if len(swc) > 1 else swc[0],
                        clearable=False, style={'width': '280px', 'fontSize': '13px'}),
                ], style={'display': 'inline-block'}),
            ], id='corr-compare-controls', style={'display': 'none', 'marginTop': '15px'}),
            html.Div([
                dcc.Checklist(id='corr-show-values', options=[{'label': ' Show values', 'value': 'show'}],
                    value=[], style={'display': 'inline-block', 'marginRight': '30px', 'fontSize': '14px'}),
                dcc.Checklist(id='corr-abs-diff', options=[{'label': ' Absolute diff', 'value': 'abs'}],
                    value=[], style={'display': 'inline-block', 'fontSize': '14px'}),
            ], style={'marginTop': '15px'}),
        ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginBottom': '20px'}),
        dcc.Store(id='corr-dataset-store', data=dataset_choice),
        dcc.Graph(id='corr-graph', style={'height': '800px'},
            config={'toImageButtonOptions': {'format': 'svg', 'filename': 'corr_plot', 'scale': 2}}),
        html.Div(id='corr-info-panel', style={'padding': '20px', 'backgroundColor': '#e9ecef', 'marginTop': '10px', 'fontSize': '14px'})
    ])

@callback(Output('corr-single-controls', 'style'), Output('corr-compare-controls', 'style'),
          Input('corr-plot-type', 'value'))
def toggle_corr(pt):
    if pt == 'single': return {'marginTop': '15px'}, {'display': 'none', 'marginTop': '15px'}
    return {'display': 'none'}, {'display': 'block', 'marginTop': '15px'}

@callback(Output('corr-graph', 'figure'), Output('corr-info-panel', 'children'),
    Input('corr-plot-type', 'value'), Input('corr-sim-dropdown', 'value'),
    Input('corr-sim1-dropdown', 'value'), Input('corr-sim2-dropdown', 'value'),
    Input('corr-show-values', 'value'), Input('corr-abs-diff', 'value'),
    Input('corr-dataset-store', 'data'))
def update_corr(pt, sn, s1, s2, sv, ad, dsc):
    sims, _ = get_active_data(dsc)
    if pt == 'single': return _corr_single(sims, sn, sv)
    elif pt == 'compare': return _corr_compare(sims, s1, s2, sv)
    elif pt == 'diff': return _corr_diff(sims, s1, s2, sv, ad)
    return go.Figure(), ""

def _corr_single(sims, sn, sv):
    if 'correlations' not in sims.get(sn, {}): return go.Figure(), "No data"
    cm = sims[sn]['correlations']
    labels = get_parameter_labels(sims[sn].get('est_parameters', []))
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=cm, x=labels, y=labels, colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title='Corr')))
    if 'show' in sv:
        anns = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                anns.append(dict(x=labels[j], y=labels[i], text=f'{cm[i,j]:.2f}', showarrow=False,
                    font=dict(size=10, color='black' if abs(cm[i,j]) < 0.5 else 'white')))
        fig.update_layout(annotations=anns)
    fig.update_layout(title=dict(text=f'Correlations: {sn}', x=0.5, font=dict(size=TITLE_FONT_SIZE)),
        xaxis=dict(tickangle=45, side='bottom'), yaxis=dict(autorange='reversed'),
        width=800, height=800, font=dict(size=TICK_FONT_SIZE))
    mx = np.max(np.abs(cm[np.triu_indices_from(cm, k=1)]))
    return fig, [html.B(f"Max off-diag |corr|: {mx:.4f}")]

def _corr_compare(sims, s1, s2, sv):
    if 'correlations' not in sims.get(s1, {}) or 'correlations' not in sims.get(s2, {}):
        return go.Figure(), "Missing data"
    c1, c2 = sims[s1]['correlations'], sims[s2]['correlations']
    l1 = get_parameter_labels(sims[s1].get('est_parameters', []))
    l2 = get_parameter_labels(sims[s2].get('est_parameters', []))
    fig = make_subplots(rows=1, cols=2, subplot_titles=[s1, s2], horizontal_spacing=0.15,
        specs=[[{'type':'heatmap'}, {'type':'heatmap'}]])
    fig.add_trace(go.Heatmap(z=c1, x=l1, y=l1, colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
        colorbar=dict(x=0.45, title='Corr')), row=1, col=1)
    fig.add_trace(go.Heatmap(z=c2, x=l2, y=l2, colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
        colorbar=dict(x=1.02, title='Corr')), row=1, col=2)
    for c in [1, 2]:
        fig.update_xaxes(tickangle=45, side='bottom', row=1, col=c)
        fig.update_yaxes(autorange='reversed', row=1, col=c)
    fig.update_layout(title=dict(text=f'Compare: {s1} vs {s2}', x=0.5, font=dict(size=TITLE_FONT_SIZE)),
        width=1400, height=700, font=dict(size=TICK_FONT_SIZE))
    fig.update_annotations(font_size=SUBPLOT_TITLE_FONT_SIZE)
    d = c1 - c2
    mx = np.max(np.abs(d[np.triu_indices_from(d, k=1)])) if c1.shape == c2.shape else 0
    return fig, [html.B(f"Max |diff|: {mx:.4f}")]

def _corr_diff(sims, s1, s2, sv, ad):
    if 'correlations' not in sims.get(s1, {}) or 'correlations' not in sims.get(s2, {}):
        return go.Figure(), "Missing data"
    c1, c2 = sims[s1]['correlations'], sims[s2]['correlations']
    if c1.shape != c2.shape: return go.Figure(), "Shape mismatch"
    d = c1 - c2
    if 'abs' in ad: d = np.abs(d)
    labels = get_parameter_labels(sims[s1].get('est_parameters', []))
    cs = 'Reds' if 'abs' in ad else 'RdBu'
    zm = None if 'abs' in ad else 0
    zn = 0 if 'abs' in ad else -1
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=d, x=labels, y=labels, colorscale=cs, zmid=zm, zmin=zn, zmax=1))
    if 'show' in sv:
        anns = []
        for i in range(len(d)):
            for j in range(len(d)):
                anns.append(dict(x=labels[j], y=labels[i], text=f'{d[i,j]:.2f}', showarrow=False,
                    font=dict(size=10, color='black' if abs(d[i,j]) < 0.3 else 'white')))
        fig.update_layout(annotations=anns)
    suffix = '|Diff|' if 'abs' in ad else 'Diff'
    fig.update_layout(title=dict(text=f'Correlation {suffix}: {s1} − {s2}', x=0.5, font=dict(size=TITLE_FONT_SIZE)),
        xaxis=dict(tickangle=45, side='bottom'), yaxis=dict(autorange='reversed'),
        width=800, height=800, font=dict(size=TICK_FONT_SIZE))
    od = d[np.triu_indices_from(d, k=1)]
    return fig, [html.B(f"Max |diff|: {np.max(np.abs(od)):.4f}, Mean: {np.mean(np.abs(od)):.4f}")]


# ============================================================================
# SECTION 11: RUN APP
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True, port=8051)