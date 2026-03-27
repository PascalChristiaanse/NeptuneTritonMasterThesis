
import os
import yaml
import json
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from typing import Dict, List, Tuple


# tudatpy imports
from tudatpy import math
from tudatpy import constants

from tudatpy.interface import spice
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.estimation.observable_models_setup import links

import tudatpy.estimation
from tudatpy import util
#import tudatpy.estimation_setup

#from tudatpy.numerical_simulation import estimation

#from tudatpy.numerical_simulation import estimation_setup #,Time


from tudatpy import numerical_simulation

from tudatpy.astro import time_conversion, element_conversion,frame_conversion
from tudatpy.astro.time_conversion import DateTime


from tudatpy.data import save2txt

import sys
from pathlib import Path

# Add parent directory to Python path
#sys.path.append(str(Path(__file__).resolve().parent.parent))


import Analysis_All_Estimations_and_Data as UncertantyPropUtils

# Get the path to the directory containing this file
current_dir = Path(__file__).resolve().parent

# Append the HelperFunctions directory
sys.path.append(str(current_dir / "HelperFunctions"))

import ProcessingUtils
import PropFuncs
import FigUtils
import ObsFunc
import nsdc
import ObservationImplementation
#import RunMultipleEstimations
import MainPostprocessing as PostProc
import EstimationAnalysisTemplates as EstimationTemplates

matplotlib.use("PDF")  #tkagg



# Define temporal scope of the simulation - equal to the time JUICE will spend in orbit around Jupiter
simulation_start_epoch = DateTime(1963, 1,  1).epoch() #2006, 8,  27 1963, 3,  4   1989 1996
simulation_end_epoch   = DateTime(2025, 1, 1).epoch()   #2025, 1, 1    2003  2010

simulation_initial_epoch = DateTime(2006, 10, 1).epoch() #2006, 10, 1
global_frame_origin = 'SSB'
global_frame_orientation = 'ECLIPJ2000'

#--------------------------------------------------------------------------------------------
# ENVIORONMENT SETTINGS 
#--------------------------------------------------------------------------------------------
settings_env = dict()
settings_env["start_epoch"] = simulation_start_epoch
settings_env["end_epoch"] = simulation_end_epoch
settings_env["bodies"] = ['Sun','Jupiter', 'Saturn','Neptune','Triton','Uranus','Mercury','Venus','Mars','Earth'] #
settings_env["global_frame_origin"] = global_frame_origin
settings_env["global_frame_orientation"] = global_frame_orientation
settings_env["interpolator_triton_cadance"] = 60*8
settings_env["neptune_extended_gravity"] = "Jacobson2009"
settings_env['use_created_env'] = False

settings_env['Neptune_rot_model_type'] = 'IAU2015' 
    # Model Type for rotation model of Neptune:
    #  'simple_from_spice' - simple spice,
    #  'spice' - full spice,
    #  'IAU2015' - based on the IAU2015 paper
    #   'Pole_Model_Jacobson2009' - IAU rotation model estimated by Jacobson 2009
    
#--------------------------------------------------------------------------------------------
# ACCELERATION SETTINGS 
#--------------------------------------------------------------------------------------------

settings_acc = dict()
settings_acc['bodies_to_propagate'] = ['Triton']
settings_acc['central_bodies'] = ['Neptune']
settings_acc['bodies_to_simulate'] = ['Sun','Jupiter', 'Saturn','Neptune','Triton','Uranus','Mercury','Venus','Mars','Earth'] 
settings_acc['bodies'] = settings_env["bodies"]

settings_acc['neptune_extended_gravity'] =  "Jacobson2009"


accelerations_cfg = PropFuncs.build_acceleration_config(settings_acc)
settings_acc['accelerations_cfg'] = accelerations_cfg
#--------------------------------------------------------------------------------------------
# PROPAGATOR SETTINGS 
#--------------------------------------------------------------------------------------------

settings_prop = dict()
settings_prop['start_epoch'] = settings_env["start_epoch"]
settings_prop['end_epoch'] = settings_env["end_epoch"]
settings_prop['initial_epoch'] = simulation_initial_epoch
settings_prop['bodies_to_propagate'] = settings_acc['bodies_to_propagate'] 
settings_prop['central_bodies'] = settings_acc['central_bodies']
settings_prop['global_frame_orientation'] = settings_env["global_frame_orientation"]
settings_prop['fixed_step_size'] = 60*60 # 60 minutes

#--------------------------------------------------------------------------------------------
# OBSERVATION SETTINGS 
#--------------------------------------------------------------------------------------------

# --- Load names of data files you wish to include
with open("file_names.json", "r") as f:
    file_names_loaded = json.load(f)

# weights = pd.read_csv(
#         "Results/PoleEstimationRealObservations/LoopTest2/initial_state_only/0/summary.txt", #Results/BetterFigs/AllModernObservations/PostProcessing/First/weights.txt
#         sep="\t",
#         index_col="id")

settings_obs = dict()
settings_obs["mode"] = ["pos"]
settings_obs["bodies"] = [("Triton", "Neptune")]                           # bodies to observe
settings_obs["cadence"] = 60*60*3 # Every 3 hours
settings_obs["type"] = "Real" # Simulated or Real observations

#TEST FILE CHANGE
# file_names_loaded = [
#         'Triton_286_nm0090.csv',]


settings_obs["files"] = file_names_loaded             
settings_obs["observations_folder_path"] = "Observations/AllModernECLIPJ2000"  #RelativeObservations AllModernECLIPJ2000 AllModernJ2000

# weights = weights.reset_index()

settings_obs["use_weights"] = True
# settings_obs["ra_dec_independent_weights"] = False
# settings_obs["timeframe_weights"] = False
# settings_obs["weights"] = weights

settings_obs["use_loaded_obs"] = False

settings_obs["residual_filtering"] = True
settings_obs["epoch_filter_dict"] = None 


#Make sure all other weight types are off
# settings_obs['std_weights'] = False
# settings_obs["per_night_weights"] = False
# settings_obs["per_night_weights_id"] = False 
# settings_obs['per_night_weights_hybrid'] = False


# settings_obs['use_old_obs_func'] = False



#--------------------------------------------------------------------------------------------
# ESTIMATION SETTINGS 
#--------------------------------------------------------------------------------------------

settings_est = dict()
#settings_est['pseudo_observations_settings'] = pseudo_observations_settings
#settings_est['pseudo_observations'] = pseudo_observations

settings_est['est_parameters'] = ['initial_state'] #,'iau_rotation_model_pole','iau_rotation_model_pole_rate'] 
    #Possible settings: 
    # initial state - default
    #GM_Neptune - gravitational parameter Neptune
    #GM_Triton - gravitational parameter Triton
    # iau_rotation_model_pole - rotation pole position (alpha,delta) with IAU rotation model
    # iau_rotation_model_pole_rate - rotation pole rate  (alpha_dot, delta_dot) with IAU rotation model
    # iau_rotation_model_pole_librations - 1st order libration terms  
    # Spherical Harmonics Neptune (C20,C40) - extended body gravity of Neptune C20,C40 (J2,J4)
    
    #This is the proper order keep in mind !!!

    # Rotation_Pole_Position_Neptune - fixed rotation pole position (only with simple rotational model !)

settings_est['a_priori_covariance'] = False

#fill in settings 
settings = dict()
settings["env"] = settings_env
settings["acc"] = settings_acc
settings["prop"] = settings_prop
settings["obs"] = settings_obs
settings["est"] = settings_est



def make_timestamped_folder(base_path="Results"):
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = Path(base_path) / folder_name
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path


##############################################################################################
# LOAD SPICE KERNELS
##############################################################################################

from pathlib import Path

# Path to the current script
current_dir = Path(__file__).resolve().parent

# Kernel folder 
kernel_folder = "Kernels" #current_dir.parent / 

#kernel_folder = "/Kernels/"
kernel_paths=[
    "pck00010.tpc",
    "gm_de440.tpc",
    "nep097.bsp",     
    #"nep105.bsp",
    "naif0012.tls"
    ]

spice.load_standard_kernels()

# Load your kernels
for k in kernel_paths:
    spice.load_kernel(os.path.join(kernel_folder, k))



settings['env']['Neptune_rot_model_type'] = 'IAU2015'

# Common settings — per-variant overrides applied in the loop below
settings['est']['a_priori_pole'] = False

# ---- Create environment and load observations ----
body_settings, system_of_bodies = PropFuncs.Create_Env(settings['env'])

observations, observations_settings, observation_set_ids, epochs_rejected = ObsFunc.LoadObservations(
        settings["obs"]["observations_folder_path"],
        system_of_bodies,
        settings['obs']["files"],
        Residual_filtering=settings["obs"]["residual_filtering"])



##############################################################################################
# INVESTIGATION: residuals per file — SPICE (nep097) vs tudatpy, accepted vs rejected
##############################################################################################

from tudatpy.estimation import observations_setup as tud_obs_setup
from matplotlib.backends.backend_pdf import PdfPages

arcsec_to_rad = np.pi / (180.0 * 3600.0)
j2000_epoch   = dt.datetime(2000, 1, 1, 12)

# ---- Load all files UNFILTERED ----
obs_unfiltered, obs_settings_all, set_ids_all, _ = ObsFunc.LoadObservations(
    settings["obs"]["observations_folder_path"],
    system_of_bodies,
    file_names_loaded,
    Residual_filtering=False,
    epoch_filter_dict=None
)

# ---- Load all files FILTERED — only to get rejected epochs ----
_, _, _, epochs_rej_filtered = ObsFunc.LoadObservations(
    settings["obs"]["observations_folder_path"],
    system_of_bodies,
    file_names_loaded,
    Residual_filtering=True,
    epoch_filter_dict=None
)

# ---- Compute tudatpy residuals on the unfiltered collection ----
obs_simulators = tud_obs_setup.observations_simulation_settings.create_observation_simulators(
    obs_settings_all, system_of_bodies
)
tudatpy.estimation.observations.compute_residuals_and_dependent_variables(
    obs_unfiltered, obs_simulators, system_of_bodies
)

# ---- Per-set times and flat interleaved residuals [ra0,dec0,ra1,dec1,...] in rad ----
obs_times_per_set = obs_unfiltered.get_observation_times()
res_concat        = np.array(obs_unfiltered.get_concatenated_residuals())

obs_counts  = [len(obs_times_per_set[j]) for j in range(len(set_ids_all))]
set_offsets = np.cumsum([0] + obs_counts)

# ---- SPICE (nep097) residuals ----
print("\nComputing SPICE (nep097) residuals...")
ra_spice_flat, dec_spice_flat = ObsFunc.Get_SPICE_residual_from_observations(
    obs_unfiltered, set_ids_all, system_of_bodies,
    global_frame_orientation=global_frame_orientation
)

##############################################################################################
# PLOT: one page per file — RA + Dec, accepted (circles) vs rejected (x)
##############################################################################################

with PdfPages("observation_residuals_all_files.pdf") as pdf:
    for j, set_id in enumerate(set_ids_all):
        s, e = set_offsets[j], set_offsets[j + 1]

        times_j     = np.array(obs_times_per_set[j])
        ra_spice_j  = ra_spice_flat[s:e]               # arcsec
        dec_spice_j = dec_spice_flat[s:e]              # arcsec
        ra_tud_j    = res_concat[2*s:2*e:2]   / arcsec_to_rad   # rad → arcsec
        dec_tud_j   = res_concat[2*s+1:2*e:2] / arcsec_to_rad

        times_dt_j  = [j2000_epoch + dt.timedelta(seconds=float(t)) for t in times_j]

        # Rejected mask
        rejected_set = set(epochs_rej_filtered.get(set_id, []))
        mask_rej = np.array([float(t) in rejected_set for t in times_j])
        mask_acc = ~mask_rej

        acc_dt = [times_dt_j[i] for i in np.where(mask_acc)[0]]
        rej_dt = [times_dt_j[i] for i in np.where(mask_rej)[0]]

        fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig.suptitle(f"{set_id}  |  n={e-s}  rejected={mask_rej.sum()}", fontsize=10)

        # --- RA ---
        ax_ra.scatter(acc_dt, ra_tud_j[mask_acc],   s=6,  color='C0', alpha=0.8, label='tudatpy (accepted)')
        ax_ra.scatter(acc_dt, ra_spice_j[mask_acc],  s=6,  color='C1', alpha=0.8, label='SPICE nep097 (accepted)')
        if mask_rej.any():
            ax_ra.scatter(rej_dt, ra_tud_j[mask_rej],   s=30, color='C0', marker='x', zorder=5, label='tudatpy (rejected)')
            ax_ra.scatter(rej_dt, ra_spice_j[mask_rej],  s=30, color='C1', marker='x', zorder=5, label='SPICE nep097 (rejected)')
        ax_ra.axhline(0, color='k', lw=0.5, ls='--')
        ax_ra.set_ylabel('RA residual [arcsec]')
        ax_ra.legend(fontsize=7, markerscale=2, ncol=2)

        # --- Dec ---
        ax_dec.scatter(acc_dt, dec_tud_j[mask_acc],  s=6,  color='C0', alpha=0.8, label='tudatpy (accepted)')
        ax_dec.scatter(acc_dt, dec_spice_j[mask_acc], s=6,  color='C1', alpha=0.8, label='SPICE nep097 (accepted)')
        if mask_rej.any():
            ax_dec.scatter(rej_dt, dec_tud_j[mask_rej],  s=30, color='C0', marker='x', zorder=5, label='tudatpy (rejected)')
            ax_dec.scatter(rej_dt, dec_spice_j[mask_rej], s=30, color='C1', marker='x', zorder=5, label='SPICE nep097 (rejected)')
        ax_dec.axhline(0, color='k', lw=0.5, ls='--')
        ax_dec.set_ylabel('Dec residual [arcsec]')
        ax_dec.set_xlabel('Date')
        ax_dec.legend(fontsize=7, markerscale=2, ncol=2)

        ax_dec.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        fig.autofmt_xdate()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print("Saved: observation_residuals_all_files.pdf")

##############################################################################################
# COMBINED PLOT: all files from file_names.json — accepted (blue) vs rejected (red)
##############################################################################################

all_times_dt  = []
all_ra_spice  = []
all_dec_spice = []
all_mask_rej  = []

for j, set_id in enumerate(set_ids_all):
    s, e = set_offsets[j], set_offsets[j + 1]
    times_j      = np.array(obs_times_per_set[j])
    rejected_set = set(epochs_rej_filtered.get(set_id, []))
    mask_rej_j   = np.array([float(t) in rejected_set for t in times_j])
    all_times_dt.extend([j2000_epoch + dt.timedelta(seconds=float(t)) for t in times_j])
    all_ra_spice.extend(ra_spice_flat[s:e])
    all_dec_spice.extend(dec_spice_flat[s:e])
    all_mask_rej.extend(mask_rej_j)

all_times_dt  = np.array(all_times_dt)
all_ra_spice  = np.array(all_ra_spice)
all_dec_spice = np.array(all_dec_spice)
all_mask_rej  = np.array(all_mask_rej, dtype=bool)
all_mask_acc  = ~all_mask_rej

fig_all, (ax_ra_all, ax_dec_all) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig_all.suptitle(
    f"All files in file_names.json  |  n={len(all_times_dt)}  rejected={all_mask_rej.sum()}",
    fontsize=11
)

ax_ra_all.scatter(all_times_dt[all_mask_acc], all_ra_spice[all_mask_acc],
                  s=4, color='C0', alpha=0.6, label=f'accepted  (n={all_mask_acc.sum()})')
ax_ra_all.scatter(all_times_dt[all_mask_rej], all_ra_spice[all_mask_rej],
                  s=14, color='red', alpha=0.9, zorder=5, label=f'rejected  (n={all_mask_rej.sum()})')
ax_ra_all.axhline(0, color='k', lw=0.5, ls='--')
ax_ra_all.set_ylabel('RA residual [arcsec]  (SPICE nep097)')
ax_ra_all.legend(fontsize=8, markerscale=2)

ax_dec_all.scatter(all_times_dt[all_mask_acc], all_dec_spice[all_mask_acc],
                   s=4, color='C0', alpha=0.6, label=f'accepted  (n={all_mask_acc.sum()})')
ax_dec_all.scatter(all_times_dt[all_mask_rej], all_dec_spice[all_mask_rej],
                   s=14, color='red', alpha=0.9, zorder=5, label=f'rejected  (n={all_mask_rej.sum()})')
ax_dec_all.axhline(0, color='k', lw=0.5, ls='--')
ax_dec_all.set_ylabel('Dec residual [arcsec]  (SPICE nep097)')
ax_dec_all.set_xlabel('Date')
ax_dec_all.legend(fontsize=8, markerscale=2)

ax_dec_all.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig_all.autofmt_xdate()
fig_all.tight_layout()
fig_all.savefig("observation_residuals_combined.pdf")
plt.close(fig_all)
print("Saved: observation_residuals_combined.pdf")

##############################################################################################
# EXCLUDED FILES: in folder but NOT in file_names.json
##############################################################################################

folder_path      = settings["obs"]["observations_folder_path"]
all_folder_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
excluded_files   = [f for f in all_folder_files if f not in file_names_loaded]
print(f"\nFiles in folder but not in file_names.json: {excluded_files}")

if excluded_files:
    obs_excl_unfilt, obs_settings_excl, set_ids_excl, _ = ObsFunc.LoadObservations(
        folder_path, system_of_bodies, excluded_files,
        Residual_filtering=False, epoch_filter_dict=None
    )
    _, _, _, epochs_rej_excl = ObsFunc.LoadObservations(
        folder_path, system_of_bodies, excluded_files,
        Residual_filtering=True, epoch_filter_dict=None
    )

    obs_sim_excl = tud_obs_setup.observations_simulation_settings.create_observation_simulators(
        obs_settings_excl, system_of_bodies
    )
    tudatpy.estimation.observations.compute_residuals_and_dependent_variables(
        obs_excl_unfilt, obs_sim_excl, system_of_bodies
    )

    obs_times_excl  = obs_excl_unfilt.get_observation_times()
    counts_excl     = [len(obs_times_excl[j]) for j in range(len(set_ids_excl))]
    offsets_excl    = np.cumsum([0] + counts_excl)

    print("\nComputing SPICE (nep097) residuals for excluded files...")
    ra_spice_excl, dec_spice_excl = ObsFunc.Get_SPICE_residual_from_observations(
        obs_excl_unfilt, set_ids_excl, system_of_bodies,
        global_frame_orientation=global_frame_orientation
    )

    all_times_excl_dt = []
    all_ra_excl       = []
    all_dec_excl      = []
    all_rej_excl      = []

    for j, set_id in enumerate(set_ids_excl):
        s, e = offsets_excl[j], offsets_excl[j + 1]
        times_j      = np.array(obs_times_excl[j])
        rejected_set = set(epochs_rej_excl.get(set_id, []))
        mask_rej_j   = np.array([float(t) in rejected_set for t in times_j])
        all_times_excl_dt.extend([j2000_epoch + dt.timedelta(seconds=float(t)) for t in times_j])
        all_ra_excl.extend(ra_spice_excl[s:e])
        all_dec_excl.extend(dec_spice_excl[s:e])
        all_rej_excl.extend(mask_rej_j)

    all_times_excl_dt = np.array(all_times_excl_dt)
    all_ra_excl       = np.array(all_ra_excl)
    all_dec_excl      = np.array(all_dec_excl)
    all_rej_excl      = np.array(all_rej_excl, dtype=bool)
    all_acc_excl      = ~all_rej_excl

    fig_excl, (ax_ra_excl, ax_dec_excl) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig_excl.suptitle(
        f"Files NOT in file_names.json: {', '.join(excluded_files)}\n"
        f"n={len(all_times_excl_dt)}  rejected={all_rej_excl.sum()}",
        fontsize=9
    )

    ax_ra_excl.scatter(all_times_excl_dt[all_acc_excl], all_ra_excl[all_acc_excl],
                       s=6, color='C0', alpha=0.8, label=f'accepted  (n={all_acc_excl.sum()})')
    ax_ra_excl.scatter(all_times_excl_dt[all_rej_excl], all_ra_excl[all_rej_excl],
                       s=20, color='red', alpha=0.9, zorder=5, label=f'rejected  (n={all_rej_excl.sum()})')
    ax_ra_excl.axhline(0, color='k', lw=0.5, ls='--')
    ax_ra_excl.set_ylabel('RA residual [arcsec]  (SPICE nep097)')
    ax_ra_excl.legend(fontsize=8, markerscale=2)

    ax_dec_excl.scatter(all_times_excl_dt[all_acc_excl], all_dec_excl[all_acc_excl],
                        s=6, color='C0', alpha=0.8, label=f'accepted  (n={all_acc_excl.sum()})')
    ax_dec_excl.scatter(all_times_excl_dt[all_rej_excl], all_dec_excl[all_rej_excl],
                        s=20, color='red', alpha=0.9, zorder=5, label=f'rejected  (n={all_rej_excl.sum()})')
    ax_dec_excl.axhline(0, color='k', lw=0.5, ls='--')
    ax_dec_excl.set_ylabel('Dec residual [arcsec]  (SPICE nep097)')
    ax_dec_excl.set_xlabel('Date')
    ax_dec_excl.legend(fontsize=8, markerscale=2)

    ax_dec_excl.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig_excl.autofmt_xdate()
    fig_excl.tight_layout()
    fig_excl.savefig("observation_residuals_excluded_files.pdf")
    plt.close(fig_excl)
    print("Saved: observation_residuals_excluded_files.pdf")
