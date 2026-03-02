
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

weights = pd.read_csv(
        "Results/PoleEstimationRealObservations/LoopTest2/initial_state_only/0/summary.txt", #Results/BetterFigs/AllModernObservations/PostProcessing/First/weights.txt
        sep="\t",
        index_col="id")

settings_obs = dict()
settings_obs["mode"] = ["pos"]
settings_obs["bodies"] = [("Triton", "Neptune")]                           # bodies to observe
settings_obs["cadence"] = 60*60*3 # Every 3 hours
settings_obs["type"] = "Real" # Simulated or Real observations

#TEST FILE CHANGE
# file_names_loaded = [
#         #'Triton_874_nm0013.csv',
#         'Triton_689_nm0007.csv',
#         'Triton_337_nm0085.csv',
#         'Triton_337_nm0015.csv',
#         'Triton_337_nm0019.csv']


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


# ObservationImplementation.main(settings,make_timestamped_folder("Results/EstimatedParametersSimulatedObservations/Test"))
    
print("#######################################################################################################")
print("LOAD KERNELS")
print("#######################################################################################################")
#Load kernels
kernel_folder = "Kernels/"
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

    

out_dir = make_timestamped_folder("Results/EstimationTemplatesTest")

#VARIANTS = EstimationTemplates.CASE1_Manual_Bias(settings,out_dir)


VARIANTS = EstimationTemplates.WeightSchemeAnalysis(settings,out_dir,runSim=False)


  # Load simulations
simulations = {
    name: PostProc.load_npy_files(cfg["simulation_path"])
    for name, cfg in VARIANTS.items()
}




# for name, cfg in VARIANTS.items():
#     simulations[name]["est_parameters"] = cfg["est_parameters"]




runSimulationsBetter = False
if runSimulationsBetter == True:
    # Define all variants for CASE 1
    VARIANTS = {
        # #Pole IAU2015
        # #---------------------------------------------------------------------------------------------------------------------------------------
        "initial_state_only": {
            "simulation_path": "Results/PoleEstimationRealObservations/FullDuration/initial_state_only",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': ['IAU'],
        }, 
        "initial_state_only_outliers": {
            "simulation_path": "Results/PoleEstimationRealObservations/NewWeightsFull/initial_state_only",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        }, 
        "hybrid_new": {
            "simulation_path": "Results/PoleEstimationRealObservations/NewWeightsFull/hybrid_new",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        }, 
        "Timeframe_new": {
            "simulation_path": "Results/PoleEstimationRealObservations/NewWeightsFull/Timeframe_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "ID_Weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/NewWeightsFull/ID_Weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        }, 
        "Hybrid_old_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/NewWeightsFull/Hybrid_old_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        }, 
        "Old_Obs_Func": {
            "simulation_path": "Results/PoleEstimationRealObservations/NewWeightsFull/Old_Obs_Func",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "Hybrid_old_no_loop": {
            "simulation_path": "Results/PoleEstimationRealObservations/NewWeightsFull/Hybrid_old_weights_no_loop",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        }, 

                                                
    }

    # Weight Test CASE 2 (small set)
    VARIANTS = {
        "Initial_State_No_Weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE2/Initial_State_No_Weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "initial_state_hybrid_old_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE2/initial_state_hybrid_old_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "initial_state_hybrid_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE2/initial_state_hybrid_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "initial_state_id_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE2/initial_state_id_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "initial_state_tf_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE2/initial_state_tf_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
    }

    # Weight Test CASE 1 (full set)
    VARIANTS = {
        "Initial_State_No_Weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE1/Initial_State_No_Weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "initial_state_hybrid_old_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE1/initial_state_hybrid_old_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "initial_state_hybrid_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE1/initial_state_hybrid_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "initial_state_id_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE1/initial_state_id_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
        "initial_state_tf_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/WeightTest_CASE1/initial_state_tf_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009'
        },
    }


    # Weight Estimation Loop (iterative weight refinement)
    VARIANTS = {
        "Initial_State_No_Weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/EstimationWeightLoop/Initial_State_No_Weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'IAU2015'
        },
        "initial_state_hybrid_weights_0": {
            "simulation_path": "Results/PoleEstimationRealObservations/EstimationWeightLoop/initial_state_hybrid_weights_0",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'IAU2015'
        },
        "initial_state_hybrid_weights_1": {
            "simulation_path": "Results/PoleEstimationRealObservations/EstimationWeightLoop/initial_state_hybrid_weights_1",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'IAU2015'
        },
        "initial_state_hybrid_weights_2": {
            "simulation_path": "Results/PoleEstimationRealObservations/EstimationWeightLoop/initial_state_hybrid_weights_2",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'IAU2015'
        },
        "initial_state_hybrid_weights_3": {
            "simulation_path": "Results/PoleEstimationRealObservations/EstimationWeightLoop/initial_state_hybrid_weights_3",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'IAU2015'
        },
        "initial_state_hybrid_weights_4": {
            "simulation_path": "Results/PoleEstimationRealObservations/EstimationWeightLoop/initial_state_hybrid_weights_4",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'IAU2015'
        },
    }
    

    # Pole pos and Pole lib estimation (no weights, no cov), (hybrid weights and cov)
    VARIANTS = {
        "initial_state_hybrid_weights_4": {
            "simulation_path": "Results/PoleEstimationRealObservations/EstimationWeightLoop/initial_state_hybrid_weights_4",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "no_weights_no_cov_pole_pos": {
            "simulation_path": "Results/PoleEstimationRealObservations/PoleInitCASE2/no_weights_no_cov_pole_pos",
            'est_parameters': ['initial_state','iau_rotation_model_pole'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "no_weights_no_cov_pole_lib": {
            "simulation_path": "Results/PoleEstimationRealObservations/PoleInitCASE2/no_weights_no_cov_pole_lib",
            'est_parameters': ['initial_state','iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "no_weights_no_cov_pole_pos_lib": {
            "simulation_path": "Results/PoleEstimationRealObservations/PoleInitCASE2/no_weights_no_cov_pole_pos_lib",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "hybrid_weights_pole_pos_cov": {
            "simulation_path": "Results/PoleEstimationRealObservations/PoleInitCASE2/hybrid_weights_pole_pos_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'pole_pos_cov': True 
        },
        "hybrid_weights_pole_lib_cov": {
            "simulation_path": "Results/PoleEstimationRealObservations/PoleInitCASE2/hybrid_weights_pole_lib_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'pole_pos_cov': False, 
            'pole_lib_cov': True,
        },
        "hybrid_weights_pole_pos_cov_pole_lib": {
            "simulation_path": "Results/PoleEstimationRealObservations/PoleInitCASE2/hybrid_weights_pole_pos_cov_pole_lib",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'pole_pos_cov': True, 
            'pole_lib_cov': False,
        },
        "hybrid_weights_pole_pos_pole_lib_cov": {
            "simulation_path": "Results/PoleEstimationRealObservations/PoleInitCASE2/hybrid_weights_pole_pos_pole_lib_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'pole_pos_cov': False, 
            'pole_lib_cov': True,
        },
        "hybrid_weights_pole_pos_cov_pole_lib_cov": {
            "simulation_path": "Results/PoleEstimationRealObservations/PoleInitCASE2/hybrid_weights_pole_pos_cov_pole_lib_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'a_priori_pole': True, 
            'pole_lib_cov': True,
        },
    }

    
    # Pole pos and Pole lib estimation hybrid weights, initial state from simulated obs
    # VARIANTS = {
    #     "no_weights_initial_state": {
    #         "simulation_path": "Results/PoleEstimationRealObservations/PoleSimPoleCASE2/no_weights_initial_state",
    #         'est_parameters': ['initial_state'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "hybrid_weights_initial_state": {
    #         "simulation_path": "Results/PoleEstimationRealObservations/PoleSimPoleCASE2/hybrid_weights_initial_state",
    #         'est_parameters': ['initial_state'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': True,
    #         'use_apriori_cov': False,
    #     },
    #     "hybrid_weights_pole_pos_cov": {
    #         "simulation_path": "Results/PoleEstimationRealObservations/PoleSimPoleCASE2/hybrid_weights_pole_pos_cov",
    #         'est_parameters': ['initial_state','iau_rotation_model_pole'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': True,
    #         'use_apriori_cov': True,
    #         'pole_pos_cov': True 
    #     },
    #     "hybrid_weights_pole_lib_cov": {
    #         "simulation_path": "Results/PoleEstimationRealObservations/PoleSimPoleCASE2/hybrid_weights_pole_lib_cov",
    #         'est_parameters': ['initial_state','iau_rotation_model_pole_librations'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': True,
    #         'use_apriori_cov': True,
    #         'pole_pos_cov': False, 
    #         'pole_lib_cov': True,
    #     },
    #     "hybrid_weights_pole_pos_cov_pole_lib": {
    #         "simulation_path": "Results/PoleEstimationRealObservations/PoleSimPoleCASE2/hybrid_weights_pole_pos_cov_pole_lib",
    #         'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_librations'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': True,
    #         'use_apriori_cov': True,
    #         'pole_pos_cov': True, 
    #         'pole_lib_cov': False,
    #     },
    #     "hybrid_weights_pole_pos_pole_lib_cov": {
    #         "simulation_path": "Results/PoleEstimationRealObservations/PoleSimPoleCASE2/hybrid_weights_pole_pos_pole_lib_cov",
    #         'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_librations'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': True,
    #         'use_apriori_cov': True,
    #         'pole_pos_cov': False, 
    #         'pole_lib_cov': True,
    #     },
    #     "hybrid_weights_pole_pos_cov_pole_lib_cov": {
    #         "simulation_path": "Results/PoleEstimationRealObservations/PoleSimPoleCASE2/hybrid_weights_pole_pos_cov_pole_lib_cov",
    #         'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_librations'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': True,
    #         'use_apriori_cov': True,
    #         'a_priori_pole': True, 
    #         'pole_lib_cov': True,
    #     },
    # }

    
    # ULTIMATE ANALYSIS
    VARIANTS = {
        "IAUPole_initial_state": {
            "simulation_path": "Results/PoleEstimationRealObservations/EstimationWeightLoop/initial_state_hybrid_weights_4",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': False,
        },
        "IAUPole_pole_pos_cov": {
            "simulation_path": "Results/PoleEstimationRealObservations/PoleInitCASE2/hybrid_weights_pole_pos_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'pole_pos_cov': True 
        },
        "IAUPole_pole_lib_cov": {
            "simulation_path": "Results/PoleEstimationRealObservations/PoleInitCASE2/hybrid_weights_pole_lib_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'pole_pos_cov': False, 
            'pole_lib_cov': True,
        },
        "IAUPole_pole_pos_cov_pole_lib_cov": {
            "simulation_path": "Results/PoleEstimationRealObservations/PoleInitCASE2/hybrid_weights_pole_pos_cov_pole_lib_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'a_priori_pole': True, 
            'pole_lib_cov': True,
        },
        "SimPole_initial_state": {
            "simulation_path": "Results/PoleEstimationRealObservations/PoleSimPoleCASE2/hybrid_weights_initial_state",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': False,
        },
        "SimPole_pole_pos_cov": {
            "simulation_path": "Results/PoleEstimationRealObservations/PoleSimPoleCASE2/hybrid_weights_pole_pos_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'pole_pos_cov': True 
        },
        "SimPole_pole_lib_cov": {
            "simulation_path": "Results/PoleEstimationRealObservations/PoleSimPoleCASE2/hybrid_weights_pole_lib_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'pole_pos_cov': False, 
            'pole_lib_cov': True,
        },
        "SimPole_pole_pos_cov_pole_lib_cov": {
            "simulation_path": "Results/PoleEstimationRealObservations/PoleSimPoleCASE2/hybrid_weights_pole_pos_cov_pole_lib_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'a_priori_pole': True, 
            'pole_lib_cov': True,
        },
    }


  # ULTIMATE ANALYSIS 
    VARIANTS = {
        "IAUPole_initial_state_no_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/UltimateCASE1/IAUPole_initial_state_no_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "IAUPole_initial_state": {
            "simulation_path": "Results/PoleEstimationRealObservations/UltimateCASE1/IAUPole_initial_state",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': False,
        },
        "IAUPole_pole_pos_cov": {
            "simulation_path": "Results/PoleEstimationRealObservations/UltimateCASE1/IAUPole_pole_pos_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'pole_pos_cov': True 
        },
        "IAUPole_pole_lib_cov": {
            "simulation_path": "Results/PoleEstimationRealObservations/UltimateCASE1/IAUPole_pole_lib_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'pole_pos_cov': False, 
            'pole_lib_cov': True,
        },
        "IAUPole_pole_pos_cov_pole_lib_cov": {
            "simulation_path": "Results/PoleEstimationRealObservations/UltimateCASE1/IAUPole_pole_pos_cov_pole_lib_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'a_priori_pole': True, 
            'pole_lib_cov': True,
        },
        "SimPole_initial_state_no_weights": {
            "simulation_path": "Results/PoleEstimationRealObservations/UltimateCASE1/SimPole_initial_state_no_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "SimPole_initial_state": {
            "simulation_path": "Results/PoleEstimationRealObservations/UltimateCASE1/SimPole_initial_state",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': False,
        },
        "SimPole_pole_pos_cov": {
            "simulation_path": "Results/PoleEstimationRealObservations/UltimateCASE1/SimPole_pole_pos_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'pole_pos_cov': True 
        },
        "SimPole_pole_lib_cov": {
            "simulation_path": "Results/PoleEstimationRealObservations/UltimateCASE1/SimPole_pole_lib_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'pole_pos_cov': False, 
            'pole_lib_cov': True,
        },
        "SimPole_pole_pos_cov_pole_lib_cov": {
            "simulation_path": "Results/PoleEstimationRealObservations/UltimateCASE1/SimPole_pole_pos_cov_pole_lib_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'a_priori_pole': True, 
            'pole_lib_cov': True,
        },
    }

  # ULTIMATE ANALYSIS with manual bias
    VARIANTS = {
        "IAUPole_initial_state_no_weights": {
            "simulation_path": "Results/ManualBias/CASE1_Manual_Bias/IAUPole_initial_state_no_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "IAUPole_initial_state": {
            "simulation_path": "Results/ManualBias/CASE1_Manual_Bias/IAUPole_initial_state",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': False,
        },
        "IAUPole_pole_pos_cov": {
            "simulation_path": "Results/ManualBias/CASE1_Manual_Bias/IAUPole_pole_pos_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'pole_pos_cov': True 
        },
        "IAUPole_pole_lib_cov": {
            "simulation_path": "Results/ManualBias/CASE1_Manual_Bias/IAUPole_pole_lib_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'pole_pos_cov': False, 
            'pole_lib_cov': True,
        },
        "IAUPole_pole_pos_cov_pole_lib_cov": {
            "simulation_path": "Results/ManualBias/CASE1_Manual_Bias/IAUPole_pole_pos_cov_pole_lib_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'a_priori_pole': True, 
            'pole_lib_cov': True,
        },
        "SimPole_initial_state_no_weights": {
            "simulation_path": "Results/ManualBias/CASE1_Manual_Bias/SimPole_initial_state_no_weights",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "SimPole_initial_state": {
            "simulation_path": "Results/ManualBias/CASE1_Manual_Bias/SimPole_initial_state",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': False,
        },
        "SimPole_pole_pos_cov": {
            "simulation_path": "Results/ManualBias/CASE1_Manual_Bias/SimPole_pole_pos_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'pole_pos_cov': True 
        },
        "SimPole_pole_lib_cov": {
            "simulation_path": "Results/ManualBias/CASE1_Manual_Bias/SimPole_pole_lib_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'pole_pos_cov': False, 
            'pole_lib_cov': True,
        },
        "SimPole_pole_pos_cov_pole_lib_cov": {
            "simulation_path": "Results/ManualBias/CASE1_Manual_Bias/SimPole_pole_pos_cov_pole_lib_cov",
            'est_parameters': ['initial_state','iau_rotation_model_pole','iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': True,
            'use_apriori_cov': True,
            'a_priori_pole': True, 
            'pole_lib_cov': True,
        },
    }


    # #SIMULATED OBSERVATIONS
    # VARIANTS = {
    #     "initial_state_IAU": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/initial_state_IAU",
    #         'est_parameters': ['initial_state'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "initial_state_Jacobson": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/initial_state_Jacobson",
    #         'est_parameters': ['initial_state'],
    #         'Neptune_rot_model_type': 'Pole_Model_Jacobson2009',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "GM_Triton_IAU": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/GM_Triton_IAU",
    #         'est_parameters': ['initial_state', 'GM_Triton'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "GM_Neptune_IAU": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/GM_Neptune_IAU",
    #         'est_parameters': ['initial_state', 'GM_Neptune'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "GM_Both_IAU": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/GM_Both_IAU",
    #         'est_parameters': ['initial_state', 'GM_Triton', 'GM_Neptune'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "sh_IAU": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/sh_IAU",
    #         'est_parameters': ['initial_state', 'spherical_harmonics'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "pole_pos_IAU": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_pos_IAU",
    #         'est_parameters': ['initial_state', 'iau_rotation_model_pole'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "pole_rot_IAU": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_rot_IAU",
    #         'est_parameters': ['initial_state', 'iau_rotation_model_pole_rate'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "pole_pos_rot_IAU": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_pos_rot_IAU",
    #         'est_parameters': ['initial_state', 'iau_rotation_model_pole', 'iau_rotation_model_pole_rate'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "pole_lib_IAU": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_lib_IAU",
    #         'est_parameters': ['initial_state', 'iau_rotation_model_pole_librations'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "pole_full_IAU": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_full_IAU",
    #         'est_parameters': ['initial_state', 'iau_rotation_model_pole', 'iau_rotation_model_pole_rate', 'iau_rotation_model_pole_librations'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "pole_pos_Jacobson": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_pos_Jacobson",
    #         'est_parameters': ['initial_state', 'iau_rotation_model_pole'],
    #         'Neptune_rot_model_type': 'Pole_Model_Jacobson2009',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "pole_rot_Jacobson": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_rot_Jacobson",
    #         'est_parameters': ['initial_state', 'iau_rotation_model_pole_rate'],
    #         'Neptune_rot_model_type': 'Pole_Model_Jacobson2009',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "pole_lib1_Jacobson": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_lib1_Jacobson",
    #         'est_parameters': ['initial_state', 'pole_librations_deg2'],
    #         'Neptune_rot_model_type': 'Pole_Model_Jacobson2009',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "pole_lib2_Jacobson": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_lib2_Jacobson",
    #         'est_parameters': ['initial_state', 'pole_librations_deg2'],
    #         'Neptune_rot_model_type': 'Pole_Model_Jacobson2009',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "pole_full_Jacobson": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_full_Jacobson",
    #         'est_parameters': ['initial_state', 'iau_rotation_model_pole', 'iau_rotation_model_pole_rate', 'pole_librations_deg2'],
    #         'Neptune_rot_model_type': 'Pole_Model_Jacobson2009',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "SH_pole_full_IAU": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/SH_pole_full_IAU",
    #         'est_parameters': ['initial_state', 'spherical_harmonics', 'iau_rotation_model_pole', 'iau_rotation_model_pole_rate', 'iau_rotation_model_pole_librations'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "GM_SH_IAU": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/GM_SH_IAU",
    #         'est_parameters': ['initial_state', 'GM_Neptune', 'GM_Triton', 'spherical_harmonics'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    #     "all_IAU": {
    #         "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/all_IAU",
    #         'est_parameters': ['initial_state', 'GM_Triton', 'GM_Neptune', 'spherical_harmonics', 'iau_rotation_model_pole', 'iau_rotation_model_pole_rate', 'iau_rotation_model_pole_librations'],
    #         'Neptune_rot_model_type': 'IAU2015',
    #         'use_weights': False,
    #         'use_apriori_cov': False,
    #     },
    # }


    #settings['obs']['type'] = 'Simulated'
    runSim = False
    if runSim == True:
        results = {}
        for name, content in VARIANTS.items():
            print("######################################")
            print("Running Sim ",name)
            print("######################################")
            
            out_dir_current = out_dir / name
            out_dir_current.mkdir(parents=True, exist_ok=True)
            
            settings_est['est_parameters'] = content['est_parameters'] 
            
            #Select different Pole Model or return to default
           
            settings['env']['Neptune_rot_model_type'] = content['Neptune_rot_model_type']

            #Run estimation
            ObservationImplementation.main(settings,out_dir_current)





    #-----------------#-----------------#-----------------#-----------------#-----------------#-----------------

    #Load 
    print("#######################################################################################################")
    print("LOAD SIMPOLE MODEL") # LOAD OBSERVATIONS WITH WEIGHTS")
    print("#######################################################################################################")
    
    
    # # #Create Environment 
    # body_settings,system_of_bodies = PropFuncs.Create_Env(settings['env'])

    # #Load observations
    # observations,observations_settings,observation_set_ids, epochs_rejected = ObsFunc.LoadObservations(
    #         settings["obs"]["observations_folder_path"],
    #         system_of_bodies,
    #         settings['obs']["files"],
    #         Residual_filtering = settings["obs"]["residual_filtering"])

    #-------------------------
    fitted_pole_pos_lib_sim = PostProc.load_npy_files("Results/EstimatedParametersSimulatedObservations/PoleLibrations/pole_pos_and_libration_amplitude")

    settings['prop']['initial_state'] =  fitted_pole_pos_lib_sim['parameter_history'][0:6,-1]    
    

    pole_params_SimPole = fitted_pole_pos_lib_sim['parameter_history'][6:,-1]
    # runSim = False
    # if runSim == True :
    #     #Load observations with weights
    #     print("#######################################################################################################")
    #     print("RUN INITIAL ESTIMATION AND CREATE WEIGHTS SIM POLE") # LOAD OBSERVATIONS WITH WEIGHTS")
    #     print("#######################################################################################################")
        

    #     settings['env']['initial_Pole_Pos'] = pole_params_SimPole[0:2]
    #     settings['env']['initial_Pole_lib_deg1'] = pole_params_SimPole[2:4]



    #     out_dir_current = out_dir / 'SimPole_initial_state_no_weights'
    #     out_dir_current.mkdir(parents=True, exist_ok=True)
        
    #     # Set estimation parameters
    #     settings['est']['est_parameters'] = VARIANTS['SimPole_initial_state_no_weights']['est_parameters'] 
        
    #     #Set covariances if any
        
    #     # Set if to use a priori cov or not
    #     settings['est']['a_priori_covariance'] = False #VARIANTS['SimPole_initial_state_no_weights']['use_apriori_cov']

    #     settings['est']['a_priori_pole'] = False
    #     settings['est']['a_priori_lib'] = False
            

    #     estimation_output,observations,observations_settings,body_settings,system_of_bodies = ObservationImplementation.main(
    #             settings,
    #             out_dir_current)


    #     #First estimation (initial_state only) without weights is used to generate weights

    #     #simulation_weights_path = "Results/PoleEstimationRealObservations/UltimateCASE1/SimPole_initial_state_no_weights"
    #     simulation = PostProc.load_npy_files(out_dir_current)

    #     residuals = simulation['residual_history_arcseconds'][-1]

    #     # Convert RA and DEC columns from arcseconds to radians
    #     residuals[:, 1] = residuals[:, 1] / (3600 * 180 / np.pi)  # RA
    #     residuals[:, 2] = residuals[:, 2] / (3600 * 180 / np.pi)  # DEC


    #     # #Create Environment 
    #     body_settings,system_of_bodies = PropFuncs.Create_Env(settings['env'])

    #     #Load observations
    #     observations,observations_settings,observation_set_ids, epochs_rejected = ObsFunc.LoadObservations(
    #             settings["obs"]["observations_folder_path"],
    #             system_of_bodies,
    #             settings['obs']["files"],
    #             Residual_filtering = settings["obs"]["residual_filtering"])



    #     bias_dict = {
    #     "689_nm0077": -0.2,  # arcsec
    #     }

    #     times_sec = observations.get_concatenated_observation_times()
    #     residuals_old = observations.get_concatenated_residuals()

    #     # observations and observations_biased are pointing in the same memory block
    #     # therefore observations are overwritten !!
    #     observations_biased, applied = ObsFunc.apply_dec_bias_to_observations(
    #         observations,
    #         observations_settings,
    #         system_of_bodies,
    #         bias_dict
    #     )

    #     residuals_new = observations_biased.get_concatenated_residuals()
    #     fig = ObsFunc.PlotResidualBiased(times_sec,residuals_old,residuals_new)
    #     fig.savefig(out_dir / "ManualBias_SimPole.pdf")



    #     # EXTRACT RESIDUALS FROM INITIAL SIM 
    #     # AND COMPUTE/ASSIGN WEIGHTS FROM THEM
    #     observations_SimPole, weights_info_SimPole = ObsFunc.compute_and_assign_weights(
    #         residuals=residuals,
    #         observations=observations_biased,
    #         gap_threshold_hours=4.0,
    #         min_obs_per_frame=1,
    #         weight_type = 'hybrid'
    #     )



    #     #Load observations with weights
    #     print("#######################################################################################################")
    #     print("RUN INITIAL ESTIMATION AND CREATE WEIGHTS IAU POLE") # LOAD OBSERVATIONS WITH WEIGHTS")
    #     print("#######################################################################################################")
        
    #     out_dir_current = out_dir / 'IAUPole_initial_state_no_weights'
    #     out_dir_current.mkdir(parents=True, exist_ok=True)
        

    #     settings['env'].pop('initial_Pole_Pos', None)
    #     settings['env'].pop('initial_Pole_lib_deg1', None)
    #     settings['prop'].pop('initial_state',None)

    #     # # Set estimation parameters
    #     settings['est']['est_parameters'] = VARIANTS['IAUPole_initial_state_no_weights']['est_parameters'] 
        
    #     # #Set covariances if any
        
    #     # # Set if to use a priori cov or not
    #     settings['est']['a_priori_covariance'] = VARIANTS['IAUPole_initial_state_no_weights']['use_apriori_cov']

    #     settings['est']['a_priori_pole'] = False
    #     settings['est']['a_priori_lib'] = False
            

    #     estimation_output,observations,observations_settings,body_settings,system_of_bodies = ObservationImplementation.main(
    #             settings,
    #             out_dir_current)




    #     # #First estimation (initial_state only) without weights is used to generate weights
    #     # out_dir_current
    #     # # simulation_weights_path = "Results/PoleEstimationRealObservations/EstimationWeightLoop/Initial_State_No_Weights"
    #     simulation = PostProc.load_npy_files(out_dir_current)

    #     residuals = simulation['residual_history_arcseconds'][-1]

    #     # # Convert RA and DEC columns from arcseconds to radians
    #     residuals[:, 1] = residuals[:, 1] / (3600 * 180 / np.pi)  # RA
    #     residuals[:, 2] = residuals[:, 2] / (3600 * 180 / np.pi)  # DEC


    #     #--------------------------------------------------------------------------
    #     # #Create Environment 
    #     body_settings,system_of_bodies = PropFuncs.Create_Env(settings['env'])

    #     #Load observations
    #     observations,observations_settings,observation_set_ids, epochs_rejected = ObsFunc.LoadObservations(
    #             settings["obs"]["observations_folder_path"],
    #             system_of_bodies,
    #             settings['obs']["files"],
    #             Residual_filtering = settings["obs"]["residual_filtering"])



    #     bias_dict = {
    #     "689_nm0077": -0.2,  # arcsec
    #     }

    #     times_sec = observations.get_concatenated_observation_times()
    #     residuals_old = observations.get_concatenated_residuals()

    #     # observations and observations_biased are pointing in the same memory block
    #     # therefore observations are overwritten !!
    #     observations_biased, applied = ObsFunc.apply_dec_bias_to_observations(
    #         observations,
    #         observations_settings,
    #         system_of_bodies,
    #         bias_dict
    #     )

    #     residuals_new = observations_biased.get_concatenated_residuals()
    #     fig = ObsFunc.PlotResidualBiased(times_sec,residuals_old,residuals_new)
    #     fig.savefig(out_dir / "ManualBias_IAUPole.pdf")

    #     #--------------------------------------------------------------------------
    #     # EXTRACT RESIDUALS FROM INITIAL SIM 
    #     # AND COMPUTE/ASSIGN WEIGHTS FROM THEM
    #     observations_IAUPole, weights_info_IAUPole = ObsFunc.compute_and_assign_weights(
    #         residuals=residuals,
    #         observations=observations_biased,
    #         gap_threshold_hours=4.0,
    #         min_obs_per_frame=1,
    #         weight_type = 'hybrid'
    #     )


    #     print("#######################################################################################################")
    #     print("RUN SIMULATIONS")
    #     print("#######################################################################################################")
    

    #     settings['env']['Neptune_rot_model_type'] ='IAU2015'


    #     for name, content in VARIANTS.items():
    #         if name != 'IAUPole_initial_state_no_weights' and name != 'SimPole_initial_state_no_weights':
    #             print("######################################")
    #             print("Running Sim ",name)
    #             print("######################################")
                
    #             out_dir_current = out_dir / name
    #             out_dir_current.mkdir(parents=True, exist_ok=True)
                
    #             estimation_type = name.split('_')[0]
    #             #Assign initial pole pos and lib based on estimation type
    #             if estimation_type == 'IAUPole':
    #                 settings['env'].pop('initial_Pole_Pos', None)
    #                 settings['env'].pop('initial_Pole_lib_deg1', None)
    #                 settings['prop'].pop('initial_state',None)
    #             elif estimation_type == 'SimPole':
    #                 settings['prop']['initial_state'] =  fitted_pole_pos_lib_sim['parameter_history'][0:6,-1]    
    
    #                 settings['env']['initial_Pole_Pos'] = pole_params_SimPole[0:2]
    #                 settings['env']['initial_Pole_lib_deg1'] = pole_params_SimPole[2:4]



    #             # Set estimation parameters
    #             settings['est']['est_parameters'] = content['est_parameters'] 
                
    #             #Set covariances if any
                
    #             # Set if to use a priori cov or not
    #             settings['est']['a_priori_covariance'] = content['use_apriori_cov']

    #             if 'pole_pos_cov' in content :
    #                 settings['est']['a_priori_pole'] = content['pole_pos_cov']
    #             else:
    #                 settings['est']['a_priori_pole'] = False
                
    #             if 'pole_lib_cov' in content :
    #                 settings['est']['a_priori_lib'] = content['pole_lib_cov']
    #                 settings['est']['a_priori_lib_deg'] = 1
    #             else:
    #                 settings['est']['a_priori_lib'] = False
                    


    #             #Run estimation + provide weights if needed
    #             if content['use_weights'] == True:
    #                 settings['obs']['use_loaded_obs'] = True 
    #                 settings['obs']['use_old_obs_func'] = False
    #                 settings["obs"]["manual_dec_bias"] = bias_dict
            
    #                 if estimation_type == 'IAUPole':
    #                     estimation_output,_,_,body_settings,system_of_bodies = ObservationImplementation.main(
    #                             settings,
    #                             out_dir_current,
    #                             observations=observations_IAUPole,
    #                             observations_settings=observations_settings)
    #                     weights_info_IAUPole.to_csv(out_dir_current / 'observation_weights.csv', index=False)
                    
                        
    #                 elif estimation_type == 'SimPole':
    #                     estimation_output,_,_,body_settings,system_of_bodies = ObservationImplementation.main(
    #                             settings,
    #                             out_dir_current,
    #                             observations=observations_SimPole,
    #                             observations_settings=observations_settings)
    #                     weights_info_SimPole.to_csv(out_dir_current / 'observation_weights.csv', index=False)
                    
                        
    #             else:
    #                 estimation_output,observations,observations_settings,body_settings,system_of_bodies = ObservationImplementation.main(
    #                         settings,
    #                         out_dir_current)




#Test
#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------

# EXAMPLE CODE 
##################################################################################################################

# #Create Environment 
# body_settings,system_of_bodies = PropFuncs.Create_Env(settings['env'])

# #Load observations
# observations,observations_settings,observation_set_ids, epochs_rejected = ObsFunc.LoadObservations(
#         settings["obs"]["observations_folder_path"],
#         system_of_bodies,
#         settings['obs']["files"],
#         Residual_filtering = settings["obs"]["residual_filtering"])

# #Load residuals from previous simulation to use as weights
# simulation = PostProc.load_npy_files('Results/PoleEstimationRealObservations/TestNewWeights_Full_NoImprovement/First')

# best_iteration = simulation['best_iteration']
# residuals = simulation['residual_history_arcseconds'][best_iteration]

# # Convert RA and DEC columns from arcseconds to radians
# residuals[:, 1] = residuals[:, 1] / (3600 * 180 / np.pi)  # RA
# residuals[:, 2] = residuals[:, 2] / (3600 * 180 / np.pi)  # DEC


# settings['prop']['initial_state'] = simulation['parameter_history'][:,best_iteration]
# settings['obs']['use_loaded_obs'] = True

##################################################################################################################




# # RUN ESTIMATION WITHOUT WEIGHTS FIRST
print("######################################")
print("Running NO WEIGHTS INITIAL STATE SIM")
print("######################################")

#fitted_pole_pos_lib_sim = PostProc.load_npy_files("Results/EstimatedParametersSimulatedObservations/PoleLibrations/pole_pos_and_libration_amplitude")

# settings['est']['a_priori_covariance'] = False

# settings['obs']['use_loaded_obs'] = False
# settings['prop']['initial_state'] = None #simulation['parameter_history'][:,best_iteration]


# settings['est']['est_parameters'] = ['initial_state']

# # pole_params = fitted_pole_pos_lib_sim['parameter_history'][6:,-1]
# # # Create the numpy array for [alpha_i, delta_i] as a 2x1 column vector

# # settings['env']['initial_Pole_Pos'] = pole_params[0:2]

# # settings['env']['initial_Pole_lib_deg1'] = pole_params[2:4]


# out_dir_current = out_dir / "Initial_State_No_Weights"
# out_dir_current.mkdir(parents=True, exist_ok=True)

# estimation_output,observations,observations_settings,body_settings,system_of_bodies = ObservationImplementation.main(
#         settings,
#         out_dir_current)

#Create Environment 
# body_settings,system_of_bodies = PropFuncs.Create_Env(settings['env'])

# #Load observations
# observations,observations_settings,observation_set_ids, epochs_rejected = ObsFunc.LoadObservations(
#         settings["obs"]["observations_folder_path"],
#         system_of_bodies,
#         settings['obs']["files"],
#         Residual_filtering = settings["obs"]["residual_filtering"])

# path_CASE1_weights = "Results/PoleEstimationRealObservations/UltimateCASE1/IAUPole_initial_state_no_weights"
# simulation = PostProc.load_npy_files(path_CASE1_weights)

# residuals = simulation['residual_history_arcseconds'][-1]

# # Convert RA and DEC columns from arcseconds to radians
# residuals[:, 1] = residuals[:, 1] / (3600 * 180 / np.pi)  # RA
# residuals[:, 2] = residuals[:, 2] / (3600 * 180 / np.pi)  # DEC


# # EXTRACT RESIDUALS FROM INITIAL SIM 
# # AND COMPUTE/ASSIGN WEIGHTS FROM THEM
# observations_weighted, weights_info = ObsFunc.compute_and_assign_weights(
#     residuals=residuals,
#     observations=observations,
#     gap_threshold_hours=4.0,
#     min_obs_per_frame=1,
#     weight_type = 'hybrid'
# )



# # RUN ESTIMATION HYBRID WEIGHTS
# print("######################################") 
# print("Running INITIAL STATE HYBRID WEIGHTS SIM LOOP")
# print("######################################")

# for i in range(5):

#     observations_weighted, weights_info = ObsFunc.compute_and_assign_weights(
#         residuals=residuals,
#         observations=observations,
#         gap_threshold_hours=4.0,
#         min_obs_per_frame=1,
#         weight_type = 'hybrid'
#     )

#     settings['prop']['initial_state'] = simulation['parameter_history'][:,-1]
#     settings['obs']['use_loaded_obs'] = True

#     out_dir_current = out_dir / ("initial_state_hybrid_weights_" + str(i))
#     out_dir_current.mkdir(parents=True, exist_ok=True)


#     # # Save weights to CSV
#     weights_info.to_csv(out_dir_current / 'observation_weights.csv', index=False)

#     #print("Observation weights: ",observations_weighted.get_concatenated_weights())
#     estimation_output,observations,observations_settings,body_settings,system_of_bodies = ObservationImplementation.main(
#             settings,
#             out_dir_current,
#             observations=observations_weighted,
#             observations_settings=observations_settings)

#     #COMPUTE RESIDUALS FOR NEXT ITERATION
#     simulation = PostProc.load_npy_files(out_dir_current)

#     residuals = simulation['residual_history_arcseconds'][-1]

#     # Convert RA and DEC columns from arcseconds to radians
#     residuals[:, 1] = residuals[:, 1] / (3600 * 180 / np.pi)  # RA
#     residuals[:, 2] = residuals[:, 2] / (3600 * 180 / np.pi)  # DEC





print("#########################################################################################")
print("CREATE SPICE STATES")
print("#######################################################################################################")

##############################################################################################
# LOAD SPICE KERNELS
##############################################################################################
kernel_folder = "Kernels/"
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


# # GET SPICE Results (any sim should work)
# epochs = simulations['Initial_State_No_Weights']['state_history_array'][:,0]


# GET SPICE Results (any sim should work)
epochs_full = simulations[list(simulations.keys())[0]]['state_history_array'][:,0]

# Downsample: select every N-th point (adjust N for desired spacing)
# For ~4-5 hours with 1-hour spacing, use N=4 or N=5
downsample_factor = 5  # This gives you 1 point every 5 hours
epochs = epochs_full[::downsample_factor]


print(f"Original number of epochs: {len(epochs_full)}")
print(f"Downsampled number of epochs: {len(epochs)} (every {downsample_factor} hours)")


states_SPICE = np.array([
    spice.get_body_cartesian_state_at_epoch(
        target_body_name="Triton",
        observer_body_name="Neptune",
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=epoch
    )
    for epoch in epochs
])


   
time_column = epochs.reshape(-1, 1)
states_SPICE_with_time = np.hstack((time_column, states_SPICE))


states_SPICE_full = np.array([
    spice.get_body_cartesian_state_at_epoch(
        target_body_name="Triton",
        observer_body_name="Neptune",
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=epoch
    )
    for epoch in epochs_full
])

time_column_full = epochs_full.reshape(-1, 1)
states_SPICE_with_time_full = np.hstack((time_column_full, states_SPICE_full))


print("#######################################################################################################")
print("CREATE RSW FORMAL ERRORS")
print("#######################################################################################################")

# pole_params = fitted_pole_pos_lib_sim['parameter_history'][6:,-1]

# settings['env']['initial_Pole_Pos'] = pole_params[0:2]
# settings['env']['initial_Pole_lib_deg1'] = pole_params[2:4]

# Run Propagation of Covariances and rotate to RSW 
RunFormalErrors = False
if RunFormalErrors == True:

    settings["prop"].pop("initial_state",None)
    settings['env'].pop('initial_Pole_Pos', None)
    settings['env'].pop('initial_Pole_lib_deg1', None)
    for name in simulations.keys():
        if VARIANTS[name]['use_weights'] == True:

            #Assign initial pole pos and lib based on simulation
            if name.split('_')[0] == 'IAUPole':
                settings['env'].pop('initial_Pole_Pos', None)
                settings['env'].pop('initial_Pole_lib_deg1', None)
            elif name.split('_')[0] == 'SimPole':
                pole_params_sim = fitted_pole_pos_lib_sim['parameter_history'][6:,-1]

                settings['env']['initial_Pole_Pos'] = pole_params_sim[0:2]
                settings['env']['initial_Pole_lib_deg1'] = pole_params_sim[2:4]


            settings['est']['est_parameters'] = simulations[name]["est_parameters"]

            settings["prop"]["initial_covariance"] = simulations[name]['covariance'] #[:6, :6]
            settings["prop"]["initial_state"] = simulations[name]['parameter_history'][0:6, -1]



            pole_params = simulations[name]['parameter_history'][6:, -1]

            # Check the size and assign accordingly
            if len(pole_params) == 2:
                if 'iau_rotation_model_pole' in VARIANTS[name]['est_parameters']:
                    settings['env']['initial_Pole_Pos'] = pole_params[0:2]
                elif 'iau_rotation_model_pole_librations' in VARIANTS[name]['est_parameters']:
                    settings['env']['initial_Pole_lib_deg1'] = pole_params[0:2]
            elif len(pole_params) == 4:
                # Assign both pole position and libration degree 1
                settings['env']['initial_Pole_Pos'] = pole_params[0:2]
                settings['env']['initial_Pole_lib_deg1'] = pole_params[2:4]
                
            out_dir_current = out_dir / (name + '_CovarianceAnalysis')
            out_dir_current.mkdir(parents=True, exist_ok=True)

            state_history_array,covariances = UncertantyPropUtils.RunSinglePropagation(settings, out_dir_current)
            
            # # Set pole params back to initial 
            # pole_params = fitted_pole_pos_lib_sim['parameter_history'][6:,-1]

            # settings['env']['initial_Pole_Pos'] = pole_params[0:2]
            # settings['env']['initial_Pole_lib_deg1'] = pole_params[2:4]

            settings['env'].pop('initial_Pole_Pos', None)
            settings['env'].pop('initial_Pole_lib_deg1', None)

            #Check if state_history_array is the same as 
            state_history_array_sim = simulations[name]['state_history_array_full'][-1]
            print('Max Diff X: ',np.max(state_history_array[:,1]-state_history_array_sim[:,1]))
            print('Max Diff Y: ',np.max(state_history_array[:,2]-state_history_array_sim[:,2]))
            print('Max Diff Z: ',np.max(state_history_array[:,3]-state_history_array_sim[:,3]))
            #Rotate to RSW (of SPICE?) 
            Covariances_array_RSW = ProcessingUtils.rotate_covariance_inertial_to_rsw(time_column_full, covariances, states_SPICE_with_time_full)
            np.save(out_dir_current / "covariances_array_rsw.npy",Covariances_array_RSW)

            diag = np.diagonal(Covariances_array_RSW, axis1=1, axis2=2)
            formal_errors_RSW = np.sqrt(diag[:,0:3])/1e3
            
            fig_RSW = FigUtils.Residuals_RSW(formal_errors_RSW, time_column_full,type="difference",title=("RSW Formal Errors " + name))   
            fig_RSW.savefig(out_dir_current / ("Formal_Errors_Propagated_RSW" + name + ".pdf"))

            simulations[name]['formal_errors_RSW_km'] = formal_errors_RSW



print("#######################################################################################################")
print("CREATE RSW STATES FIGS")
print("#######################################################################################################")

diff_SPICE_RSW = {}
rms_SPICE = {}
rms_Norm = {}
diff_wrt_norm_RSW = {}
diff_init_SPICE_RSW = {}
time_column_initial = {}

states_SPICE_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, states_SPICE[:,0:3], states_SPICE_with_time)

# state_history_array_norm = simulations['initial_state_only']['state_history_array']
# states_norm_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array_norm[:,1:4], state_history_array_norm)

RSW_RMS_PLOTS = True
if RSW_RMS_PLOTS == True:
    for key in simulations.keys():
        state_history_array_full = simulations[key]['state_history_array_full'][-1]
        
        state_history_array_init = simulations[key]['state_history_array_full'][0]
        # Downsample the simulation state history using the same factor
        state_history_array = state_history_array_full[::downsample_factor]
        state_history_array_init_downsample = state_history_array_init[::downsample_factor]

        states_sim_RSW_SPICE = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array[:,1:4], states_SPICE_with_time)
       
        states_init_sim_RSW_SPICE = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array_init_downsample[:,1:4], states_SPICE_with_time)
       
        #states_sim_RSW_norm = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array[:,1:4], state_history_array_norm)
        #Diff wrt to SPICE
        # #---------------------------------------------------------------------------------------
        diff = (states_SPICE_RSW - states_sim_RSW_SPICE)/1e3
        diff_SPICE_RSW[key] = diff 

        diff_init = (states_SPICE_RSW - states_init_sim_RSW_SPICE)/1e3
        diff_init_SPICE_RSW[key] = diff_init 
        time_column_initial[key] = time_column

        fig_RSW = FigUtils.Residuals_RSW(diff, time_column,type="difference",title=("RSW Difference SPICE - " + key))   
        fig_RSW.savefig(out_dir / ("RSW Diff SPICE - " + key + ".pdf"))

        # unweighted scalar RMS wrt SPICE
        rms_SPICE[key] = np.sqrt(np.mean(diff**2)) 



        #Diff wrt to norm
        # #---------------------------------------------------------------------------------------
        # diff = (states_norm_RSW- states_sim_RSW_norm)/1e3
        # diff_wrt_norm_RSW[key] = diff 
        # fig_RSW = FigUtils.Residuals_RSW(diff, time_column,type="difference",title=("RSW Difference Norm - " + key))   
        # fig_RSW.savefig(out_dir / ("RSW Diff Norm_ " + key + ".pdf"))

        # rms_Norm[key] = np.sqrt(np.mean(diff**2)) 


    # #---------------------------------------------------------------------------------------
    # rms_SPICE.pop('spherical_harmonics_pole_full') 
    # rms_Norm.pop('spherical_harmonics_pole_full')


    fig_RMS_SPICE = FigUtils.plot_rms_comparison(rms_SPICE)
    fig_RMS_SPICE.savefig(out_dir / "RMS_SPICE.pdf")

    stats = {}
    for key, arr in diff_SPICE_RSW.items():
        stats[key] = {
            "mean": arr.mean(axis=0),  # [mean_R, mean_S, mean_W] as array
            "std": arr.std(axis=0),    # [std_R, std_S, std_W] as array
            "rms_R": np.sqrt(np.mean(arr[:, 0]**2)),
            "rms_S": np.sqrt(np.mean(arr[:, 1]**2)),
            "rms_W": np.sqrt(np.mean(arr[:, 2]**2)),
            "max_R": np.abs(arr[:, 0]).max(),
            "max_S": np.abs(arr[:, 1]).max(),
            "max_W": np.abs(arr[:, 2]).max(),
        }


    def plot_rsw_statistics(stats):
        """
        Create a comprehensive RSW statistics comparison plot with line plots
        
        Parameters:
        -----------
        stats : dict
            Dictionary with simulation names as keys, each containing:
            - 'mean': array [mean_R, mean_S, mean_W]
            - 'std': array [std_R, std_S, std_W]
            - 'rms_R', 'rms_S', 'rms_W': scalar RMS values
            - 'max_R', 'max_S', 'max_W': scalar max values
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure with 3x4 subplots showing RSW statistics
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Prepare data
        sim_names = list(stats.keys())
        n_sims = len(sim_names)
        
        # Extract data for each metric
        mean_R = [stats[key]['mean'][0] for key in sim_names]
        mean_S = [stats[key]['mean'][1] for key in sim_names]
        mean_W = [stats[key]['mean'][2] for key in sim_names]
        
        std_R = [stats[key]['std'][0] for key in sim_names]
        std_S = [stats[key]['std'][1] for key in sim_names]
        std_W = [stats[key]['std'][2] for key in sim_names]
        
        rms_R = [stats[key]['rms_R'] for key in sim_names]
        rms_S = [stats[key]['rms_S'] for key in sim_names]
        rms_W = [stats[key]['rms_W'] for key in sim_names]
        
        max_R = [stats[key]['max_R'] for key in sim_names]
        max_S = [stats[key]['max_S'] for key in sim_names]
        max_W = [stats[key]['max_W'] for key in sim_names]
        
        # Create figure with 3 rows, 4 columns
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        
        x_pos = np.arange(n_sims)
        
        # Row 0: R component (blue)
        axes[0, 0].plot(x_pos, mean_R, 'o-', color='tab:blue', linewidth=2, markersize=8)
        axes[0, 0].set_ylabel('R [km]', fontsize=11)
        axes[0, 0].set_title('Mean of Difference wrt SPICE RSW', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        
        axes[0, 1].plot(x_pos, std_R, 'o-', color='tab:blue', linewidth=2, markersize=8)
        axes[0, 1].set_ylabel('R [km]', fontsize=11)
        axes[0, 1].set_title('STD of Difference wrt SPICE RSW', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(x_pos, rms_R, 'o-', color='tab:blue', linewidth=2, markersize=8)
        axes[0, 2].set_ylabel('R [km]', fontsize=11)
        axes[0, 2].set_title('RMS of Difference wrt SPICE RSW', fontsize=12, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[0, 3].plot(x_pos, max_R, 'o-', color='tab:blue', linewidth=2, markersize=8)
        axes[0, 3].set_ylabel('R [km]', fontsize=11)
        axes[0, 3].set_title('Maximum of Difference wrt SPICE RSW', fontsize=12, fontweight='bold')
        axes[0, 3].grid(True, alpha=0.3)
        
        # Row 1: S component (orange)
        axes[1, 0].plot(x_pos, mean_S, 'o-', color='tab:orange', linewidth=2, markersize=8)
        axes[1, 0].set_ylabel('S [km]', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        
        axes[1, 1].plot(x_pos, std_S, 'o-', color='tab:orange', linewidth=2, markersize=8)
        axes[1, 1].set_ylabel('S [km]', fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(x_pos, rms_S, 'o-', color='tab:orange', linewidth=2, markersize=8)
        axes[1, 2].set_ylabel('S [km]', fontsize=11)
        axes[1, 2].grid(True, alpha=0.3)
        
        axes[1, 3].plot(x_pos, max_S, 'o-', color='tab:orange', linewidth=2, markersize=8)
        axes[1, 3].set_ylabel('S [km]', fontsize=11)
        axes[1, 3].grid(True, alpha=0.3)
        
        # Row 2: W component (green)
        axes[2, 0].plot(x_pos, mean_W, 'o-', color='tab:green', linewidth=2, markersize=8)
        axes[2, 0].set_ylabel('W [km]', fontsize=11)
        axes[2, 0].set_xlabel('Simulation', fontsize=11)
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        
        axes[2, 1].plot(x_pos, std_W, 'o-', color='tab:green', linewidth=2, markersize=8)
        axes[2, 1].set_ylabel('W [km]', fontsize=11)
        axes[2, 1].set_xlabel('Simulation', fontsize=11)
        axes[2, 1].grid(True, alpha=0.3)
        
        axes[2, 2].plot(x_pos, rms_W, 'o-', color='tab:green', linewidth=2, markersize=8)
        axes[2, 2].set_ylabel('W [km]', fontsize=11)
        axes[2, 2].set_xlabel('Simulation', fontsize=11)
        axes[2, 2].grid(True, alpha=0.3)
        
        axes[2, 3].plot(x_pos, max_W, 'o-', color='tab:green', linewidth=2, markersize=8)
        axes[2, 3].set_ylabel('W [km]', fontsize=11)
        axes[2, 3].set_xlabel('Simulation', fontsize=11)
        axes[2, 3].grid(True, alpha=0.3)
        
        # Set x-axis labels for all plots
        for row in range(3):
            for col in range(4):
                axes[row, col].set_xticks(x_pos)
                axes[row, col].set_xticklabels(sim_names, rotation=45, ha='right', fontsize=9)
        
        plt.tight_layout()
        
        return fig 
        
    fig = plot_rsw_statistics(stats)

    # Save it
    fig.savefig(out_dir / 'RSW_statistics_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    # Remove simulations with low RMS (initial state and rate only)
    # rms_SPICE.pop('initial_state_only')
    # rms_SPICE.pop('rot_model_rate_only')
    # rms_SPICE.pop('initial_state_only_Pole_Jacbson2009')
    # rms_SPICE.pop('rot_model_rate_only_Pole_Jacobson2009')

    # fig_RMS_SPICE_low = FigUtils.plot_rms_comparison(rms_SPICE)
    # fig_RMS_SPICE_low.savefig(out_dir / "RMS_SPICE_low_RMS_only.pdf")


    # # Compute differences for most promising simulations
    # diff_RSW = diff_SPICE_RSW['initial_state_only'] - diff_SPICE_RSW['pole_libration_amplitude']
    # fig_RSW = FigUtils.Residuals_RSW(diff_RSW, time_column,type="difference",title=("RSW Difference Initial State only vs pole libration amplitude"))   
    # fig_RSW.savefig(out_dir / ("RSW Difference Initial State only vs pole libration amplitude.pdf"))

    # diff_IAU_Jacobson2009_Pole_models_full_rot = diff_SPICE_RSW['pole_full_and_libration_amplitude_Pole_Jacobson2009'] - diff_SPICE_RSW['pole_full_and_libration_amplitude']
    # fig_RSW = FigUtils.Residuals_RSW(diff_IAU_Jacobson2009_Pole_models_full_rot, time_column,type="difference",title=("RSW Difference IAU - Jacobson 2009 full rot model"))   
    # fig_RSW.savefig(out_dir / ("RSW Difference IAU - Jacobson 2009 full rot model.pdf"))
    
    # #---------------------------------------------------------------------------------------


#Check initial state problems:
# diff =simulations['rot_model_rate_only']['state_history_array_full'][0,:,:] - simulations['GM_Triton']['state_history_array_full'][0,:,:] 
# fig_RSW = FigUtils.Residuals_RSW(diff, time_column,type="difference",title=("RSW Difference SPICE - " + key))   
# fig_RSW.savefig(out_dir / ("XYZ Diff Initial rot rate only - GM Triton.pdf"))


print("#######################################################################################################")
print("CREATE RMS PLOTS BASED ON REAL OBSERVATIONS")
print("#######################################################################################################")

# rms_ra = {}
# rms_dec = {}
# for sim_name in simulations.keys():
#     best_iteration = simulations[sim_name]['best_iteration']
#     residuals = simulations[sim_name]['residual_history_arcseconds'][-1]
#     residuals_ra = residuals[:,1]
#     residuals_dec = residuals[:,2]
#     rms_ra[sim_name] = np.sqrt(np.mean(residuals_ra**2))
#     rms_dec[sim_name] = np.sqrt(np.mean(residuals_dec**2))


# def plot_rms_comparison(rms_ra, rms_dec):
#     """Plot RMS residuals for RA and Dec as a single combined metric."""
#     import matplotlib.pyplot as plt
    
#     sim_names = list(rms_ra.keys())
#     ra_values = np.array([rms_ra[name] for name in sim_names])
#     dec_values = np.array([rms_dec[name] for name in sim_names])
    
#     # Combine RA and Dec into single RMS metric
#     combined_rms = np.sqrt(ra_values**2 + dec_values**2)
    
#     fig, ax = plt.subplots(figsize=(14, 6))
    
#     x = np.arange(len(sim_names))
    
#     # Plot with lines and markers
#     ax.plot(x, combined_rms, 'o-', markersize=8, linewidth=2, label='RMS SPICE')
    
#     # Add value labels on points with more decimal places
#     for i, val in enumerate(combined_rms):
#         ax.text(i, val, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
#     ax.set_ylabel('RMS [arcseconds]', fontsize=12)
#     ax.set_xlabel('Estimation Scenario', fontsize=12)
#     ax.set_title('RMS Comparison Across Estimation Scenarios', fontsize=14)
#     ax.set_xticks(x)
#     ax.set_xticklabels(sim_names, rotation=45, ha='right')
#     ax.legend(loc='upper right')
#     ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     return fig

# fig = plot_rms_comparison(rms_ra, rms_dec)
# fig.savefig(out_dir / 'rms_comparison.pdf', dpi=300, bbox_inches='tight')

print("#######################################################################################################")
print("CREATE CORRELATION PLOTS & PARAMETER PLOTS")
print("#######################################################################################################")

# states_SPICE_initial_epoch = spice.get_body_cartesian_state_at_epoch(
#         target_body_name="Triton",
#         observer_body_name="Neptune",
#         reference_frame_name=global_frame_orientation,
#         aberration_corrections="NONE",
#         ephemeris_time=simulation_initial_epoch
#     )

# best_parameter_update = {}
# rms_residuals = {}
# est_parameters_indicies = {}

# #Create a dict of indicies for est parameters per simulation, as it is easier to work with this for later plots.
# for key in simulations.keys():
#         correlations = simulations[key]['correlations']
#         parameter_history = simulations[key]['parameter_history']
#         best_iteration = simulations[key]['best_iteration']
       
#         best_parameter_update[key] = parameter_history[:,-1] 
#         best_parameter_update[key][0:6] = states_SPICE_initial_epoch - parameter_history[0:6,-1]
#         best_parameter_update[key][6:] = parameter_history[6:,0] - parameter_history[6:,-1]

#         est_parameters = simulations[key]['est_parameters']
        
#         #rms_residuals[key] = np.sqrt(np.mean((simulations[key]['residuals_j2000']/1e3)**2)) 


#         # Generate parameter labels and units
#         labels = []
#         units = []
#         groups = []  # For coloring by parameter type

#         for param in est_parameters:
#             if param == 'initial_state':
#                 labels.extend(['x', 'y', 'z', 'vx', 'vy', 'vz'])
#                 units.extend(['m', 'm', 'm', 'm/s', 'm/s', 'm/s'])
#                 groups.extend(['position'] * 3 + ['velocity'] * 3)
            
#             elif param == 'iau_rotation_model_pole':
#                 labels.extend(['α₀', 'δ₀'])
#                 units.extend(['rad', 'rad'])
#                 groups.extend(['pole_position'] * 2)
            
#             elif param == 'iau_rotation_model_pole_rate':
#                 labels.extend(['α̇₀', 'δ̇₀'])
#                 units.extend(['rad/s', 'rad/s'])
#                 groups.extend(['pole_rate'] * 2)
            
#             elif param == 'iau_rotation_model_pole_librations':
#                 labels.extend([[r'\alpha_{i}', r'\delta_{i}']])
#                 units.extend(['rad', 'rad'])
#                 groups.extend(['pole_lib'] * 2)
#                 if 'pole_librations_deg2' in est_parameters:
#                     groups.extend(['pole_lib_deg2']*2)

#             elif param == 'GM_Neptune':
#                 labels.append('GM_Nep')
#                 units.append('km³/s²')
#                 groups.append('gravity')
            
#             elif param == 'GM_Triton':
#                 labels.append('GM_Tri')
#                 units.append('km³/s²')
#                 groups.append('gravity')
            
#             elif param == 'spherical_harmonics':
#                 # Hardcoded for C20 and C40 only
#                 labels.extend(['C20', 'C40'])
#                 units.extend(['[-]', '[-]'])
#                 groups.extend(['spherical_harmonics', 'spherical_harmonics'])
                
#         est_parameters_indicies[key] = groups
#         # print("Plotting figs for: ",key)        
#         # fig = FigUtils.plot_correlation_matrix(correlations, est_parameters)
#         # fig.savefig(out_dir / 'correlations.pdf')

#         # fig1 = FigUtils.plot_parameter_updates(best_parameter_update[key],  est_parameters)
#         # fig1.savefig(out_dir / "parameter_update.pdf")
        
#         # fig2 = FigUtils.plot_parameter_history(parameter_history, est_parameters, best_iteration=best_iteration)
#         # fig2.savefig(out_dir / "parameter_history.pdf")
#         # print("Next")

# fig_mag, fig_comp = FigUtils.plot_state_updates_combined(best_parameter_update)
# fig_mag.savefig(out_dir / "Best_Parameter_Update_magnitude.pdf")
# fig_comp.savefig(out_dir / "Best_Parameter_Update_components.pdf")


# weight_info = pd.read_csv('Results/PoleEstimationRealObservations/WeightTest_CASE1/initial_state_tf_weights/observation_weights.csv')
print("#######################################################################################################")
print("LOAD WEIGHT DATAFRAMES AND MERGE DICTS")
print("#######################################################################################################")
    # # Load simulations
    # simulations = {
    #     name: PostProc.load_npy_files(cfg["simulation_path"])
    #     for name, cfg in VARIANTS.items()
    # }



#ADD WEIGHTS TO SIMS (IF THEY HAVE ANY)

for name,cfg in VARIANTS.items():
    simulations[name]['time_column'] = time_column
    if name != 'Initial_State_No_Weights':
        file_path = cfg['simulation_path'] + '/observation_weights.csv'
        if Path(file_path).exists():
            weight_info = pd.read_csv(file_path)
            simulations[name]['weight_info'] = weight_info

        else:
            print(f"File not found: {file_path}")
        # Append to the inner dict
      

#Merge other dicts
for name in simulations.keys():
    simulations[name]['rms_SPICE'] = rms_SPICE[name]
    simulations[name]['diff_SPICE_RSW'] = diff_SPICE_RSW[name]
    simulations[name]['diff_SPICE_RSW_initial'] = diff_init_SPICE_RSW[name]
    simulations[name]['time_column_initial'] = time_column_initial[name]
import pickle

# Save the complete dict
with open(out_dir / 'simulations.pkl', 'wb') as f:
    pickle.dump(simulations, f)

# file_path = 'Results/PoleEstimationRealObservations/EstimationWeightLoop/Analysis/simulations_with_weights.pkl'
# # Load your simulations dict
# with open(file_path, 'rb') as f:
#     simulations = pickle.load(f)


print("#######################################################################################################")
print("CREATE LATEX TABLES")
print("#######################################################################################################")


# Define parameter names and lengths
param_names = [
    'initial_state',   # 6 elements now
    'GM_Neptune',      # 1
    'GM_Triton',       # 1
    'iau_rotation_model_pole',  # 2
    'iau_rotation_model_pole_rate', # 2
    'iau_rotation_model_pole_librations', #2
    'spherical_harmonics'  # 2
]

param_lengths = [6, 1, 1, 2, 2, 2]  

# Flatten multi-element parameters into individual entries for the table
param_labels = [
    'initial_state_x', 'initial_state_y', 'initial_state_z',
    'initial_state_vx', 'initial_state_vy', 'initial_state_vz',
    'GM_Neptune', 'GM_Triton',
    'iau_rotation_model_pole_alpha', 'iau_rotation_model_pole_delta',
    'iau_rotation_model_pole_rate_alpha_dot', 'iau_rotation_model_pole_rate_delta_dot',
    'iau_rotation_model_pole_librations_alpha_1','iau_rotation_model_pole_librations_delta_1'
    'spherical_harmonics_C20', 'spherical_harmonics_C40'
]



# Loop through all simulations
tables = {}

#initial_values = simulations['all']['parameter_history'][:,0]  # always use all initial values
    
for sim_name in best_parameter_update.keys():
    estimated_values = best_parameter_update[sim_name]
    initial_values = simulations[sim_name]['parameter_history'][:,0]
    
    # Determine which parameters are estimated in this simulation
    est_parameters = simulations[sim_name]['est_parameters']
    labels = []
    
    for param in est_parameters:
        if param == 'initial_state':
            labels.extend(['$x$ [m]', '$y$ [m]', '$z$ [m]', 
                          '$v_x$ [m/s]', '$v_y$ [m/s]', '$v_z$ [m/s]'])
        elif param == 'iau_rotation_model_pole':
            labels.extend([r'$\alpha_0$ [rad]', r'$\delta_0$ [rad]'])
        elif param == 'iau_rotation_model_pole_rate':
            labels.extend([r'$\dot{\alpha}_0$ [rad/s]', r'$\dot{\delta}_0$ [rad/s]'])
        elif param == 'iau_rotation_model_pole_librations':
            labels.extend([r'$\alpha_1$ [rad/]', r'$\delta_1$ [rad]'])
            if 'pole_librations_deg2' in est_parameters:
                labels.extend([r'$\alpha_2$ [rad/]', r'$\delta_2$ [rad]'])
        elif param == 'GM_Neptune':
            labels.append(r'$GM_{\text{Nep}}$ [m$^3$/s$^2$]')
        elif param == 'GM_Triton':
            labels.append(r'$GM_{\text{Tri}}$ [m$^3$/s$^2$]')
        elif param == 'spherical_harmonics':
            labels.extend(['$C_{20}$', '$C_{40}$'])
    
    est_parameters_labels = labels
    
    # Compute % change (handle division by zero)
    percentile_change = np.where(initial_values != 0, 
                                  estimated_values/initial_values*100, 
                                  np.inf)
    
    # Create table
    df = pd.DataFrame({
        'Parameter': est_parameters_labels,
        'Initial Value': initial_values,
        'Estimated Value Update': estimated_values,
        '\% Change': percentile_change
    })
    
    tables[sim_name] = df
    
    # Save files
    csv_file = out_dir / f"estimated_parameters_{sim_name}.csv"
    latex_file = out_dir / f"estimated_parameters_{sim_name}.tex"
    
    # Save CSV (without LaTeX formatting for readability)
    df_csv = df.copy()
    df_csv['Parameter'] = df_csv['Parameter'].str.replace(r'\$', '', regex=True)
    df_csv['Parameter'] = df_csv['Parameter'].str.replace(r'\\text\{|\}|\\dot\{|\}|\{|\}|_|\\alpha|\\delta', '', regex=True)
    df_csv.to_csv(csv_file, index=False)
    
    # Save LaTeX with proper formatting
    latex_content = df.to_latex(
        index=False,
        float_format="%.2e",  # Changed to %.2e for more compact notation
        caption=f"Estimated parameters for simulation {sim_name}",
        label=f"tab:estimated_parameters_{sim_name}",
        escape=False,  # IMPORTANT: Don't escape LaTeX commands
        column_format='lrrr'
    )
    
    # Post-process LaTeX to use scientific notation properly
    import re
    
    # Replace e notation with \times 10^ notation
    def format_sci_notation(match):
        mantissa = float(match.group(1))
        exponent = int(match.group(2))
        if exponent == 0:
            return f"${mantissa:.2f}$"
        return f"${mantissa:.2f} \\times 10^{{{exponent}}}$"
    
    latex_content = re.sub(r'(-?\d+\.\d+)e([+-]\d+)', format_sci_notation, latex_content)
    
    # Replace inf with ---
    latex_content = latex_content.replace('inf', '---')
    
    with open(latex_file, 'w') as f:
        f.write(latex_content)



print("#######################################################################################################")
print("SIMULATE POLE MOVEMENT")
print("#######################################################################################################")



alpha_array_dict = {}
delta_array_dict = {}
for sim_name in best_parameter_update.keys():
    print('Creating pole movement fig for ' + sim_name)
    estimated_values = best_parameter_update[sim_name]
    est_parameters = simulations[sim_name]['est_parameters']
    parameters_indicies = est_parameters_indicies[sim_name]

    models_Jacobson = ['initial_state_only_Pole_Jacbson2009', 
                        'rot_model_pos_only_Pole_Jacobson2009',
                        'rot_model_rate_only_Pole_Jacobson2009',
                        'rot_model_full_Pole_Jacobson2009',
                        'pole_libration_amplitude_deg1_Pole_Jacobson2009',
                        'pole_libration_amplitude_deg2_Pole_Jacobson2009', 
                        'pole_pos_and_libration_amplitude_Pole_Jacobson2009',
                        'pole_full_and_libration_amplitude_Pole_Jacobson2009']
    
    #if sim_name in models_Jacobson:
    model_type = 'Jacobson2009'
    #else:
    #    model_type = 'IAU'

    parameter_update = [0,0,0,0,0,0,0,0]
    if 'pole_position' in parameters_indicies:
        index = parameters_indicies.index('pole_position')
        parameter_update[0:2] = estimated_values[index:index+2]
    if 'pole_rate' in parameters_indicies:
        index = parameters_indicies.index('pole_rate')
        parameter_update[2:4] = estimated_values[index:index+2]
    if 'pole_lib' in parameters_indicies:
        index = parameters_indicies.index('pole_lib')
        parameter_update[4:6] = estimated_values[index:index+2]
    if 'pole_lib_deg2' in parameters_indicies:
        index = parameters_indicies.index('pole_lib_deg2')
        parameter_update[6:8] = estimated_values[index:index+2]
    # alpha_0, delta_0
    # alpha_dot_0, delta_dot_0
    # alpha_dot_1, delta_dot_1
    # alpha_dot_2, delta_dot_2
    alpha_array_dict[sim_name],delta_array_dict[sim_name] = PropFuncs.PoleModel(time_column,parameter_update,model_type)

    fig = FigUtils.plot_pole_movement(time_column,alpha_array_dict[sim_name],delta_array_dict[sim_name],title=('Pole Movement vs Time ' + sim_name))

    fig.savefig(out_dir / ("pole_movement_" + sim_name + ".pdf"))
    #fig_RSW.savefig(out_dir / ("RSW Diff SPICE - " + key + ".pdf"))



# file_path = 'Results/PoleEstimationRealObservations/UltimateAnalysis/simulations_with_weights.pkl'

# with open(file_path, 'rb') as f:
#     simulations = pickle.load(f)



#Merge other dicts
for name in simulations.keys():
    simulations[name]['pole_alpha'] = alpha_array_dict[name]
    simulations[name]['pole_delta'] = delta_array_dict[name]
    simulations[name]['initial_state_diff_SPICE'] = best_parameter_update[name]
with open(out_dir / 'simulations_with_weights.pkl', 'wb') as f:
    pickle.dump(simulations, f)


print("End.")



##




