
import os
import yaml
import json
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from datetime import datetime
from pathlib import Path
import pandas as pd
# tudatpy imports
from tudatpy import math
from tudatpy import constants

from tudatpy.interface import spice
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
import tudatpy.estimation
from tudatpy import util
#import tudatpy.estimation_setup

#from tudatpy.numerical_simulation import estimation

#from tudatpy.numerical_simulation import estimation_setup #,Time


from tudatpy import numerical_simulation

from tudatpy.astro import time_conversion, element_conversion,frame_conversion
from tudatpy.astro.time_conversion import DateTime
from tudatpy.estimation.observable_models_setup import links

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


def CASE1_Manual_Bias(settings,out_dir,file_path=""):
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

    out_dir = out_dir / "CASE1_Manual_Bias"
    out_dir.mkdir(parents=True, exist_ok=True)
    #------------------------
    fitted_pole_pos_lib_sim = PostProc.load_npy_files("Results/EstimatedParametersSimulatedObservations/PoleLibrations/pole_pos_and_libration_amplitude")

    settings['prop']['initial_state'] =  fitted_pole_pos_lib_sim['parameter_history'][0:6,-1]    

    pole_params_SimPole = fitted_pole_pos_lib_sim['parameter_history'][6:,-1]
    runSim = True
    if runSim == True :
        #Load observations with weights
        print("#######################################################################################################")
        print("RUN INITIAL ESTIMATION AND CREATE WEIGHTS SIM POLE") # LOAD OBSERVATIONS WITH WEIGHTS")
        print("#######################################################################################################")
        
        settings['env']['initial_Pole_Pos'] = pole_params_SimPole[0:2]
        settings['env']['initial_Pole_lib_deg1'] = pole_params_SimPole[2:4]

        out_dir_current = out_dir / 'SimPole_initial_state_no_weights'
        out_dir_current.mkdir(parents=True, exist_ok=True)
        
        # Set estimation parameters
        settings['est']['est_parameters'] = VARIANTS['SimPole_initial_state_no_weights']['est_parameters'] 
        
        #Set covariances if any
        
        # Set if to use a priori cov or not
        settings['est']['a_priori_covariance'] = False #VARIANTS['SimPole_initial_state_no_weights']['use_apriori_cov']

        settings['est']['a_priori_pole'] = False
        settings['est']['a_priori_lib'] = False
            

        estimation_output,observations,observations_settings,body_settings,system_of_bodies = ObservationImplementation.main(
                settings,
                out_dir_current)


        #First estimation (initial_state only) without weights is used to generate weights

        #simulation_weights_path = "Results/PoleEstimationRealObservations/UltimateCASE1/SimPole_initial_state_no_weights"
        #-------------------------------------------------------------
        simulation = PostProc.load_npy_files(out_dir_current)
        residuals = simulation['residual_history_arcseconds'][-1]
        # Convert RA and DEC columns from arcseconds to radians
        residuals[:, 1] = residuals[:, 1] / (3600 * 180 / np.pi)  # RA
        residuals[:, 2] = residuals[:, 2] / (3600 * 180 / np.pi)  # DEC

        # #Create Environment 
        body_settings,system_of_bodies = PropFuncs.Create_Env(settings['env'])
        #Load observations
        observations,observations_settings,observation_set_ids, epochs_rejected = ObsFunc.LoadObservations(
                settings["obs"]["observations_folder_path"],
                system_of_bodies,
                settings['obs']["files"],
                Residual_filtering = settings["obs"]["residual_filtering"])
        #-------------------------------------------------------------
        bias_dict = {
        "689_nm0077": -0.2,  # arcsec
        }
        times_sec = observations.get_concatenated_observation_times()
        residuals_old = observations.get_concatenated_residuals()

        # observations and observations_biased are pointing in the same memory block
        # therefore observations are overwritten !!
        observations_biased, applied = ObsFunc.apply_dec_bias_to_observations(
            observations,
            observations_settings,
            system_of_bodies,
            bias_dict
        )

        residuals_new = observations_biased.get_concatenated_residuals()
        fig = ObsFunc.PlotResidualBiased(times_sec,residuals_old,residuals_new)
        fig.savefig(out_dir / "ManualBias_SimPole.pdf")
        #-------------------------------------------------------------

        # EXTRACT RESIDUALS FROM INITIAL SIM 
        # AND COMPUTE/ASSIGN WEIGHTS FROM THEM
        observations_SimPole, weights_info_SimPole = ObsFunc.compute_and_assign_weights(
            residuals=residuals,
            observations=observations_biased,
            gap_threshold_hours=4.0,
            min_obs_per_frame=1,
            weight_type = 'hybrid'
        )

        print("#######################################################################################################")
        print("RUN INITIAL ESTIMATION AND CREATE WEIGHTS IAU POLE") # LOAD OBSERVATIONS WITH WEIGHTS")
        print("#######################################################################################################")
        
        out_dir_current = out_dir / 'IAUPole_initial_state_no_weights'
        out_dir_current.mkdir(parents=True, exist_ok=True)
        
        #Remove Sim Pole settings
        settings['env'].pop('initial_Pole_Pos', None)
        settings['env'].pop('initial_Pole_lib_deg1', None)
        settings['prop'].pop('initial_state',None)

        # # Set estimation parameters
        settings['est']['est_parameters'] = VARIANTS['IAUPole_initial_state_no_weights']['est_parameters'] 
        
        # # Set if to use a priori cov or not
        settings['est']['a_priori_covariance'] = VARIANTS['IAUPole_initial_state_no_weights']['use_apriori_cov']

        settings['est']['a_priori_pole'] = False
        settings['est']['a_priori_lib'] = False
            
        estimation_output,observations,observations_settings,body_settings,system_of_bodies = ObservationImplementation.main(
                settings,
                out_dir_current)

        #Extract residuals
        #First estimation (initial_state only) without weights is used to generate weights
        simulation = PostProc.load_npy_files(out_dir_current)
        residuals = simulation['residual_history_arcseconds'][-1]
        # # Convert RA and DEC columns from arcseconds to radians
        residuals[:, 1] = residuals[:, 1] / (3600 * 180 / np.pi)  # RA
        residuals[:, 2] = residuals[:, 2] / (3600 * 180 / np.pi)  # DEC

        #Load Observations
        #--------------------------------------------------------------------------
        # #Create Environment 
        body_settings,system_of_bodies = PropFuncs.Create_Env(settings['env'])

        #Load observations
        observations,observations_settings,observation_set_ids, epochs_rejected = ObsFunc.LoadObservations(
                settings["obs"]["observations_folder_path"],
                system_of_bodies,
                settings['obs']["files"],
                Residual_filtering = settings["obs"]["residual_filtering"])
        #-------------------------------------------------------------

        #Assign manual bias
        #-------------------------------------------------------------
        bias_dict = {
        "689_nm0077": -0.2,  # arcsec
        }
        times_sec = observations.get_concatenated_observation_times()
        residuals_old = observations.get_concatenated_residuals()

        # observations and observations_biased are pointing in the same memory block
        # therefore observations are overwritten !!
        observations_biased, applied = ObsFunc.apply_dec_bias_to_observations(
            observations,
            observations_settings,
            system_of_bodies,
            bias_dict
        )

        residuals_new = observations_biased.get_concatenated_residuals()
        fig = ObsFunc.PlotResidualBiased(times_sec,residuals_old,residuals_new)
        fig.savefig(out_dir / "ManualBias_IAUPole.pdf")
        #--------------------------------------------------------------------------

        #COMPUTE & ASSIGN WEIGHTS FROM RESIDUALS OF INITIAL SIM
        observations_IAUPole, weights_info_IAUPole = ObsFunc.compute_and_assign_weights(
            residuals=residuals,
            observations=observations_biased,
            gap_threshold_hours=4.0,
            min_obs_per_frame=1,
            weight_type = 'hybrid'
        )

        print("#######################################################################################################")
        print("RUN WEIGHTED SIMULATIONS")
        print("#######################################################################################################")
    
        settings['env']['Neptune_rot_model_type'] ='IAU2015'

        for name, content in VARIANTS.items():
            if name != 'IAUPole_initial_state_no_weights' and name != 'SimPole_initial_state_no_weights':
                print("######################################")
                print("Running Sim ",name)
                print("######################################")
                
                out_dir_current = out_dir / name
                out_dir_current.mkdir(parents=True, exist_ok=True)
                
                estimation_type = name.split('_')[0]
                #Assign initial pole pos and lib based on estimation type
                if estimation_type == 'IAUPole':
                    settings['env'].pop('initial_Pole_Pos', None)
                    settings['env'].pop('initial_Pole_lib_deg1', None)
                    settings['prop'].pop('initial_state',None)
                elif estimation_type == 'SimPole':
                    settings['prop']['initial_state'] =  fitted_pole_pos_lib_sim['parameter_history'][0:6,-1]    
    
                    settings['env']['initial_Pole_Pos'] = pole_params_SimPole[0:2]
                    settings['env']['initial_Pole_lib_deg1'] = pole_params_SimPole[2:4]



                # Set estimation parameters
                settings['est']['est_parameters'] = content['est_parameters'] 
                
                #Set covariances if any
                
                # Set if to use a priori cov or not
                settings['est']['a_priori_covariance'] = content['use_apriori_cov']

                if 'pole_pos_cov' in content :
                    settings['est']['a_priori_pole'] = content['pole_pos_cov']
                else:
                    settings['est']['a_priori_pole'] = False
                
                if 'pole_lib_cov' in content :
                    settings['est']['a_priori_lib'] = content['pole_lib_cov']
                    settings['est']['a_priori_lib_deg'] = 1
                else:
                    settings['est']['a_priori_lib'] = False
                    


                #Run estimation + provide weights if needed
                if content['use_weights'] == True:
                    settings['obs']['use_loaded_obs'] = True 
                    settings['obs']['use_old_obs_func'] = False
                    settings["obs"]["manual_dec_bias"] = bias_dict
            
                    if estimation_type == 'IAUPole':
                        estimation_output,_,_,body_settings,system_of_bodies = ObservationImplementation.main(
                                settings,
                                out_dir_current,
                                observations=observations_IAUPole,
                                observations_settings=observations_settings)
                        weights_info_IAUPole.to_csv(out_dir_current / 'observation_weights.csv', index=False)
                    
                        
                    elif estimation_type == 'SimPole':
                        estimation_output,_,_,body_settings,system_of_bodies = ObservationImplementation.main(
                                settings,
                                out_dir_current,
                                observations=observations_SimPole,
                                observations_settings=observations_settings)
                        weights_info_SimPole.to_csv(out_dir_current / 'observation_weights.csv', index=False)     
                else:
                    estimation_output,observations,observations_settings,body_settings,system_of_bodies = ObservationImplementation.main(
                            settings,
                            out_dir_current)

    print("#######################################################################################################")
    print("END OF ESTIMATIONS")
    print("#######################################################################################################")

    return VARIANTS


def SimObs_ParameterAnalysis(settings,out_dir,folder_name=""):
    # #SIMULATED OBSERVATIONS
    VARIANTS = {
        "initial_state_IAU": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/initial_state_IAU",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "initial_state_Jacobson": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/initial_state_Jacobson",
            'est_parameters': ['initial_state'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "GM_Triton_IAU": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/GM_Triton_IAU",
            'est_parameters': ['initial_state', 'GM_Triton'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "GM_Neptune_IAU": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/GM_Neptune_IAU",
            'est_parameters': ['initial_state', 'GM_Neptune'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "GM_Both_IAU": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/GM_Both_IAU",
            'est_parameters': ['initial_state', 'GM_Triton', 'GM_Neptune'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "sh_IAU": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/sh_IAU",
            'est_parameters': ['initial_state', 'spherical_harmonics'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "pole_pos_IAU": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_pos_IAU",
            'est_parameters': ['initial_state', 'iau_rotation_model_pole'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "pole_rot_IAU": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_rot_IAU",
            'est_parameters': ['initial_state', 'iau_rotation_model_pole_rate'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "pole_pos_rot_IAU": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_pos_rot_IAU",
            'est_parameters': ['initial_state', 'iau_rotation_model_pole', 'iau_rotation_model_pole_rate'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "pole_lib_IAU": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_lib_IAU",
            'est_parameters': ['initial_state', 'iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "pole_full_IAU": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_full_IAU",
            'est_parameters': ['initial_state', 'iau_rotation_model_pole', 'iau_rotation_model_pole_rate', 'iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "pole_pos_Jacobson": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_pos_Jacobson",
            'est_parameters': ['initial_state', 'iau_rotation_model_pole'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "pole_rot_Jacobson": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_rot_Jacobson",
            'est_parameters': ['initial_state', 'iau_rotation_model_pole_rate'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "pole_lib1_Jacobson": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_lib1_Jacobson",
            'est_parameters': ['initial_state', 'pole_librations_deg2'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "pole_lib2_Jacobson": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_lib2_Jacobson",
            'est_parameters': ['initial_state', 'pole_librations_deg2'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "pole_full_Jacobson": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/pole_full_Jacobson",
            'est_parameters': ['initial_state', 'iau_rotation_model_pole', 'iau_rotation_model_pole_rate', 'pole_librations_deg2'],
            'Neptune_rot_model_type': 'Pole_Model_Jacobson2009',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "SH_pole_full_IAU": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/SH_pole_full_IAU",
            'est_parameters': ['initial_state', 'spherical_harmonics', 'iau_rotation_model_pole', 'iau_rotation_model_pole_rate', 'iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "GM_SH_IAU": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/GM_SH_IAU",
            'est_parameters': ['initial_state', 'GM_Neptune', 'GM_Triton', 'spherical_harmonics'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
        "all_IAU": {
            "simulation_path": "Results/EstimatedParametersSimulatedObservations/NewFinal/all_IAU",
            'est_parameters': ['initial_state', 'GM_Triton', 'GM_Neptune', 'spherical_harmonics', 'iau_rotation_model_pole', 'iau_rotation_model_pole_rate', 'iau_rotation_model_pole_librations'],
            'Neptune_rot_model_type': 'IAU2015',
            'use_weights': False,
            'use_apriori_cov': False,
        },
    }


    out_dir = out_dir / "SimObs_ParameterAnalysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    settings['obs']['type'] = 'Simulated'
    runSim = True
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

    print("######################################")
    print("END SIMULATIONS.")
    print("######################################")
    
    return VARIANTS


def WeightSchemeAnalysis(settings,out_dir,runSim=True,path_file=""):
    # WEIGHT SCHEME ANALYSIS
    VARIANTS = {
        "id_new_1_weights": {
            "simulation_path": "Results/EstimationTemplatesTest/WeightScheme/id_new_1_weights",
            'weight_type': 'id_new_1',
        },
        "id_new_2_weights": {
            "simulation_path": "Results/EstimationTemplatesTest/WeightScheme/id_new_2_weights",
            'weight_type': 'id_new_2',
        },
        "id_weights": {
            "simulation_path": "Results/EstimationTemplatesTest/WeightScheme/id_weights",
            'weight_type': 'id',
        },
        "tf_weights": {
            "simulation_path": "Results/EstimationTemplatesTest/WeightScheme/tf_weights",
            'weight_type': 'timeframe',
        },
        "tf_weights_no_limit": {
            "simulation_path": "Results/EstimationTemplatesTest/WeightScheme/tf_weights_no_limit",
            'weight_type': 'timeframe',
        },
        "hybrid_weights": {
            "simulation_path": "Results/EstimationTemplatesTest/WeightScheme/hybrid_weights",
            'weight_type': 'hybrid',
        },
        "hybrid_old_weights": {
            "simulation_path": "Results/EstimationTemplatesTest/WeightScheme/hybrid_old_weights",
            'weight_type': 'hybrid_old',
        },
        "hybrid_new_id_weights": {
            "simulation_path": "Results/EstimationTemplatesTest/WeightScheme/hybrid_new_id_weights",
            'weight_type': 'hybrid_new_id',
        },
        "hybrid_old_new_id_weights": {
            "simulation_path": "Results/EstimationTemplatesTest/WeightScheme/hybrid_old_new_id_weights",
            'weight_type': 'hybrid_old_new_id',
        },
    }



    out_dir = out_dir / "Weight_Scheme_Analysis"
    out_dir.mkdir(parents=True, exist_ok=True)


    #Create Environment 
    body_settings,system_of_bodies = PropFuncs.Create_Env(settings['env'])

    #Load observations
    observations,observations_settings,observation_set_ids, epochs_rejected = ObsFunc.LoadObservations(
            settings["obs"]["observations_folder_path"],
            system_of_bodies,
            settings['obs']["files"],
            Residual_filtering = settings["obs"]["residual_filtering"])

    path_CASE1_weights = "Results/PoleEstimationRealObservations/UltimateCASE1/IAUPole_initial_state_no_weights"
    simulation = PostProc.load_npy_files(path_CASE1_weights)

    residuals = simulation['residual_history_arcseconds'][-1]

    # Convert RA and DEC columns from arcseconds to radians
    residuals[:, 1] = residuals[:, 1] / (3600 * 180 / np.pi)  # RA
    residuals[:, 2] = residuals[:, 2] / (3600 * 180 / np.pi)  # DEC


    settings['obs']["use_loaded_obs"] = True
    tf_weights_clip_threshold_arcseconds = 0.01
    if runSim == True:
        results = {}
        for name, content in VARIANTS.items():
            print("######################################")
            print("Running Sim ",name)
            print("######################################")
            
            out_dir_current = out_dir / name
            out_dir_current.mkdir(parents=True, exist_ok=True)

            if name == "tf_weights_no_limit":
                tf_weights_clip_threshold_arcseconds = 0.0
            else:
                tf_weights_clip_threshold_arcseconds = 0.01

            # EXTRACT RESIDUALS FROM INITIAL SIM 
            # AND COMPUTE/ASSIGN WEIGHTS FROM THEM
            observations, weights_info = ObsFunc.compute_and_assign_weights(
                residuals=residuals,
                observations=observations,
                gap_threshold_hours=4.0,
                min_obs_per_frame=1,
                weight_type = content['weight_type'],
                min_sigma_arcsec = tf_weights_clip_threshold_arcseconds
            )

            #Run estimation
            estimation_output,_,_,body_settings,system_of_bodies = ObservationImplementation.main(
            settings,
            out_dir_current,
            observations=observations,
            observations_settings=observations_settings)


            weights_info.to_csv(out_dir_current / 'observation_weights.csv', index=False)

    print("######################################")
    print("END SIMULATIONS.")
    print("######################################")
    
    return VARIANTS



def WeightLoop(settings,out_dir,runSim=True,path_file=""):
    # WEIGHT SCHEME ANALYSIS
    VARIANTS = {
        "id_new_2_weights": {
            "simulation_path": "Results/EstimationTemplatesTest/WeightScheme/id_new_2_weights",
            'weight_type': 'id_new_2',
        },
        "tf_weights": {
            "simulation_path": "Results/EstimationTemplatesTest/WeightScheme/tf_weights",
            'weight_type': 'timeframe',
        },
    }



    out_dir = out_dir / "Weight_Loop"
    out_dir.mkdir(parents=True, exist_ok=True)


    #Create Environment 
    body_settings,system_of_bodies = PropFuncs.Create_Env(settings['env'])

    #Load observations
    observations,observations_settings,observation_set_ids, epochs_rejected = ObsFunc.LoadObservations(
            settings["obs"]["observations_folder_path"],
            system_of_bodies,
            settings['obs']["files"],
            Residual_filtering = settings["obs"]["residual_filtering"])

    path_CASE1_weights = "Results/PoleEstimationRealObservations/UltimateCASE1/IAUPole_initial_state_no_weights"
    simulation = PostProc.load_npy_files(path_CASE1_weights)

    residuals = simulation['residual_history_arcseconds'][-1]

    # Convert RA and DEC columns from arcseconds to radians
    residuals[:, 1] = residuals[:, 1] / (3600 * 180 / np.pi)  # RA
    residuals[:, 2] = residuals[:, 2] / (3600 * 180 / np.pi)  # DEC


    settings['obs']["use_loaded_obs"] = True
    if runSim == True:
        results = {}
        for name, content in VARIANTS.items():
            print("######################################")
            print("Running Sim ",name)
            print("######################################")
            
            out_dir_current = out_dir / name
            out_dir_current.mkdir(parents=True, exist_ok=True)
            
            # EXTRACT RESIDUALS FROM INITIAL SIM 
            # AND COMPUTE/ASSIGN WEIGHTS FROM THEM
            observations, weights_info = ObsFunc.compute_and_assign_weights(
                residuals=residuals,
                observations=observations,
                gap_threshold_hours=4.0,
                min_obs_per_frame=1,
                weight_type = content['weight_type']
            )
            for i in range(5):
                #create directory
                #Run estimation
                #Add estimation to VARIANTS
                #Run estimation
                estimation_output,_,_,body_settings,system_of_bodies = ObservationImplementation.main(
                settings,
                out_dir_current,
                observations=observations,
                observations_settings=observations_settings)


                weights_info.to_csv(out_dir_current / 'observation_weights.csv', index=False)

    print("######################################")
    print("END SIMULATIONS.")
    print("######################################")
    
    return VARIANTS