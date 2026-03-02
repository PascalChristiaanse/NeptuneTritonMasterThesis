import numpy as np
from pathlib import Path
from typing import Dict
from datetime import datetime, timedelta
import sys
import json
import pandas as pd
import matplotlib
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm

import pickle
import os 
import yaml

from datetime import datetime, timedelta
import matplotlib.dates as mdates


from tudatpy.interface import spice

from tudatpy.astro import time_conversion, element_conversion,frame_conversion
from tudatpy.astro.time_conversion import DateTime
from tudatpy import numerical_simulation
import tudatpy
from tudatpy import util
from tudatpy.dynamics import parameters_setup
from tudatpy.dynamics import simulator

from tudatpy.estimation import estimation_analysis

# Get the path to the directory containing this file
current_dir = Path(__file__).resolve().parent

# Append the HelperFunctions directory
sys.path.append(str(current_dir / "HelperFunctions"))

import MainPostprocessing as PostProc


import ProcessingUtils
import PropFuncs
import FigUtils
import ObsFunc
import nsdc


J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)


def make_timestamped_folder(base_path="Results"):
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = Path(base_path) / folder_name
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path

def RunSinglePropagation(settings: dict, out_dir,folder_name=None):

    ##############################################################################################
    # CREATE ENVIRONMENT  
    ##############################################################################################

    body_settings,system_of_bodies = PropFuncs.Create_Env(settings["env"])

    ##############################################################################################
    # CREATE ACCELERATION MODELS  
    ##############################################################################################

    acceleration_models,accelerations_cfg = PropFuncs.Create_Acceleration_Models(settings["acc"],system_of_bodies)

    ##############################################################################################
    # CREATE PROPAGATOR
    ##############################################################################################

    propagator_settings = PropFuncs.Create_Propagator_Settings(settings["prop"],acceleration_models)

    ##############################################################################################
    # CREATE ESTIMATION PARAMETERS SETTINGS  
    ##############################################################################################
    
    # Create initial state variation equation
    parameters_to_estimate_settings = parameters_setup.initial_states(propagator_settings, system_of_bodies)

    #Only use with IAU rotational model
    if 'iau_rotation_model_pole' in settings['est']["est_parameters"]:
        parameters_to_estimate_settings.append(parameters_setup.iau_rotation_model_pole('Neptune'))
    if 'iau_rotation_model_pole_rate' in settings['est']['est_parameters']:
        parameters_to_estimate_settings.append(parameters_setup.iau_rotation_model_pole_rate('Neptune'))
    if 'iau_rotation_model_pole_librations' in settings['est']['est_parameters']:
        if settings['env']['Neptune_rot_model_type'] == 'Pole_Model_Jacobson2009':
            w_n_1 = np.deg2rad(52.3836218446110/36525/24/3600)
            w_n_2 = 2*np.deg2rad(52.3836218446110/36525/24/3600)
            if 'pole_librations_deg1' in settings['est']['est_parameters']:
                parameters_to_estimate_settings.append(parameters_setup.iau_rotation_model_pole_librations('Neptune',[w_n_1]))
            elif 'pole_librations_deg2' in settings['est']['est_parameters']:
                parameters_to_estimate_settings.append(parameters_setup.iau_rotation_model_pole_librations('Neptune',[w_n_1,w_n_2]))
        elif settings['env']['Neptune_rot_model_type'] == 'IAU2015':
            w_n_i = np.deg2rad(52.316/36525/24/3600)
            parameters_to_estimate_settings.append(parameters_setup.iau_rotation_model_pole_librations('Neptune',[w_n_i]))
    



    # Create the parameters that will be estimated
    parameters_to_estimate = parameters_setup.create_parameter_set(parameters_to_estimate_settings, system_of_bodies)

    print("################################################################################")
    print("START OF SIMULATION")
    print("################################################################################")
    
    # dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    # system_of_bodies, propagator_settings)
   
    # Create the variational equation solver and propagate the dynamics
    variational_equations_solver = simulator.create_variational_equations_solver(
        system_of_bodies, propagator_settings, parameters_to_estimate, simulate_dynamics_on_creation=True
    )

    print("################################################################################")
    print("END OF SIMULATION")
    print("################################################################################")

    ##############################################################################################
    # SAVE PROPAGATION RESULTS  
    ##############################################################################################
    
    # state_history = dynamics_simulator.propagation_results.state_history
    # state_history_array = util.result2array(state_history)
    
    # dep_vars_history = dynamics_simulator.propagation_results.dependent_variable_history
    # dep_vars_history_array = util.result2array(dep_vars_history)


    # Extract the resulting state history, state transition matrix history, and sensitivity matrix history
    state_history = variational_equations_solver.state_history
    state_transition_interface = variational_equations_solver.state_transition_interface
    
    state_history_array = util.result2array(state_history)

    output_times = state_history_array[:,0]
    # Define the linear variation in the initial state
    #initial_state_variation = settings["prop"]["initial_state_uncertanity"] 
    initial_covariance = settings["prop"]["initial_covariance"]

    delta_initial_state_dict = dict()

    #Propagate the covariancees and the formal errors
    propagated_covariances = estimation_analysis.propagate_covariance_split_output(
            initial_covariance,
            state_transition_interface,
            output_times)
    # # Propagate formal errors over the course of the orbit
    # propagated_formal_errors = estimation_analysis.propagate_formal_errors_split_output(
    #     initial_covariance=initial_covariance,
    #     state_transition_interface=state_transition_interface,
    #     output_times=output_times)
   
    # Split tuple into epochs and formal errors
    #epochs = np.array(propagated_formal_errors[0])
    #formal_errors1 = np.array(propagated_formal_errors[1])
    covariances = np.array(propagated_covariances[1])


    if folder_name is not None:
        out_dir = out_dir / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)


    # Save residuals as numpy files

    np.save(out_dir / "state_history_array.npy", state_history_array)


    #np.save(out_dir / "delta_initial_state_array.npy", delta_initial_state_array)
    #np.save(out_dir / "state_transition_matrices.npy", state_transition_matrices)
    np.save(out_dir / "covariances_array.npy",covariances)
    #np.save(out_dir / "formal_errors_array.npy",formal_errors1)

    #Convert initial state to list to be able to save to yaml
    settings_copy = settings.copy()
    settings_copy["prop"]["initial_state"] = settings_copy["prop"]["initial_state"].tolist()
    settings_copy['prop']['initial_covariance'] = settings_copy['prop']['initial_covariance'].tolist()
    #settings["prop"]["initial_state_uncertanity"] = settings["prop"]["initial_state_uncertanity"].tolist()
    settings_copy['obs'].pop('weights', None)

    if 'initial_Pole_Pos' in settings_copy['env']:
        settings_copy['env']['initial_Pole_Pos'] = settings_copy['env']['initial_Pole_Pos'].tolist()
    if 'initial_Pole_lib_deg1' in settings_copy['env']:
        settings_copy['env']['initial_Pole_lib_deg1'] = settings_copy['env']['initial_Pole_lib_deg1'].tolist()
    #Save yaml settings file
    with open(out_dir / "settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(settings_copy, f, sort_keys=False, allow_unicode=True)
    
    return state_history_array,covariances


if __name__ == "__main__":
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



    # Define temporal scope of the simulation - equal to the time JUICE will spend in orbit around Jupiter
    simulation_start_epoch = DateTime(1963, 1,  1).epoch() #2006, 8,  27 1963, 3,  4  
    simulation_end_epoch   = DateTime(2025, 1, 1).epoch()   #2006, 9, 2 2019, 10, 1
    
    simulation_initial_epoch = DateTime(2006, 10, 1).epoch()
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
    

    settings = dict()
    settings["env"] = settings_env
    settings["acc"] = settings_acc
    settings["prop"] = settings_prop
    
    ################################################################################################
    # Load initial states and other stuff 
    ################################################################################################
    
    out_dir = make_timestamped_folder("Results/AnalysisComparison")
    
        
    
    #Test if simulation answers loaded files
    ValidatedSim = False
    if ValidatedSim == True:
        state_history_loaded = sim1['state_history_array']
        state_history_simulated = state_history_array1
        time_column = state_history_loaded[:, [0]]  
        
        states_loaded_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_loaded[:,1:4], state_history_loaded)
        states_simulated_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_simulated[:,1:4], state_history_loaded)

        diff_RSW = (states_loaded_RSW - states_simulated_RSW)/1e3

        fig_final_sim_SPICE_rsw = FigUtils.Residuals_RSW(diff_RSW, time_column,type="difference",title="RSW Difference Loaded vs Simulated")
        fig_final_sim_SPICE_rsw.savefig(sim1_path / "RSW_Diff_with_without_weights.pdf")


    runSimulationsBetter = True
    if runSimulationsBetter == True:
        # Define all experiment variants here only once
        VARIANTS = {
            "no_weights": {
                "estimation_path": "Results/AllModernObservations/First",
                "simulation_path": "Results/AnalysisComparison/Sims/no_weights",
            },
            "id_rmse_weights": {
                "estimation_path": "Results/RMSE_Weights/ID_weights",
                "simulation_path": "Results/AnalysisComparison/Sims/id_rmse_weights",
            },
            "id_rmse_weights_descaled": {
                "estimation_path": "Results/RMSE_Weights/ID_weights_descaled",
                "simulation_path": "Results/AnalysisComparison/Sims/id_rmse_weights_descaled",
            },
            "hybrid_weights_descaled": {
                "estimation_path": "Results/RMSE_Weights/Hybrid_weights_descaled",
                "simulation_path": "Results/AnalysisComparison/Sims/hybrid_weights_descaled",
            },
            "id_weights_std_descaled": {
                "estimation_path": "Results/RMSE_Weights/ID_weights_std_descaled",
                "simulation_path": "Results/AnalysisComparison/Sims/id_weights_std_descaled",
            },
            "id_weights_std": {
                "estimation_path": "Results/AllModernObservations/Weights",
                "simulation_path": "Results/AnalysisComparison/Sims/id_weights_std",
            },
            "tf_weights_50mas_descaled": {
                "estimation_path": "Results/AllModernObservations/PerNightTimeframes",
                "simulation_path": "Results/AnalysisComparison/Sims/tf_weights_50mas_descaled",
            },                                     
        }


        # Load estimations
        estimations = {
            name: PostProc.load_npy_files(cfg["estimation_path"])
            for name, cfg in VARIANTS.items()
        }

        runSim = False
        if runSim == True:
            results = {}
            for name, est in estimations.items():

                settings["prop"]["initial_state"] = est["final_paramaters"]
                settings["prop"]["initial_covariance"] = est["covariance"]

                _, sim_array = RunSinglePropagation(settings, out_dir, name)
                results[name] = sim_array


        # Load simulations
        simulations = {
            name: PostProc.load_npy_files(cfg["simulation_path"])
            for name, cfg in VARIANTS.items()
        }




    print("#######################################################################################################")
    print("CREATE SPICE STATES")
    print("#######################################################################################################")

    # GET SPICE Results (any sim should work)
    epochs = simulations['no_weights']['state_history_array'][:,0]
    states_SPICE = np.array([
        spice.get_body_cartesian_state_at_epoch(
            target_body_name="Triton",
            observer_body_name="Neptune",
            reference_frame_name="ECLIPJ2000",
            aberration_corrections="NONE",
            ephemeris_time=epoch
        )
        for epoch in epochs
    ])
    
    time_column = simulations['no_weights']['state_history_array'][:,[0]]  # keep it 2D 
    states_SPICE_with_time = np.hstack((time_column, states_SPICE))

    print("#######################################################################################################")
    print("CREATE RSW STATES FIGS")
    print("#######################################################################################################")

    diff_SPICE_RSW = {}
    max_formal_errors_RSW = {}
    max_formal_errors = {}
    initial_formal_errors = {}
    number_of_matches = {}

    for key in simulations.keys():
        state_history_array = simulations[key]['state_history_array']

        states_SPICE_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, states_SPICE[:,0:3], state_history_array)
        states_sim_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array[:,1:4], state_history_array)

        diff = (states_SPICE_RSW - states_sim_RSW)/1e3
        diff_SPICE_RSW[key] = diff 
        fig_RSW = FigUtils.Residuals_RSW(diff, time_column,type="difference",title=("RSW Difference SPICE - " + key))   
        fig_RSW.savefig(out_dir / ("RSW Diff SPICE - " + key + ".pdf"))

        #---------------------------------------------------------------------------------------
        #Plot formal errors
        final  = estimations[key]['final_paramaters']      # note the key name as provided
        initial = estimations[key]['initial_paramaters']
        sigma   = estimations[key]['formal_errors']

        fig_formal_errors = PostProc.plot_formal_errors(initial,final,sigma)
        fig_formal_errors.savefig(out_dir / ("Formal_Errors_" + key + ".pdf"))

        # Plot propagated formal errors
        formal_errors = simulations[key]['formal_errors_array']/1e3
        # formal_errors_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, formal_errors[:,0:3], state_history_array)
        
        Covariances_array = simulations[key]['covariances_array']
        Covariances_array_RSW = simulations[key]['covariances_array_rsw']
        
        # RECOMPUTING RSW ROTATION TAKES TIME
        #-----------------------------------------------------------
        #Covariances_array_RSW = ProcessingUtils.rotate_covariance_inertial_to_rsw(time_column, Covariances_array, state_history_array)
        #sim_dir = Path('Results/AnalysisComparison/Sims')
        #np.save(sim_dir / key / "covariances_array_rsw.npy",Covariances_array_RSW)
        #-----------------------------------------------------------
        
        diag = np.diagonal(Covariances_array_RSW, axis1=1, axis2=2)

        formal_errors_RSW = np.sqrt(diag[:,0:3])/1e3
        
       

        fig_RSW = FigUtils.Residuals_RSW(formal_errors_RSW, time_column,type="difference",title=("RSW Formal Errors " + key))   
        fig_RSW.savefig(out_dir / ("Formal_Errors_Propagated_RSW" + key + ".pdf"))

        fig_RSW = FigUtils.Residuals_RSW(formal_errors[:,0:3], time_column,type="difference",title=("NOT RSW Formal Errors " + key))   
        fig_RSW.savefig(out_dir / ("Formal_Errors_Propagated_NOT_RSW" + key + ".pdf"))



        #Save formal error values per key
        max_formal_errors_RSW[key] = np.max(formal_errors_RSW, axis=0)
        max_formal_errors[key] = np.max(formal_errors,axis=0)
        initial_formal_errors[key] = sigma[0:3]/1e3

        #Compute number of coinciding initial directions wrt NEP097
        pos_diff, vel_diff = (final[:3] - initial[:3])/1e3, final[3:] - initial[3:]
        pos_err, vel_err = sigma[:3]/1e3, sigma[3:]
        number_of_matches[key] = np.sum(abs(pos_err)>abs(pos_diff))+ np.sum(abs(vel_err)>abs(vel_diff))

        #---------------------------------------------------------------------------------------

    diff_wrt_norm_RSW = {}
    number_of_matches_norm = {}
    state_history_array_norm = simulations['id_rmse_weights_descaled']['state_history_array']
    states_norm_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array_norm[:,1:4], state_history_array_norm)
    parameters_norm  = estimations['id_rmse_weights_descaled']['final_paramaters'] 
    for key in simulations.keys():
        if key is not 'id_rmse_weights_descaled':
            state_history_array = simulations[key]['state_history_array']
            states_sim_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array[:,1:4], state_history_array_norm)

            parameters_sim  = estimations[key]['final_paramaters']      # note the key name as provided
            sigma   = estimations[key]['formal_errors']

            pos_diff, vel_diff = (parameters_norm[:3] - parameters_sim[:3])/1e3, parameters_norm[3:] - parameters_sim[3:]
            pos_err, vel_err = sigma[:3]/1e3, sigma[3:]
            number_of_matches_norm[key] = np.sum(abs(pos_err)>abs(pos_diff))+ np.sum(abs(vel_err)>abs(vel_diff))


            diff = (states_norm_RSW- states_sim_RSW)/1e3
            diff_wrt_norm_RSW[key] = diff 
            fig_RSW = FigUtils.Residuals_RSW(diff, time_column,type="difference",title=("RSW Difference Norm - " + key))   
            fig_RSW.savefig(out_dir / ("RSW Diff Norm_ " + key + ".pdf"))


            




    print("#######################################################################################################")
    print("COMPUTE UNCERTANITY IN RSW")
    print("#######################################################################################################")
    
    #uncertanity_states = simulations["timeframe_weights_10mas_min"]['delta_initial_state_array']/1e3


    print("#######################################################################################################")
    print("CREATE MEAN and STD FIG")
    print("#######################################################################################################")
    #Calculated mean and stds
    stats = {}

    for key, arr in diff_SPICE_RSW.items():
        stats[key] = {
            "mean": arr.mean(axis=0),
            "std":  arr.std(axis=0),
            "max_formal_errors_RSW": np.array([0,0,0]) if key == "no_weights" else max_formal_errors_RSW[key],
            "max_formal_errors":      np.array([0,0,0])  if key == "no_weights" else max_formal_errors[key],
            "initial_formal_errors":  np.array([0,0,0])  if key == "no_weights" else initial_formal_errors[key],
            "nr_matches_NEP": number_of_matches[key]
        }

    #Calculated mean and stds
    stats_norm = {}

    for key, arr in diff_wrt_norm_RSW.items():
        stats_norm[key] = {
            "mean": arr.mean(axis=0),
            "std":  arr.std(axis=0),
            "max_formal_errors_RSW": np.array([0,0,0]) if key == "no_weights" else max_formal_errors_RSW[key],
            "max_formal_errors":      np.array([0,0,0])  if key == "no_weights" else max_formal_errors[key],
            "initial_formal_errors":  np.array([0,0,0])  if key == "no_weights" else initial_formal_errors[key],
            "nr_matches_NEP": number_of_matches_norm[key]
        }
    


    def plot_means_and_stds(stats_dict):
        keys = list(stats_dict.keys())
        
        # Extract means and stds in R,S,W order
        # means_R = [stats_dict[k]["mean"][0] for k in keys]
        # means_S = [stats_dict[k]["mean"][1] for k in keys]
        # means_W = [stats_dict[k]["mean"][2] for k in keys]

        # stds_R = [stats_dict[k]["std"][0] for k in keys]
        # stds_S = [stats_dict[k]["std"][1] for k in keys]
        # stds_W = [stats_dict[k]["std"][2] for k in keys]

        def extract_3_entries(row_key,keys):
            # Extract means and stds in R,S,W order
            entry_R = [stats_dict[k][row_key][0] for k in keys]
            entry_S = [stats_dict[k][row_key][1] for k in keys]
            entry_W = [stats_dict[k][row_key][2] for k in keys]
            return entry_R,entry_S,entry_W

        means_R,means_S,means_W = extract_3_entries("mean",keys)
        stds_R,stds_S,stds_W = extract_3_entries("std",keys)
        max_fe_R,max_fe_S,max_fe_W = extract_3_entries("max_formal_errors_RSW",keys)
        max_fe_X,max_fe_Y,max_fe_Z = extract_3_entries("max_formal_errors",keys)
        init_fe_X,init_fe_Y,init_fe_Z = extract_3_entries("initial_formal_errors",keys)
        nr_matches_NEP = [stats_dict[k]['nr_matches_NEP'] for k in keys]


        fig, axes = plt.subplots(3, 4, figsize=(14, 10), sharex=True)

        # Row/Column labels
        ylabels = ["R [km]", "S [km]", "W [km]"]

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = (colors + colors + colors)[:3]  # ensure at least 3


        # --- 1st COLUMN: MEANS ---
        axes[0,0].plot(keys, means_R, marker='o',color=colors[0])
        axes[1,0].plot(keys, means_S, marker='o',color=colors[1])
        axes[2,0].plot(keys, means_W, marker='o',color=colors[2])

        # --- 2nd COLUMN: STDs ---
        axes[0,1].plot(keys, stds_R, marker='o',color=colors[0])
        axes[1,1].plot(keys, stds_S, marker='o',color=colors[1])
        axes[2,1].plot(keys, stds_W, marker='o',color=colors[2])


        # --- 3rd COLUMN: Max FE RSW ---
        axes[0,2].plot(keys, max_fe_R, marker='o',color=colors[0])
        axes[1,2].plot(keys, max_fe_S, marker='o',color=colors[1])
        axes[2,2].plot(keys, max_fe_W, marker='o',color=colors[2])
        axes[0,2].set_title("Maxium Formal Error RSW")

        # # --- 3rd COLUMN: Max FE RSW ---
        # axes[0,3].plot(keys, max_fe_X, marker='o',color=colors[0])
        # axes[1,3].plot(keys, max_fe_Y, marker='o',color=colors[1])
        # axes[2,3].plot(keys, max_fe_Z, marker='o',color=colors[2])
        # axes[0,3].set_title("Maxium Formal Error Inertial")

        # # --- 3rd COLUMN: Max FE RSW ---
        # axes[0,4].plot(keys, init_fe_X, marker='o',color=colors[0])
        # axes[1,4].plot(keys, init_fe_Y, marker='o',color=colors[1])
        # axes[2,4].plot(keys, init_fe_Z, marker='o',color=colors[2])
        # axes[0,4].set_title("Initial Formal Error Inertial")

         # --- 3rd COLUMN: Matches with NEP ---
        axes[0,3].plot(keys, nr_matches_NEP, marker='o',color=colors[0])
        axes[0,3].axhline(6,linestyle='--',color='red',alpha=0.6,label="Max Possible")
        axes[0,3].set_title("Number of matches with NEP097")
        axes[0,3].legend()

        # Apply labels and grid lines
        for i in range(3):
            axes[i,0].set_ylabel(ylabels[i])
            for j in range(3):
                axes[i,j].grid(True, linestyle='--', alpha=0.6)
                #axes[i,1].grid(True, linestyle='--', alpha=0.6)

        # Fix x labels
        for ax in axes[2]:
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=45, ha='right')

        axes[0,0].set_title("Mean of Difference wrt SPICE RSW")
        axes[0,1].set_title("STD of Difference wrt SPICE RSW")
    

        fig.tight_layout()
        return fig


    def plot_means_and_stds_rotated(stats_dict):
        """
        Plot means, stds, and max formal errors with file IDs on Y-axis (rotated layout).
        
        Parameters:
        -----------
        stats_dict : dict
            Dictionary with keys as observation IDs and values containing statistics
        
        Returns:
        --------
        fig : matplotlib figure
        """
        keys = list(stats_dict.keys())
        
        def extract_3_entries(row_key, keys):
            # Extract means and stds in R,S,W order
            entry_R = [stats_dict[k][row_key][0] for k in keys]
            entry_S = [stats_dict[k][row_key][1] for k in keys]
            entry_W = [stats_dict[k][row_key][2] for k in keys]
            return entry_R, entry_S, entry_W
        
        means_R, means_S, means_W = extract_3_entries("mean", keys)
        stds_R, stds_S, stds_W = extract_3_entries("std", keys)
        max_fe_R, max_fe_S, max_fe_W = extract_3_entries("max_formal_errors_RSW", keys)
        nr_matches_NEP = [stats_dict[k]['nr_matches_NEP'] for k in keys]

        # Create figure with 3 rows (one per component), 3 columns (mean, std, max_fe)
        fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharey=True)
        
        # Component labels
        xlabels = ["R [km]", "S [km]", "W [km]"]
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = (colors + colors + colors)[:3]  # ensure at least 3
        
        # Y-axis positions for file IDs
        y = np.arange(len(keys))
        
        # --- 1st COLUMN: MEANS ---
        axes[0, 0].plot(means_R, y, marker='o', color=colors[0])
        axes[1, 0].plot(means_S, y, marker='o', color=colors[1])
        axes[2, 0].plot(means_W, y, marker='o', color=colors[2])
        
        # --- 2nd COLUMN: STDs ---
        axes[0, 1].plot(stds_R, y, marker='o', color=colors[0])
        axes[1, 1].plot(stds_S, y, marker='o', color=colors[1])
        axes[2, 1].plot(stds_W, y, marker='o', color=colors[2])
        
        # --- 3rd COLUMN: Max FE RSW ---
        axes[0, 2].plot(max_fe_R, y, marker='o', color=colors[0])
        axes[1, 2].plot(max_fe_S, y, marker='o', color=colors[1])
        axes[2, 2].plot(max_fe_W, y, marker='o', color=colors[2])
        
        # Set column titles (top row only)
        axes[0, 0].set_title("Mean of Difference wrt SPICE RSW", fontsize=14)
        axes[0, 1].set_title("STD of Difference wrt SPICE RSW", fontsize=14)
        axes[0, 2].set_title("Maximum Formal Error RSW", fontsize=14)

         # --- 4th COLUMN: Matches with NEP ---
        axes[0,3].plot(nr_matches_NEP,y, marker='o',color=colors[0])
        #axes[0,3].axhline(6,linestyle='--',color='red',alpha=0.6,label="Max Possible")
        axes[0,3].axvline(x=6, color='red', linestyle='--', alpha=0.5,label="Max Possible")
        axes[0,3].set_title("Number of matches with NEP097")
        axes[0,3].legend()


        # Apply labels and grid lines
        for i in range(3):
            # Set X labels for bottom row
            axes[i, 0].set_xlabel(xlabels[i], fontsize=12)
            axes[i, 1].set_xlabel(xlabels[i], fontsize=12)
            axes[i, 2].set_xlabel(xlabels[i], fontsize=12)
            
            # Add grid to all subplots
            for j in range(3):
                axes[i, j].grid(True, linestyle='--', alpha=0.6)
                
                # Add zero line for means (first column only)
                if j == 0:
                    axes[i, j].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Set Y-axis labels (shared across all columns, only show on leftmost)
        for i in range(3):
            axes[i, 0].set_yticks(y)
            axes[i, 0].set_yticklabels(keys, fontsize=10)
        
        # Add row labels on the right side to indicate component
        for i, label in enumerate(['R Component', 'S Component', 'W Component']):
            axes[i, 2].text(1.15, 0.5, label, transform=axes[i, 2].transAxes,
                        fontsize=12, fontweight='bold', rotation=270, 
                        va='center', ha='left')
        
        fig.tight_layout()
        
        return fig


    fig = plot_means_and_stds(stats)
    fig.savefig(out_dir / f"stats_comparison.pdf")
    plt.close(fig)    

    fig = plot_means_and_stds_rotated(stats)
    fig.savefig(out_dir / f"stats_comparison_rotated.pdf")
    plt.close(fig)
    
    
    fig = plot_means_and_stds(stats_norm)
    fig.savefig(out_dir / f"stats_norm_comparison.pdf")
    plt.close(fig)    

    fig = plot_means_and_stds_rotated(stats_norm)
    fig.savefig(out_dir / f"stats_norms_comparison_rotated.pdf")
    plt.close(fig)
    
    
    
    print('end')
