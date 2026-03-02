# General imports
#import math

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

# Get the path to the directory containing this file
current_dir = Path(__file__).resolve().parent

# Append the HelperFunctions directory
sys.path.append(str(current_dir / "HelperFunctions"))

import ProcessingUtils
import PropFuncs
import FigUtils
import ObsFunc
import nsdc

matplotlib.use("PDF")  #tkagg

def main(settings: dict,out_dir,
        body_settings=None,
        system_of_bodies=None,
        observations=None,
        observations_settings=None):
    """
    if settings['env']['use_created_env'] == True
        body_settings,system_of_bodies are required
    if settings['obs']['use_loaded_obs'] == True
        observation: ObservationCollection is required
        observations_settings: ObservationSettings is required 
    """
    print("Running Main File...")

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

    ##############################################################################################
    # CREATE ENVIRONMENT  
    ##############################################################################################
    #if settings['env']['use_created_env'] == False:
    body_settings,system_of_bodies = PropFuncs.Create_Env(settings['env'])
    # elif settings['env']['use_created_env'] == True:
    #     body_settings,system_of_bodies = PropFuncs.Create_Env(settings['env'],observations)

    
    simulation_start_epoch = settings['env']["start_epoch"]
    simulation_end_epoch = settings['env']["end_epoch"]
    step_size = settings['prop']['fixed_step_size']
    epochs = np.arange(simulation_start_epoch, simulation_end_epoch + step_size, step_size)

    # Get Neptune rotation model
    # nep_rot_model = system_of_bodies.get("Neptune").rotation_model

    # # Generate all rotation matrices at once
    # rotation_matrices = np.array([
    #     nep_rot_model.body_fixed_to_inertial_rotation(epoch) 
    #     for epoch in epochs
    # ])

    ##############################################################################################
    # CREATE ACCELERATION MODELS  
    ##############################################################################################

    acceleration_models,accelerations_cfg = PropFuncs.Create_Acceleration_Models(settings['acc'],system_of_bodies)

    ##############################################################################################
    # CREATE PROPAGATOR
    ##############################################################################################

    propagator_settings = PropFuncs.Create_Propagator_Settings(settings['prop'],acceleration_models)

    ##############################################################################################
    # CREATE PSEUDO OR LOAD REAL OBSERVATIONS
    ##############################################################################################
    if settings["obs"]["type"] == "Simulated":
        observations,observations_settings = PropFuncs.make_relative_position_pseudo_observations(
            simulation_start_epoch,simulation_end_epoch, system_of_bodies, settings)
    elif settings["obs"]["type"] == "Real":
        
        files = settings["obs"]["files"]
        observations_folder_path = settings["obs"]["observations_folder_path"]
        
        #If observations are already loaded and weights are assigned do not redo it.
        if settings['obs']['use_loaded_obs'] == True:
            #CREATE EARTH BASED STATIONS EARTH 
            #---------------------------------------------------------------------------------------------------------------------------------------------------
            if observations != None:
                Observatories = []
            
                # Extract observation sets
                sets = observations.sorted_observation_sets
                ObservableType = list(sets.keys())[0]
                
                all_observations_sets = []
                for observable_type, inner_dict in sets.items():
                    for link_end_id, observation_list in inner_dict.items():
                        all_observations_sets.extend(observation_list)
                
                for i in range(len(all_observations_sets)):
                    # Get reference point ID for this set
                    Observatories.append(all_observations_sets[i].link_definition.link_end_id(links.receiver).reference_point)
                

                for set_id in Observatories:
                    # Define the position of the observatory on Earth
                    observatory_longitude, observatory_latitude, observatory_altitude = ObsFunc.observatory_info(set_id.split("_")[0])

                    # Add the ground station to the environment
                    environment_setup.add_ground_station(
                    system_of_bodies.get_body("Earth"),
                    set_id,
                    [observatory_altitude, observatory_latitude, observatory_longitude],
                    element_conversion.geodetic_position_type)
                #---------------------------------------------------------------------------------------------------------------------------------------------------
        
        elif settings['obs']['use_old_obs_func'] == True: 
            observations,observations_settings,observation_set_ids,epochs_rejected = ObsFunc.LoadObservations(
                        observations_folder_path,
                        system_of_bodies,
                        files,
                        weights = settings["obs"]["weights"],
                        ra_dec_independent_weights = settings["obs"]["ra_dec_independent_weights"],
                        std_weights = settings["obs"]["std_weights"],
                        timeframe_weights = settings["obs"]['timeframe_weights'],
                        per_night_weights = settings["obs"]['per_night_weights'],
                        per_night_weights_id = settings["obs"]["per_night_weights_id"],
                        per_night_weights_hybrid = settings["obs"]["per_night_weights_hybrid"],
                        Residual_filtering = settings["obs"]["residual_filtering"],
                        epoch_filter_dict = settings["obs"]["epoch_filter_dict"])
        elif settings["obs"]["use_loaded_obs"] == False and settings['obs']['use_old_obs_func'] == False:
                
            observations,observations_settings,observation_set_ids, epochs_rejected = ObsFunc.LoadObservations(
            observations_folder_path,
            system_of_bodies,
            files,
            Residual_filtering = settings["obs"]["residual_filtering"]
            )


            bias_dict = {
            "689_nm0077": -0.2,  # arcsec
            }

            # observations and observations_biased are pointing in the same memory block
            # therefore observations are overwritten !!
            observations, applied = ObsFunc.apply_dec_bias_to_observations(
                observations,
                observations_settings,
                system_of_bodies,
                bias_dict
            )

            settings["obs"]["manual_dec_bias"] = bias_dict
            

            #Always save the rejected epochs for analysis
            #--------------------------------------------------------------
            output_file = out_dir / "residuals_rejected_epochs.json"

            with open(output_file, 'w') as f:
                json.dump(epochs_rejected, f, indent=2)
            #------------------------------------------------

    #else:



        Observatories = []

            
        # Extract observation sets
        sets = observations.sorted_observation_sets
        ObservableType = list(sets.keys())[0]
        
        all_observations_sets = []
        for observable_type, inner_dict in sets.items():
            for link_end_id, observation_list in inner_dict.items():
                all_observations_sets.extend(observation_list)
        
        for i in range(len(all_observations_sets)):
            # Get reference point ID for this set
            Observatories.append(all_observations_sets[i].link_definition.link_end_id(links.receiver).reference_point)
        


    else:
        print("No Observation type selected")
    ##############################################################################################
    # CREATE & RUN ESTIMATOR  
    ##############################################################################################


    estimation_output, original_parameter_vector, parameters_desc,covariances = PropFuncs.Create_Estimation_Output(settings,
    system_of_bodies,propagator_settings,observations_settings,observations)

    print("END OF ESTIMATION")

    output_file = out_dir / "estimation_summary.txt"

    with open(output_file, "w") as f:
        f.write(f"Original initial states:\n{original_parameter_vector}\n\n")
        f.write(
            "Estimated Parameters Descriptions are:\n"
            f"{parameters_desc}\n"
        )
    ##############################################################################################
    # RETRIEVE INFO
    ##############################################################################################

    state_history = estimation_output.simulation_results_per_iteration[-1].dynamics_results.state_history_float
    state_history_array = util.result2array(state_history)

    state_history_initial = estimation_output.simulation_results_per_iteration[0].dynamics_results.state_history_float
    state_history_initial_array = util.result2array(state_history_initial)

    dep_vars_history = estimation_output.simulation_results_per_iteration[-1].dynamics_results.dependent_variable_history
    dep_vars_array = util.result2array(dep_vars_history)

    #Get all results
    arrays = []
    for i in range(5):
        state_history_current = (
            estimation_output.simulation_results_per_iteration[i]
            .dynamics_results.state_history_float
        )
        arrays.append(util.result2array(state_history_current))

    state_history_array_full = np.stack(arrays, axis=0)
    ##############################################################################################
    # GET SPICE OBSERVATIONS
    ##############################################################################################

    fixed_step_size = settings['prop']["fixed_step_size"]
    
    # Get Triton's state relative to Neptune SPICE
    epochs = state_history_array[:,0]  #np.arange(simulation_start_epoch, simulation_end_epoch+60*5, fixed_step_size ) #test_settings_obs["cadence"]
    states_SPICE = np.array([
        spice.get_body_cartesian_state_at_epoch(
            target_body_name="Triton",
            observer_body_name="Neptune",
            reference_frame_name=settings['env']["global_frame_orientation"],
            aberration_corrections="NONE",
            ephemeris_time=epoch
        )
        for epoch in epochs
    ])
  

    ##############################################################################################
    # EXTRACTING RESIDUALS, PLOTTING AND SAVING DATA AND FIGS
    ##############################################################################################

    if settings['obs']['type'] == 'Real':
        print("Saving real observations residuals...")
        
        residual_history = ProcessingUtils.format_residual_history_abs_astrometric(
            estimation_output.residual_history,
            observations.get_concatenated_observation_times()
        )

        # residual_history_arcseconds = residual_history
        # for i in range(np.shape(residual_history)[0]):
        #     #residual_history_arcseconds.append(residual_history[i])
        #     residual_history_arcseconds[i][:,1:] = residual_history_arcseconds[i][:,1:]*3600 * 180/np.pi
        
        residual_history_arcseconds = [
        np.column_stack([arr[:,0], arr[:,1:] * 3600 * 180/np.pi]) 
        for arr in residual_history
        ]

        observation_times = observations.get_concatenated_observation_times()
        

        residuals_ra_initial_arcseconds = residual_history_arcseconds[0][:,1]
        residuals_dec_initial_arcseconds = residual_history_arcseconds[0][:,2]
        
        residuals_ra_arcseconds = residual_history_arcseconds[-1][:,1]
        residuals_dec_arcseconds = residual_history_arcseconds[-1][:,2]
        
        ##############################################################################################
        # RESIDUALS RA / DEC
        ##############################################################################################

        observation_times_DateFormat = FigUtils.ConvertToDateTime(observation_times)
                
        residuals = ObsFunc.Get_SPICE_residual_from_observations(observations,Observatories,system_of_bodies,settings['env']["global_frame_orientation"])
        residuals_RA_SPICE = residuals[0]
        residuals_DEC_SPICE = residuals[1]

        fig_estimation_residuals = FigUtils.plot_RA_DEC_residuals(
            observation_times_DateFormat,
            residuals_ra_arcseconds,
            residuals_ra_initial_arcseconds,
            residuals_dec_arcseconds,
            residuals_dec_initial_arcseconds)

        labels = ['final','SPICE']
        fig_SPICE_residuals = FigUtils.plot_RA_DEC_residuals(
            observation_times_DateFormat,
            residuals_ra_arcseconds,
            residuals_RA_SPICE,
            residuals_dec_arcseconds,
            residuals_DEC_SPICE,
            labels)


        #--------------------------------------------------------------------------------
        # Extract the time column (first column)
        time_column = state_history_array[:, [0]]   # keep it 2D (shape = (289, 1))
        states_SPICE_with_time = np.hstack((time_column, states_SPICE))

        fig_Cartesian = FigUtils.PlotCartesianDifference(state_history_array, states_SPICE_with_time)
        

        fig_rms = FigUtils.Residuals_RMS(residual_history_arcseconds)


        states_SPICE_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, states_SPICE[:,0:3], states_SPICE_with_time)
        states_sim_RSW = []
        states_sim_RSW.append(ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_initial_array[:,1:4], states_SPICE_with_time))
        states_sim_RSW.append(ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array[:,1:4], states_SPICE_with_time))

        diff_sim_RSW = (states_sim_RSW[0] - states_sim_RSW[-1])/1e3
        diff_sim_SPICE_RSW = (states_sim_RSW[0] - states_SPICE_RSW)/1e3
        diff_final_sim_SPICE_RSW = (states_sim_RSW[-1] - states_SPICE_RSW)/1e3

        fig_sim_RSW = FigUtils.Residuals_RSW(diff_sim_RSW, time_column,type="difference",title="RSW Difference Initial Final")
        fig_sim_SPICE_rsw = FigUtils.Residuals_RSW(diff_sim_SPICE_RSW, time_column,type="difference",title="RSW Difference Initial SPICE")
        fig_final_sim_SPICE_rsw = FigUtils.Residuals_RSW(diff_final_sim_SPICE_RSW, time_column,type="difference",title="RSW Difference Final SPICE")
            
        fig_sim_RSW_abs = FigUtils.Residuals_RSW(states_sim_RSW[-1], time_column,type="normal",title="RSW Final Simulation Absolute")
        fig_SPICE_RSW_abs = FigUtils.Residuals_RSW(states_SPICE_RSW, time_column,type="normal",title="RSW SPICE Absolute")
        #--------------------------------------------------------------------------------
        #save figs
        fig_rms.savefig(out_dir / "Residual_RMS.pdf")
        fig_sim_RSW.savefig(out_dir / "RSW_diff_Sim.pdf")
        fig_sim_SPICE_rsw.savefig(out_dir / "RSW_diff_initial_Sim_SPICE.pdf")
        fig_final_sim_SPICE_rsw.savefig(out_dir / "RSW_diff_final_Sim_SPICE.pdf")
        
        #fig_sim_RSW_abs.savefig(out_dir / "RSW_abs_final_Sim.pdf")
        #fig_SPICE_RSW_abs.savefig(out_dir / "RSW_abs_SPICE.pdf")
        
        fig_Cartesian.savefig(out_dir / "Cartesian_Difference_SPICE.pdf")
        fig_estimation_residuals.savefig(out_dir / "Estimation_residual.pdf")
        fig_SPICE_residuals.savefig(out_dir / "SPICE_residual.pdf")
        #fig_RA.savefig(out_dir / "RA_res.pdf")
        #fig_DEC.savefig(out_dir / "DEC_res.pdf")


        ##############################################################################################
        # EXTRACT RSW FORMAL ERRORS FROM COVARIANCES
        ##############################################################################################

        Covariances_array_RSW = ProcessingUtils.rotate_covariance_inertial_to_rsw(
                time_column, 
                covariances,
                states_SPICE_with_time
        )

        diag = np.diagonal(Covariances_array_RSW, axis1=1, axis2=2)
        formal_errors_RSW = np.sqrt(diag[:,0:3])/1e3
        
        fig_RSW = FigUtils.Residuals_RSW(formal_errors_RSW, time_column,type="difference",title=("RSW Formal Errors "))   
        

        fig_RSW.savefig(out_dir / ("Formal_Errors_Propagated_RSW.pdf"))
        np.save(out_dir / "covariances_array_rsw.npy",Covariances_array_RSW)
        np.save(out_dir / "formal_errors_RSW_km.npy",formal_errors_RSW)
        ##############################################################################################
        # SAVE OTHER FILES
        ##############################################################################################


        arr = np.stack(residual_history_arcseconds, axis=0)   # shape ??
        np.save(out_dir / "residual_history_arcseconds.npy", arr)

      
        arr1 = np.stack(state_history_array, axis=0)
        np.save(out_dir / "state_history_array.npy", arr1)

        arr2 = np.stack(state_history_initial_array, axis=0)
        np.save(out_dir / "state_history_initial_array.npy", arr2)

        np.save(out_dir / "state_history_array_full.npy",state_history_array_full)


        arr3 = np.stack(states_SPICE_with_time, axis=0)
        np.save(out_dir / "states_SPICE_with_time.npy", arr3)



        #save formal errors, covariance, initial state and estimated state
        np.save(out_dir / "formal_errors.npy",estimation_output.formal_errors)
        np.save(out_dir / "covariance.npy",estimation_output.covariance)
        np.save(out_dir / "final_paramaters.npy" ,estimation_output.final_parameters)
        np.save(out_dir / "initial_paramaters.npy" ,estimation_output.parameter_history[:,0])


        observations_sorted = observations.get_observations()
        np.save(out_dir / "observations_sorted.npy",np.array(observations_sorted,dtype=object))

        np.save(out_dir / "observation_times.npy",np.array(observation_times))

        residuals_sorted = observations.get_residuals()
        np.save(out_dir / "residuals_sorted.npy",np.array(residuals_sorted,dtype=object))

        parameter_history = estimation_output.parameter_history
        best_iteration = estimation_output.best_iteration
        correlations = estimation_output.correlations

        best_parameter_update = parameter_history[:,0] - parameter_history[:,best_iteration]

        np.save(out_dir / "parameter_history.npy", parameter_history)
        np.save(out_dir / "best_iteration.npy",best_iteration)
        np.save(out_dir / "correlations.npy",correlations)    




    elif settings['obs']['type'] == 'Simulated':
        print("Saving simulated observations residuals...")
        residuals_j2000, residuals_rsw = ProcessingUtils.format_residual_history(estimation_output.residual_history,
                                                                observations.get_concatenated_observation_times(),
                                                                state_history)

        ##############################################################################################
        #PLOTTING
        ##############################################################################################
        
        residuals_j2000_final = residuals_j2000[-1][:,1:4]/1e3
        residauls_rsw_final_time = residuals_rsw[-1][:,0]
        residuals_rsw_final = residuals_rsw[-1][:,1:4]/1e3
        
        residuals_rsw_fig = FigUtils.Residuals_RSW(residuals_rsw_final, residauls_rsw_final_time)

        #-------------------------------------------------------------------------------
        rms_fig = FigUtils.Residuals_RMS(residuals_j2000)


        #--------------------------------------------------------------------------------
        # Extract the time column (first column)
        time_column = state_history_array[:, [0]]   # keep it 2D (shape = (289, 1))
        states_SPICE_with_time = np.hstack((time_column, states_SPICE))

        fig_Cartesian = FigUtils.PlotCartesianDifference(state_history_array, states_SPICE_with_time)
        

        

        states_SPICE_RSW = ProcessingUtils.rotate_inertial_3_to_rsw(time_column, states_SPICE[:,0:3], states_SPICE_with_time)
        states_sim_RSW = []
        states_sim_RSW.append(ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_initial_array[:,1:4], states_SPICE_with_time))
        states_sim_RSW.append(ProcessingUtils.rotate_inertial_3_to_rsw(time_column, state_history_array[:,1:4], states_SPICE_with_time))

        diff_sim_RSW = (states_sim_RSW[0] - states_sim_RSW[-1])/1e3
        diff_sim_SPICE_RSW = (states_sim_RSW[0] - states_SPICE_RSW)/1e3
        diff_final_sim_SPICE_RSW = (states_sim_RSW[-1] - states_SPICE_RSW)/1e3

        fig_sim_RSW = FigUtils.Residuals_RSW(diff_sim_RSW, time_column,type="difference",title="RSW Difference Initial Final")
        fig_sim_SPICE_rsw = FigUtils.Residuals_RSW(diff_sim_SPICE_RSW, time_column,type="difference",title="RSW Difference Initial SPICE")
        fig_final_sim_SPICE_rsw = FigUtils.Residuals_RSW(diff_final_sim_SPICE_RSW, time_column,type="difference",title="RSW Difference Final SPICE")
            
        fig_sim_RSW_abs = FigUtils.Residuals_RSW(states_sim_RSW[-1], time_column,type="normal",title="RSW Final Simulation Absolute")
        fig_SPICE_RSW_abs = FigUtils.Residuals_RSW(states_SPICE_RSW, time_column,type="normal",title="RSW SPICE Absolute")
        #--------------------------------------------------------------------------------
        #save figs
   
        fig_sim_RSW.savefig(out_dir / "RSW_diff_Sim.pdf")
        fig_sim_SPICE_rsw.savefig(out_dir / "RSW_diff_initial_Sim_SPICE.pdf")
        fig_final_sim_SPICE_rsw.savefig(out_dir / "RSW_diff_final_Sim_SPICE.pdf")
        


        ##############################################################################################
        #PLOT PARAMETER UPDATE
        ##############################################################################################
        
        parameter_history = estimation_output.parameter_history
        best_iteration = estimation_output.best_iteration
        correlations = estimation_output.correlations

        best_parameter_update = parameter_history[:,0] - parameter_history[:,best_iteration]

        np.save(out_dir / "parameter_history.npy", parameter_history)
        np.save(out_dir / "best_iteration.npy",best_iteration)
        np.save(out_dir / "correlations.npy",correlations)    


        # fig = FigUtils.plot_correlation_matrix(correlations, settings['est']['est_parameters'])
        # fig.savefig(out_dir / 'correlations.pdf')

        # fig1 = FigUtils.plot_parameter_updates(best_parameter_update,  settings['est']['est_parameters'])
        # fig1.savefig(out_dir / "parameter_update.pdf")
        
        # fig2 = FigUtils.plot_parameter_history(parameter_history, settings['est']['est_parameters'], best_iteration=best_iteration)
        #fig2.savefig(out_dir / "parameter_history.pdf")
        #--------------------------------------------------------------------------
        #Get Different Flavors of FFTs
        #--------------------------------------------------------------------------
        #Get Triton Mean Motion
        kep = dep_vars_array[:, 10:16]  # [a, e, i, ω, Ω, ν]
        a = kep[:, 0]                   # meters
        # GM values (Tudat):
        mu_N = spice.get_body_gravitational_parameter("Neptune")
        mu_T = spice.get_body_gravitational_parameter("Triton")
        mu = mu_N + mu_T               
        # Mean motion time series (rad/s) 
        n_series = np.sqrt(mu / a**3)
        n_med = np.nanmedian(n_series)  # one way to take the mean of the mean motion
        f_rot_hz = 1/(n_med / (2*np.pi)) #  T (seconds)

        fft_fig_Jonas = FigUtils.create_fft_residual_figure(residuals_rsw[-1],f_rot_hz)

        fft_fig_Spectrum = FigUtils.periodogram_rsw(residuals_rsw[-1],1/f_rot_hz,mode='spectrum')


        ##############################################################################################
        #SAVE FIGS AND WRITE TO FILE
        ##############################################################################################
        residuals_rsw_fig.savefig(out_dir / "Residuals_RSW.pdf")
        rms_fig.savefig(out_dir / "rms.pdf")
        #Orbit_3D_fig.savefig(out_dir / "Orbit_3D.pdf")
        fft_fig_Jonas.savefig(out_dir / "fft_Jonas.pdf")

        fft_fig_Spectrum.savefig(out_dir / "fft_spectrum.pdf")

        #----------------------------------------------------------------------------------------------
        # Save residuals as numpy files
        arr = np.stack(residuals_rsw, axis=0)   # shape (5, 254, 4)
        np.save(out_dir / "residuals_rsw.npy", arr)

        arr2 = np.stack(residuals_j2000,axis=0)
        np.save(out_dir / "residuals_j2000.npy", arr2)
        
        arr1 = np.stack(state_history_array, axis=0)
        np.save(out_dir / "state_history_array.npy", arr1)

        np.save(out_dir / "state_history_array_full.npy",state_history_array_full)

    import copy

    settings_copy = copy.deepcopy(settings)
    
    #settings_copy['prop']['initial_covariance'] = settings_copy['prop']['initial_covariance'].tolist()
    #settings["prop"]["initial_state_uncertanity"] = settings["prop"]["initial_state_uncertanity"].tolist()
    settings_copy['obs'].pop('weights', None)
    if 'initial_state' in settings['prop']:
        settings_copy["prop"]["initial_state"] = settings_copy["prop"]["initial_state"].tolist()
    if 'initial_Pole_Pos' in settings_copy['env']:
        settings_copy['env']['initial_Pole_Pos'] = settings_copy['env']['initial_Pole_Pos'].tolist()
    if 'initial_Pole_lib_deg1' in settings_copy['env']:
        settings_copy['env']['initial_Pole_lib_deg1'] = settings_copy['env']['initial_Pole_lib_deg1'].tolist()
    #Save yaml settings file
    with open(out_dir / "settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(settings_copy, f, sort_keys=False, allow_unicode=True)
    

    # Close all figures
    plt.close('all')


    return estimation_output,observations,observations_settings,body_settings,system_of_bodies 
#---------------------------------------------------------------------------------------------------

def make_timestamped_folder(base_path="Results"):
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = Path(base_path) / folder_name
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path

if __name__ == "__main__":
        
    # Define temporal scope of the simulation - equal to the time JUICE will spend in orbit around Jupiter
    simulation_start_epoch = DateTime(2024, 1,  1).epoch() #2006, 8,  27 1963, 3,  4  
    simulation_end_epoch   = DateTime(2025, 1, 1).epoch()   #2006, 9, 2 2019, 10, 1
    
    simulation_initial_epoch = DateTime(2024, 10, 1).epoch() #2006, 10, 1
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


    settings_env['Neptune_rot_model_type'] = 'IAU2015' 
        # Model Type for rotation model of Neptune:
        #  'simple_from_spice' - simple spice,
        #  'spice' - full spice,
        #  'IAU2015' - based on the IAU2015 paper
        
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
    
    # --- Load names of data files
    with open("file_names.json", "r") as f:
        file_names_loaded = json.load(f)

    weights = pd.read_csv(
            "summary.txt", #Results/BetterFigs/AllModernObservations/PostProcessing/First/weights.txt
            sep="\t",
            index_col="id")

    settings_obs = dict()
    settings_obs["mode"] = ["pos"]
    settings_obs["bodies"] = [("Triton", "Neptune")]                           # bodies to observe
    settings_obs["cadence"] = 60*60*3 # Every 3 hours
    settings_obs["type"] = "Simulated" # Simulated or Real observations

    settings_obs["files"] = file_names_loaded             
    settings_obs["observations_folder_path"] = "Observations/AllModernECLIPJ2000"  #RelativeObservations AllModernECLIPJ2000 AllModernJ2000

    weights = weights.reset_index()
    
    settings_obs["use_weights"] = False
    settings_obs['std_weights'] = False
    settings_obs["timeframe_weights"] = False
    settings_obs["per_night_weights"] = False
    settings_obs["per_night_weights_id"] = False 
    settings_obs['per_night_weights_hybrid'] = False
    settings_obs["weights"] = weights
    
    #--------------------------------------------------------------------------------------------
    # ESTIMATION SETTINGS 
    #--------------------------------------------------------------------------------------------

    settings_est = dict()
    #settings_est['pseudo_observations_settings'] = pseudo_observations_settings
    #settings_est['pseudo_observations'] = pseudo_observations
    

    settings_est['est_parameters'] = ['initial_state','iau_rotation_model_pole'] 
        #Possible settings: 
        # initial state - default
        # Rotation_Pole_Position_Neptune - fixed rotation pole position (only with simple rotational model !)
        # iau_rotation_model_pole - rotation pole position (alpha,delta) with IAU rotation model
        # iau_rotation_model_pole_rate - rotation pole rate  (alpha_dot, delta_dot) with IAU rotation model
        # GM_Neptune - Gravitational Parameter of Neptune (GM_Neptune)
        # GM_Triton - Gravitational Parameter of Triton (GM_Triton)
        # spherical_harmonics - C20 and C40 spherical harmonics gravity of Neptune 
        
    #fill in settings 
    settings = dict()
    settings["env"] = settings_env
    settings["acc"] = settings_acc
    settings["prop"] = settings_prop
    settings["obs"] = settings_obs
    settings["est"] = settings_est
   

    main(settings,make_timestamped_folder("Results/PoleSimObservations"))


    #path_list = ["PoleOrientation/SimpleRotationModel/residuals_rsw.npy","PoleOrientation/EstimationSimpleRotationModel/residuals_rsw.npy"]
    #label_list = ["No Estimation","Estimated Simple Rotational Model"]
    #fig,fig_diff = FigUtils.Compare_RSW_Different_Solutions(path_list,label_list)

    #fig.savefig("R_RSW_Estimation_Comparison.pdf")
    #fig_diff.savefig("Diff_R_RSW_Estimation_Comparison.pdf")


    #----------------------------------------------------------------------------
