import os
import yaml
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from datetime import datetime, timedelta
from pathlib import Path
import csv

# tudatpy imports
from tudatpy import math
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
import tudatpy.estimation as estimation
from tudatpy import util
from tudatpy.estimation.observable_models_setup import links, model_settings
from tudatpy.estimation import observations_setup 

from tudatpy import numerical_simulation

from tudatpy.astro import time_conversion, element_conversion,frame_conversion
from tudatpy.astro.time_conversion import DateTime


from tudatpy.data import save2txt
from tudatpy.estimation.observations import observations_processing
import sys
from pathlib import Path


import pandas as pd
import warnings

from typing import Dict, List, Tuple
from tudatpy.estimation.observable_models_setup import links



# Get the path to the directory containing this file
current_dir = Path(__file__).resolve().parent

sys.path.append(str(current_dir / "HelperFunctions"))

import ProcessingUtils
import PropFuncs
import FigUtils
import ObsFunc
import nsdc

def observatory_info (Observatory): #Positive to north and east
    if len(Observatory) == 2:                   #Making sure 098 and 98 are the same
        Observatory = '0' + Observatory
    elif len(Observatory) == 1:                   #Making sure 098 and 98 are the same
        Observatory = '00' + Observatory
    with open('Observations/Observatories.txt', 'r') as file:    #https://www.projectpluto.com/obsc.htm, https://www.projectpluto.com/mpc_stat.txt
        lines = file.readlines()
        for line in lines[1:]:  # Ignore the first line
            columns = line.split()
            if columns[1] == Observatory:
                longitude = float(columns[2])
                latitude = float(columns[3])
                altitude = float(columns[4])
                return np.deg2rad(longitude),  np.deg2rad(latitude), altitude
        print('No matching Observatory found')

def LoadObservations(
        folder_path,
        system_of_bodies,
        files='None',
        weights = None,
        ra_dec_independent_weights = True,
        std_weights = False,
        timeframe_weights=False,
        per_night_weights=False,
        per_night_weights_id=False,
        per_night_weights_hybrid=False,
        Residual_filtering = True,
        epoch_filter_dict = None,
        ):
    """
    Load Observations from a specific folder, assign weights if promted.  

    Parameters
    ----------
    folder_path : strings
        path to the folder containing .csv files of the observations
    system_of_bodies : Tudatpy::SystemOfBodies
        required Tudatpy object to create ObservationCollection
    files: List of strings (not required)
        choose specific files to load from the folder_path, they must exist in that folder
    weights: Pandas Dataframe
        required if weights need to be assigned. Must contain required columns for weight assignment.
        Default is weights per id
    
    ra_dec_independent_weights: bool
        Default True: rmse hybrid weights indepedent for RA/DEC requires weights DataFrame
    
    std_weights: bool
        Default False: rmse weights, True choose std weight column 
    timeframe_weights: bool
        indicate if weights should be assigned per timeframe per id
    per_night_weights: bool
        indicate if the timeframe weights are descaled per night (different column is selected)
    per_night_weights_id: bool
        indicate if id weights descaled per night are provided (different column is selected)
    per_night_weights_hybrid: bool
        indicate if hybrid id/tf weights descaled per night are provided (different column is selected)
    Residual_filtering: bool
        indicate if filtering will be done by residual or by predifined dict file (True by default)
    epoch_filter_dict: dict
        key is file id, content is exact epochs of filtered observations per file  (None by default)

    Returns
    -------
    dict
        observations : Tudatpy::ObservationCollection
            Contains all loaded observations with appropriate weights
        observation_settings_list : Tudatpy::ObservationSettings
            Tudatpy object containing the observation settings
        observation_set_ids : List
            List of the all loaded file ids in order
    """

    #folder_path = 'ObservationsProcessed/CurrentProcess'
    if files == 'None':
        raw_observation_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    else:
        #print("Taking from list: ",files)
        # Instead of iterating over os.listdir(), iterate over files
        raw_observation_files = [os.path.join(folder_path, f) for f in files if f in os.listdir(folder_path)]
        print('observation files selected: ',files)
    

    obstimes = []
    observation_set_list = []
    observation_settings_list = []
    #Observatories = []
    observation_set_ids = []
    #Start loop over every csv in folder
    observation_collections_list = []
    epochs_rejected = {}

    for file in raw_observation_files:
        # if file != 'Observations/ObservationsProcessedTest/Triton_327_nm0082.csv':
        #     continue
        arc_start_times_local = []
        bias_values_local = []
        #Reading information from file name
        string_to_split = file.split("/")[-1]
        split_string = string_to_split.split("_")

        Moon = split_string[0]
        Observatory = split_string[1]
        file_id = split_string[2].split(".")[0]
        set_id = Observatory + "_" + file_id
        observation_set_ids.append(set_id)


        # Define the position of the observatory on Earth
        observatory_longitude, observatory_latitude, observatory_altitude = observatory_info(Observatory)

        # Add the ground station to the environment
        environment_setup.add_ground_station(
        system_of_bodies.get_body("Earth"),
        set_id,
        [observatory_altitude, observatory_latitude, observatory_longitude],
        element_conversion.geodetic_position_type)



        #Reading observational data from file
        ObservationList = []
        Timelist = []
        uncertainty_ra = []
        uncertainty_dec = []
        with open(file, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader) 

            arc_times = []
            arc_uncertainties_ra = []
            arc_uncertainties_dec = []

            for row in csv_reader:
                time = float(row[0])
                Timelist.append(time)
                obstimes.append(time)
                ObservationList.append(np.asarray([float(row[1]), float(row[2])]))
                uncertainty_ra.append(float(row[3]))
                uncertainty_dec.append(float(row[4]))
        
        angles = ObservationList
        times = Timelist

        
        # Define link ends
        link_ends = dict()                  #To change
        link_ends[links.transmitter] = links.body_origin_link_end_id("Triton")
        link_ends[links.receiver] = links.body_reference_point_link_end_id("Earth", str(set_id))
        link_definition = links.LinkDefinition(link_ends)

        #Create current observation settings
        observation_single_set_current = estimation.observations.single_observation_set(
            model_settings.angular_position_type, 
            link_definition,
            angles,
            times, 
            links.LinkEndType.receiver 
            )

        #Create current observation settings
        current_observation_settings = model_settings.angular_position(link_definition)

        #Convert to observation collection because can't compute residuals otherwise
        observation_collection_current = estimation.observations.ObservationCollection([observation_single_set_current]) 

        #-----------------------------------------------------------------------------------                   
        FilterObservations = True
        if FilterObservations == True and weights is None:
            #Compute residuals
            observation_simulators = observations_setup.observations_simulation_settings.create_observation_simulators(
                    [current_observation_settings],
                    system_of_bodies
                    )

            estimation.observations.compute_residuals_and_dependent_variables(
                    observation_collection_current,
                    observation_simulators,
                    system_of_bodies
                    )
            residuals = observation_collection_current.get_concatenated_residuals()


            #Create filter
            arcsec_to_rad = np.pi / (180.0 * 3600.0)
            upper_bound = 1.5 #arcseconds
            upper_bound_rad = upper_bound*arcsec_to_rad

            outlier_filter = observations_processing.observation_filter(
                    observations_processing.ObservationFilterType.residual_filtering
                    ,upper_bound_rad)


            opposite_outlier_filter = observations_processing.observation_filter(
                    observations_processing.ObservationFilterType.residual_filtering
                    ,upper_bound_rad,use_opposite_condition=True)

            
            
            #Filter single observation set
            observation_single_set_current_filtered =  estimation.observations.create_filtered_observation_set(observation_single_set_current,outlier_filter)
            
            
            #Print how many observations were filtered if any()
            rejected_observation_single_set_current =  estimation.observations.create_filtered_observation_set(
                    observation_single_set_current,
                    opposite_outlier_filter)
            #if np.shape(rejected_observation_single_set_current.residuals)[0] > 0:
            
            nr_all = np.shape(observation_single_set_current.residuals)[0] # all observations
            nr_filtered = np.shape(observation_single_set_current_filtered.residuals)[0] # filtered observations
            nr_rejected = np.shape(rejected_observation_single_set_current.residuals)[0] # rejected observations
            
            print("====================")
            print("FOR SET ID: ", set_id)
            print("all: ",nr_all)
            print("filtered: ",nr_filtered)
            print("rejected: ",nr_rejected)
            print("====================")

            #Compute rejected epochs
            epochs_all = observation_single_set_current.observation_times
            epochs_filtered = observation_single_set_current_filtered.observation_times
            epochs_rejected_current = [t for t in epochs_all if t not in epochs_filtered]

            epochs_rejected[set_id] = [t.to_float() for t in epochs_rejected_current]

            #" is ", np.shape(rejected_observation_single_set_current.residuals))
            




            #-----------------------------------------------------------------------------------   
            # Create a time based filter 
            if Residual_filtering == False:
                if np.shape(epoch_filter_dict[set_id])[0] != 0:
                    epoch_filter = observations_processing.observation_filter(
                        observations_processing.ObservationFilterType.epochs_filtering
                            ,epoch_filter_dict[set_id])

                    observation_single_set_current_filtered =  estimation.observations.create_filtered_observation_set(
                        observation_single_set_current,epoch_filter)
                else:
                    observation_single_set_current_filtered = observation_single_set_current

            #-----------------------------------------------------------------------------------   




        #-----------------------------------------------------------------------------------    
        # Append to list
        if FilterObservations == True and weights is None:
            observation_set_list.append(observation_single_set_current_filtered)
        else:
            observation_set_list.append(observation_single_set_current)

        observation_settings_list.append(current_observation_settings)

        #Create Observation Collection for the current file and assign weights
        if weights is not None:
            observation_single_set_current = estimation.observations.single_observation_set(
            model_settings.angular_position_type, 
            link_definition,
            angles,
            times, 
            links.LinkEndType.receiver)
            #-----------------------------------------------------------------------------------          
            if FilterObservations == True:
                #Compute residuals
                observation_simulators = observations_setup.observations_simulation_settings.create_observation_simulators(
                        [current_observation_settings],
                        system_of_bodies
                        )

                estimation.observations.compute_residuals_and_dependent_variables(
                        observation_collection_current,
                        observation_simulators,
                        system_of_bodies
                        )
                residuals = observation_collection_current.get_concatenated_residuals()


                #Create filter
                arcsec_to_rad = np.pi / (180.0 * 3600.0)
                upper_bound = 1.5 #arcseconds
                upper_bound_rad = upper_bound*arcsec_to_rad

                outlier_filter = observations_processing.observation_filter(
                        observations_processing.ObservationFilterType.residual_filtering
                        ,upper_bound_rad)


                opposite_outlier_filter = observations_processing.observation_filter(
                        observations_processing.ObservationFilterType.residual_filtering
                        ,upper_bound_rad,use_opposite_condition=True)

                
                
                #Filter single observation set
                observation_single_set_current_filtered =  estimation.observations.create_filtered_observation_set(observation_single_set_current,outlier_filter)
                
                
                #Print how many observations were filtered if any()
                rejected_observation_single_set_current =  estimation.observations.create_filtered_observation_set(
                        observation_single_set_current,
                        opposite_outlier_filter)

                nr_all = np.shape(observation_single_set_current.residuals)[0] # all observations
                nr_filtered = np.shape(observation_single_set_current_filtered.residuals)[0] # filtered observations
                nr_rejected = np.shape(rejected_observation_single_set_current.residuals)[0] # rejected observations
                
                print("====================")
                print("FOR SET ID: ", set_id)
                print("all: ",nr_all)
                print("filtered: ",nr_filtered)
                print("rejected: ",nr_rejected)
                print("====================")


                #-----------------------------------------------------------------------------------   
                # Create a time based filter 
                if Residual_filtering == False:
                    epoch_filter = observations_processing.observation_filter(
                        observations_processing.ObservationFilterType.epochs_filtering
                            ,epoch_filter_dict[set_id])

                    observation_single_set_current_filtered =  estimation.observations.create_filtered_observation_set(
                        observation_single_set_current,epoch_filter)
                                
                    print("====================")
                    print("FOR EPOCH FILTER: ", set_id)
                    print("all: ",nr_all)
                    print("filtered: ",len(observation_single_set_current_filtered.residuals))
                    print("====================")


                #-----------------------------------------------------------------------------------   




            #-----------------------------------------------------------------------------------     
            if FilterObservations == True:
                observation_collection_current = estimation.observations.ObservationCollection([observation_single_set_current_filtered]) 
            else:
                observation_collection_current = estimation.observations.ObservationCollection([observation_single_set_current]) 

            if timeframe_weights == True:
                # Filter all rows belonging to this id  
                weights_id = weights[weights["id"] == set_id]

                # Repeat rows based on n_obs
                #expanded = weights_id.loc[weights_id.index.repeat(weights_id['n_obs'])]
                
                expanded = weights_id.loc[weights_id.index.repeat(weights_id['n_obs'] * 2)]

                #choose the approriate column based on conditions
                if ra_dec_independent_weights == False:
                    if per_night_weights == True:
                        weight_column = 'mean_weights_std_scaled' if std_weights else 'mean_weight_rmse_scaled' 
                    elif per_night_weights_id ==True:
                        weight_column = 'mean_weight_std_id_scaled' if std_weights else 'mean_weight_rmse_id_scaled'
                    elif per_night_weights_hybrid == True:
                        weight_column = 'mean_weight_std_tf_id_scaled' if std_weights else  'mean_weight_rmse_tf_id_scaled'
                    else:
                        weight_column = 'mean_weight_std' if std_weights else 'mean_weight_rmse'
                    
                    #assign values based on weight column
                    tabulated_weights = expanded[weight_column].values
                  
                elif ra_dec_independent_weights == True:
                    weight_ra_column = 'weight_rmse_ra_tf_id_scaled'
                    weight_dec_column = 'weight_rmse_dec_tf_id_scaled'
                    ra_weights = expanded[weight_ra_column].values
                    dec_weights = expanded[weight_dec_column].values
                
                    tabulated_weights = np.empty((ra_weights.size + dec_weights.size,), dtype=ra_weights.dtype)

                    #Assign values based on 2 weight columns intertwined
                    tabulated_weights[0::2] = ra_weights  # Every even index gets RA
                    tabulated_weights[1::2] = dec_weights  # Every odd index gets DEC


                n_obs = np.shape(observation_collection_current.get_observation_times())[1]
                expected_length = 2 * n_obs  # Should be 402 for 201 observations

                current_length = len(tabulated_weights)
                missing_values = expected_length - current_length

                if missing_values > 0:
                        mean_weight = np.mean(tabulated_weights)
                        print(f"Adding {missing_values} values with mean weight {mean_weight}")
                        tabulated_weights = np.append(tabulated_weights, [mean_weight] * missing_values)

                tabulated_weights = tabulated_weights.reshape(-1, 1)

                #new = np.full_like(tabulated_weights, 10**10)

                observation_collection_current.set_tabulated_weights(tabulated_weights)


            #------------------------------------------------------------------------------------        
            #Assign weights from ID
            else:
                weights_id = weights[weights["id"] == set_id]
                w_avg = (weights_id['weight_ra'] + weights_id["weight_dec"])/2

                #assign weights
                observation_collection_current.set_constant_weight(
                        w_avg,
                        estimation.observations.observations_processing.observation_parser(model_settings.angular_position_type)
                        )
            #------------------------------------------------------------------------------------        
            #append current ObservationCollection to list 
            observation_collections_list.append(observation_collection_current) #
            #------------------------------------------------------------------------------------        
            
    #Create observation Collection with Weights
    observation_collection_full = estimation.observations.merge_observation_collections(observation_collections_list)
    
    #Create ObservationCollection without weights
    observations = estimation.observations.ObservationCollection(observation_set_list) 
    
    print('size of full observations without weights: ', len(observations.get_concatenated_observation_times()))
   
    if weights is not None:
        observations = observation_collection_full
        print('size of full observations with weights: ', len(observation_collection_full.get_concatenated_observation_times()))
    

    return observations,observation_settings_list,observation_set_ids,epochs_rejected





def Get_SPICE_residual_from_observations(observations,Observatories,system_of_bodies,global_frame_orientation='ECLIPJ2000'):
    observation_times_all = observations.get_observation_times()

    #Does not work
    #observatories = observations.get_reference_points_in_link_ends()

    observations_list = observations.get_observations()
    
    diflist = []
    for j in range(len(observation_times_all)):
        
        observation_times = observation_times_all[j]
        reshaped_observations_list = observations_list[j].reshape(-1,2)


        for i in range(len(reshaped_observations_list)):
            observatory_ephemerides = environment_setup.create_ground_station_ephemeris(
            system_of_bodies.get_body("Earth"),
            Observatories[j],
            system_of_bodies
            )


            RA_spice, DEC_spice = nsdc.get_angle_rel_body(DateTime.from_epoch(observation_times[i]),global_frame_orientation,observatory_ephemerides, "Triton",global_frame_orientation,global_frame_origin="SSB")
            RA, DEC = reshaped_observations_list[i]
            diflist.append([RA-RA_spice,DEC-DEC_spice])
  
    diflist = np.array(diflist)
  
    uncertainty_ra_SPICE = diflist[:,0] * 180/np.pi * 3600 
    uncertainty_dec_SPICE = diflist[:,1] * 180/np.pi * 3600 
    uncertainty_SPICE = [uncertainty_ra_SPICE,uncertainty_dec_SPICE]
    return uncertainty_SPICE


def PlotCountHistogram(observation_times,path,bin_type="Month"):
    '''
    Bin Type defines how the bin count is calculated:
    Month - each month is 1 bin, suitable for large timespans and data count
    Count - number of observations = number of bins    
    '''
    flat_times = np.concatenate(observation_times).tolist()

    flat_times_sorted = np.sort(flat_times)
    flat_times_Datetime = FigUtils.ConvertToDateTime(flat_times_sorted)
    
    df_flat_times = pd.DataFrame({"datetime": flat_times_Datetime})
    
    if bin_type == "Month":
        first_entry = df_flat_times.values[0]
        last_entry = df_flat_times.values[-1]
        
        year_first = first_entry.astype('datetime64[Y]').astype(int) + 1970
        month_first = first_entry.astype('datetime64[M]').astype(int) % 12 + 1
        
        year_last = last_entry.astype('datetime64[Y]').astype(int) + 1970
        month_last = last_entry.astype('datetime64[M]').astype(int) % 12 + 1
        
        
        bins_num = 12*(year_last-year_first-1)+abs(month_first-11)+month_last
        bins_num = bins_num[0]
    elif bin_type == "Count":
        bins_num = len(flat_times_sorted)
    else:
        print("Unrecognized bin_type entry.")

    
    # Convert your datetimes to Matplotlib date numbers
    
    # Compute histogram manually (this gives you counts and bins)
    hist, bins = np.histogram(flat_times_sorted, bins=bins_num)
    
    # Find the bin with the maximum count
    max_index = np.argmax(hist)
    max_bin_start = bins[max_index] # DateTime(2006, 9, 17, 4, 43, 11.925976)
    max_bin_end = bins[max_index + 1]
    
    
    fig, ax = plt.subplots(figsize=(9, 4))
    df_flat_times.hist(bins=bins_num, ax=ax, edgecolor='black')
    
    
    
    ax.set_xlabel('Time')
    locator   = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    #ax.set_yscale("log") 
    ax.set_ylabel('Count')
    ax.set_title('Observations per Month')
    plt.tight_layout()
    
    fig.savefig(path / "Observation_Histogram_PerMonth.pdf", bbox_inches="tight")


def PlotResidualHistorgram(residuals,output_folder_path):
    
    residuals_RA = residuals[0]
    residuals_DEC = residuals[1]

    df_residuals_RA = pd.DataFrame({"residuals_RA": residuals_RA})
    df_residuals_DEC = pd.DataFrame({"residuals_DEC": residuals_DEC})

    average_max = np.average([max(residuals[0]),max(residuals[1])])
    average_min = np.average([min(residuals[0]),min(residuals[1])])
    resolution = 100 # 100 miliarcseconds
    bin_size =  int((average_max - average_min)*1000/resolution)

    if bin_size == 0:
        bin_size = 1
        print("bin size is 0, setting to 1.")
        print("number of residuals: ",len(residuals[0]))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    # Plot RA residuals
    df_residuals_RA.hist(ax=ax[0], bins=bin_size, color='steelblue', edgecolor='black')
    ax[0].set_title("RA Residuals")
    ax[0].set_xlabel("Residual [rad]")
    ax[0].set_ylabel("Count")

    # Plot DEC residuals
    df_residuals_DEC.hist(ax=ax[1], bins=bin_size, color='darkorange', edgecolor='black')
    ax[1].set_title("DEC Residuals")
    ax[1].set_xlabel("Residual [rad]")

    plt.tight_layout()
    fig.savefig(output_folder_path / "Histogram_Residuals_RA_DEC.pdf", bbox_inches="tight")


        



def PlotResidualsTime(observations,Observatories,system_of_bodies,output_folder):
    '''
    Saves a pdf figure of all the observation from a Tudat object.

    '''
    residuals_SPICE = Get_SPICE_residual_from_observations(observations,Observatories,system_of_bodies)

    observation_times_DateFormat = FigUtils.ConvertToDateTime(observations.get_concatenated_observation_times())

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.scatter(observation_times_DateFormat,residuals_SPICE[0])
    ax.set_xlabel('Observation epoch [years since J2000]')
    ax.set_ylabel('spice-observed RA [arcseconds]')
    ax.grid(True, alpha=0.3)

    locator   = mdates.AutoDateLocator()               # chooses sensible tick spacing
    formatter = mdates.ConciseDateFormatter(locator)   # compact, smart formatting
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    fig.savefig(output_folder_path / "RA_residuals_SPICE.pdf")

    #--------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.scatter(observation_times_DateFormat,residuals_SPICE[1])
    ax.set_xlabel('Observation epoch [years since J2000]')
    ax.set_ylabel('spice-observed DEC [arcseconds]')
    ax.grid(True, alpha=0.3)

    locator   = mdates.AutoDateLocator()               # chooses sensible tick spacing
    formatter = mdates.ConciseDateFormatter(locator)   # compact, smart formatting
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


    fig.savefig(output_folder_path / "DEC_residuals_SPICE.pdf")



#---------------------------------------------------------------------------------------------------------------------------
#BETTER WAY TO ASSIGN WEIGHTS
#---------------------------------------------------------------------------------------------------------------------------

def compute_and_assign_weights(residuals: np.ndarray, 
                                observations,
                                gap_threshold_hours: float = 4.0,
                                min_obs_per_frame: int = 1,
                                weight_type: str = 'hybrid',
                                min_sigma_arcsec: float = 0.01) -> Tuple:
    """
    Compute and assign weights to observation sets based on residuals.
    
    Parameters:
    -----------
    residuals : np.ndarray
        Array of shape [n, 3] containing [time, RA_residual, DEC_residual]
    observations : ObservationCollection
        Collection of all observations
    gap_threshold_hours : float
        Maximum time gap (hours) to keep observations in same timeframe
    min_obs_per_frame : int
        Minimum observations per timeframe before allowing a break
        
    Returns:
    --------
    observations : ObservationCollection
        Updated observation collection with assigned weights
    weights_df : pd.DataFrame
        DataFrame containing weight information for all observations
    """
    
    # Extract observation sets
    sets = observations.sorted_observation_sets
    ObservableType = list(sets.keys())[0]
    
    all_observations_sets = []
    for observable_type, inner_dict in sets.items():
        for link_end_id, observation_list in inner_dict.items():
            all_observations_sets.extend(observation_list)
    
    # Prepare data structures for results
    weights_data = []
    current_idx = 0
    
    # Process each observation set
    for set_idx, obs_set in enumerate(all_observations_sets):
    
        ref_point_id = obs_set.link_definition.link_end_id(links.receiver).reference_point
        
        n_obs = np.shape(obs_set.observation_times)[0]
        
        obs_times_obs = obs_set.observation_times
        obs_times_test = [t.to_float() for t in obs_times_obs]

        #Extract residuals (not from observation set)
        set_residuals = residuals[current_idx:current_idx + n_obs, :]
        times = set_residuals[:, 0]
        ra_residuals = set_residuals[:, 1]
        dec_residuals = set_residuals[:, 2]
        
      
        # Test if times are consistent between residuals and observation set
        diff = obs_times_test - times
        assert np.allclose(obs_times_test, times), \
            f"Times don't match! Max difference: {np.max(np.abs(diff))}"


        #------------------------------------------------------------
        # COMPUTE ID WEIGHTS (PER FILE)
        #------------------------------------------------------------
        ra_rmse_global = np.sqrt(np.mean(ra_residuals**2))
        dec_rmse_global = np.sqrt(np.mean(dec_residuals**2))
        
        # Clip values according to min_sigma value
        #min_sigma_arcsec = 0.01 # 10 mas minimum rmse
        min_sigma_rad = min_sigma_arcsec / (3600 * 180 / np.pi) # 10mas in rad
        

        if ra_rmse_global < min_sigma_rad:
            n_clipped = np.sum(ra_rmse_global < min_sigma_rad)
            print(f"RA ID RMSE bellow minmimum sigma {min_sigma_arcsec} [arcseconds] for {ref_point_id}")
            #warnings.warn(f"RA ID RMSE values were below minimum sigma and were clipped for {ref_point_id}")
            ra_rmse_global = min_sigma_rad
        if dec_rmse_global < min_sigma_rad:
            n_clipped = np.sum(dec_rmse_global < min_sigma_rad)
            print(f"DEC ID RMSE bellow minmimum sigma {min_sigma_arcsec} [arcseconds] for {ref_point_id}")
            dec_rmse_global = min_sigma_rad       

        weight_ra_global = 1.0 / (ra_rmse_global**2) #if ra_rmse_global > 0 else 1.0
        weight_dec_global = 1.0 / (dec_rmse_global**2) #if dec_rmse_global > 0 else 1.0
        
        #------------------------------------------------------------
        # PER TIMEFRAME RMSE AND WEIGHTS (New ID Weight version too)
        #------------------------------------------------------------
        timeframes = split_observations_into_timeframes(
            times, 
            gap_threshold_hours=gap_threshold_hours,
            min_obs_per_frame=min_obs_per_frame
        )
        
        n_timeframes = timeframes.max() + 1
        
        # 3. Compute local (per-timeframe) weights
        weight_ra_local = np.zeros(n_obs)
        weight_dec_local = np.zeros(n_obs)

        # 3. RMSE per timeframe * n_obs_tf (descaled)
        tf_rmse_descaled_ra = np.zeros(n_obs)
        tf_rmse_descaled_dec = np.zeros(n_obs)
        tf_rmse_descaled_ra_2 = np.zeros(n_timeframes)
        tf_rmse_descaled_dec_2 = np.zeros(n_timeframes)


        # Clipped frame ids
        clipped_ids_ra = {}
        clipped_ids_dec = {}

        for frame_idx in range(n_timeframes):
            # Find observations in this timeframe
            mask = (timeframes == frame_idx)
            
            if np.sum(mask) > 0:
                # Compute RMSE for this timeframe
                ra_rmse_frame = np.sqrt(np.mean(ra_residuals[mask]**2))
                dec_rmse_frame = np.sqrt(np.mean(dec_residuals[mask]**2))


                n_obs_timeframe = len(ra_residuals[mask])


                # Clip values according to min_sigma_rad
                if ra_rmse_frame < min_sigma_rad:
                    ra_rmse_frame = min_sigma_rad

                    clipped_ids_ra[frame_idx] = n_obs_timeframe
                    
                if dec_rmse_frame < min_sigma_rad:
                    dec_rmse_frame = min_sigma_rad
                    
                    clipped_ids_dec[frame_idx] = n_obs_timeframe

                # Assign weights (inverse variance)
                weight_ra_local[mask] = (1.0 / (ra_rmse_frame**2))/n_obs_timeframe #if ra_rmse_frame > 0 else 10**3 #Very bad weight
                weight_dec_local[mask] = (1.0 / (dec_rmse_frame**2))/n_obs_timeframe #if dec_rmse_frame > 0 else 10**3 #Very bad weight
        
                # New ID (per file) weighting strategy where each timeframe is
                tf_rmse_descaled_ra[mask] = ra_rmse_frame*np.sqrt(n_obs_timeframe)
                tf_rmse_descaled_dec[mask] = dec_rmse_frame*np.sqrt(n_obs_timeframe)

                tf_rmse_descaled_ra_2[frame_idx] = ra_rmse_frame*np.sqrt(n_obs_timeframe)
                tf_rmse_descaled_dec_2[frame_idx] = dec_rmse_frame*np.sqrt(n_obs_timeframe)

        print(f"For timeframe rmse, for id: {ref_point_id}, {sum(clipped_ids_ra.values())} RA obs and {sum(clipped_ids_dec.values())} DEC obs are clipped out of {n_obs} total.")    

        # COMPUTE ID WEIGHT NEW OPTION 1
        ra_rmse_id_new_1 = np.sqrt(np.mean(tf_rmse_descaled_ra**2))
        dec_rmse_id_new_1 = np.sqrt(np.mean(tf_rmse_descaled_dec**2))
        
        weight_ra_id_new_1 =  1/ra_rmse_id_new_1**2
        weight_dec_id_new_1 = 1/dec_rmse_id_new_1**2

        ra_rmse_id_new_2 = np.sqrt(np.mean(tf_rmse_descaled_ra_2**2))
        dec_rmse_id_new_2 = np.sqrt(np.mean(tf_rmse_descaled_dec_2**2))

        weight_ra_id_new_2 = 1/ra_rmse_id_new_2**2
        weight_dec_id_new_2 = 1/dec_rmse_id_new_2**2


        # COMPUTE HYBRID WEIGHTS
        weight_ra_hybrid = np.sqrt(weight_ra_global * weight_ra_local)
        weight_dec_hybrid = np.sqrt(weight_dec_global * weight_dec_local)
        
        weight_ra_hybrid_old = (weight_ra_global + weight_ra_local)/2
        weight_dec_hybrid_old = (weight_dec_global + weight_dec_local)/2

        #NEW HYBRID WEIGHTS

        weight_ra_hybrid_new_id = np.sqrt(weight_ra_id_new_2 * weight_ra_local)
        weight_dec_hybrid_new_id = np.sqrt(weight_dec_id_new_2 * weight_dec_local)
        
        weight_ra_hybrid_old_new_id = (weight_ra_id_new_2 + weight_ra_local)/2
        weight_dec_hybrid_old_new_id = (weight_dec_id_new_2 + weight_dec_local)/2
        


        # Determine which weights to use based on weight_type
        if weight_type == 'hybrid':
            weight_ra_selected = weight_ra_hybrid
            weight_dec_selected = weight_dec_hybrid
        elif weight_type == 'hybrid_old':
            weight_ra_selected = weight_ra_hybrid_old
            weight_dec_selected = weight_dec_hybrid_old
        elif weight_type =='hybrid_new_id':
            weight_ra_selected = weight_ra_hybrid_new_id
            weight_dec_selected = weight_dec_hybrid_new_id
        elif weight_type == 'hybrid_old_new_id':
            weight_ra_selected = weight_ra_hybrid_old_new_id
            weight_dec_selected = weight_dec_hybrid_old_new_id
        elif weight_type == 'timeframe':
            weight_ra_selected = weight_ra_local
            weight_dec_selected = weight_dec_local
        elif weight_type == 'id':
            weight_ra_selected = np.full(n_obs, weight_ra_global)  # Broadcast to array
            weight_dec_selected = np.full(n_obs, weight_dec_global)
        elif weight_type == 'id_new_1':
            weight_ra_selected = np.full(n_obs,weight_ra_id_new_1)
            weight_dec_selected = np.full(n_obs,weight_dec_id_new_1)
        elif weight_type == 'id_new_2':
            weight_ra_selected = np.full(n_obs,weight_ra_id_new_2)
            weight_dec_selected = np.full(n_obs,weight_dec_id_new_2)    
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
        
        # 5. Format weights for set_tabulated_weights (interleaved RA, DEC)
        weights_array = np.zeros(2 * n_obs)
        weights_array[0::2] = weight_ra_selected   # RA weights at even indices
        weights_array[1::2] = weight_dec_selected  # DEC weights at odd indices
        
        # 6. Assign weights to observation set
        obs_set.set_tabulated_weights(weights_array)
        

        # 7. Store data for DataFrame (only selected weights)
        for obs_idx in range(n_obs):
            weights_data.append({
                'set_index': set_idx,
                'ref_point_id': ref_point_id,
                'obs_index': obs_idx,
                'global_obs_index': current_idx + obs_idx,
                'timeframe': timeframes[obs_idx],
                'time': times[obs_idx],
                'ra_residual': ra_residuals[obs_idx],
                'dec_residual': dec_residuals[obs_idx],
                'ra_rmse_id': ra_rmse_global,
                'dec_rmse_id': dec_rmse_global,
                'weight_ra': weight_ra_selected[obs_idx],
                'weight_dec': weight_dec_selected[obs_idx],
                'weight_type': weight_type,  # Optional: track which type was used
            })
            
        # Update index for next set
        current_idx += n_obs
    
    # Create DataFrame
    weights_df = pd.DataFrame(weights_data)
    
    return observations, weights_df


def split_observations_into_timeframes(times: np.ndarray, 
                                       gap_threshold_hours: float = 4.0,
                                       min_obs_per_frame: int = 1) -> np.ndarray:
    """
    Split observations into timeframes based on time gaps.
    
    Parameters:
    -----------
    times : np.ndarray
        Array of observation times (assumed sorted)
    gap_threshold_hours : float
        Maximum gap between observations in same timeframe (hours)
    min_obs_per_frame : int
        Minimum observations required before allowing a frame break
        
    Returns:
    --------
    timeframes : np.ndarray
        Array of timeframe indices for each observation
    """
    n_obs = len(times)
    gap_threshold_seconds = gap_threshold_hours * 3600.0
    
    timeframes = np.zeros(n_obs, dtype=int)
    current_frame = 0
    obs_in_current_frame = 1
    
    for i in range(1, n_obs):
        time_gap = times[i] - times[i-1]
        
        # Start new frame if gap exceeds threshold AND minimum obs requirement met
        if time_gap >= gap_threshold_seconds and obs_in_current_frame >= min_obs_per_frame:
            current_frame += 1
            obs_in_current_frame = 1
        else:
            obs_in_current_frame += 1
            
        timeframes[i] = current_frame
    
    return timeframes


#---------------------------------------------------------------------------------------------------------------------------
# MANUAL BIAS ASSIGNMENT
#---------------------------------------------------------------------------------------------------------------------------

import tudatpy.estimation as estimation
from tudatpy.estimation import observations_setup 


def apply_dec_bias_to_observations(
    observations,
    observations_settings,
    system_of_bodies,
    bias_dict_arcsec
):
    """
    Apply DEC bias to observation sets based on reference point IDs.

    Parameters
    ----------
    observations : ObservationCollection
    observations_settings : list
    system_of_bodies : SystemOfBodies
    bias_dict_arcsec : dict
        {ref_point_id: bias_arcsec}

    Returns
    -------
    observations : ObservationCollection (modified)
    applied_bias_rad : dict
        {ref_point_id: bias_rad}
    """

    # --- gather all sets ---
    sets = observations.sorted_observation_sets

    all_sets = []
    for _, inner_dict in sets.items():
        for _, obs_list in inner_dict.items():
            all_sets.extend(obs_list)

    applied_bias_rad = {}

    # --- loop sets ---
    for obs_set in all_sets:

        ref_id = obs_set.link_definition.link_end_id(
            links.receiver
        ).reference_point

        if ref_id not in bias_dict_arcsec:
            continue

        # convert arcsec -> rad
        bias_rad = np.deg2rad(bias_dict_arcsec[ref_id] / 3600.0)

        obs = obs_set.concatenated_observations.copy()

        # apply to DEC
        obs[1::2] += bias_rad

        obs_set.set_observations(obs)

        applied_bias_rad[ref_id] = bias_rad


    # --- create simulators ---
    observation_simulators = observations_setup.observations_simulation_settings.create_observation_simulators(
    observations_settings,
    system_of_bodies
    )

    # --- recompute residuals ---
    estimation.observations.compute_residuals_and_dependent_variables(
        observations,
        observation_simulators,
        system_of_bodies
        )

    return observations, applied_bias_rad

def PlotResidualBiased(times_sec,residuals_old,residuals_new):
        # --- constants ---
    RAD_TO_MAS = 206264806.247
    J2000 = datetime(2000, 1, 1, 12, 0, 0)

    # --- times ---
    #times_sec = np.array([t.to_float() for t in obs_set.observation_times])
    times_dt = [J2000 + timedelta(seconds=float(s)) for s in times_sec]

    # --- residuals to mas ---
    # ra_old  = residuals_old[:, 0] * RAD_TO_MAS
    # dec_old = residuals_old[:, 1] * RAD_TO_MAS
    # ra_new  = residuals_new[:, 0] * RAD_TO_MAS
    # dec_new = residuals_new[:, 1] * RAD_TO_MAS

    ra_old  = residuals_old[0::2] * RAD_TO_MAS
    dec_old = residuals_old[1::2] * RAD_TO_MAS

    ra_new  = residuals_new[0::2] * RAD_TO_MAS
    dec_new = residuals_new[1::2] * RAD_TO_MAS
    
    # --- figure layout ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # --- scatter RA ---
    axes[0, 0].scatter(times_dt, ra_old, s=5, label="Old")
    axes[0, 0].scatter(times_dt, ra_new, s=5, label="New")
    axes[0, 0].set_ylabel("RA residual [mas]")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # --- scatter DEC ---
    axes[1, 0].scatter(times_dt, dec_old, s=5, label="Old")
    axes[1, 0].scatter(times_dt, dec_new, s=5, label="New")
    axes[1, 0].set_ylabel("DEC residual [mas]")
    axes[1, 0].set_xlabel("Datetime UTC")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # --- histogram RA ---
    axes[0, 1].hist(ra_old, bins=50, alpha=0.5, label="Old")
    axes[0, 1].hist(ra_new, bins=50, alpha=0.5, label="New")
    axes[0, 1].set_xlabel("RA residual [mas]")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # --- histogram DEC ---
    axes[1, 1].hist(dec_old, bins=50, alpha=0.5, label="Old")
    axes[1, 1].hist(dec_new, bins=50, alpha=0.5, label="New")
    axes[1, 1].set_xlabel("DEC residual [mas]")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    # --- save only ---
    #fig.savefig(out_dir / "Manual_Biased.pdf")
    return fig



    # # Extract observation sets
    # sets = observations.sorted_observation_sets
    # ObservableType = list(sets.keys())[0]
    
    # all_observations_sets = []
    # for observable_type, inner_dict in sets.items():
    #     for link_end_id,     observation_list in inner_dict.items():
    #         all_observations_sets.extend(observation_list)
    
    # # Prepare data structures for results
    # weights_data = []
    # current_idx = 0
    
    # ref_point_ids = {}
    # # Process each observation set
    # for set_idx, obs_set in enumerate(all_observations_sets):
    #     # Get reference point ID for this set
    #     ref_point_ids[set_idx] = obs_set.link_definition.link_end_id(links.receiver).reference_point

    # obs_set = all_observations_sets[17]
    
    # obs_old = obs_set.concatenated_observations
    # residuals_old = np.array(obs_set.residuals)


    # obs = obs_set.concatenated_observations.copy()

    # bias_rad = -np.deg2rad(0.2 / 3600.0)  # 210 marcsec example

    # # DEC entries are every second element
    # obs[1::2] += bias_rad


    # obs_set.set_observations(obs)

    # import tudatpy.estimation as estimation
    # from tudatpy.estimation import observations_setup 

    # observation_simulators = observations_setup.observations_simulation_settings.create_observation_simulators(
    #     observations_settings,
    #     system_of_bodies
    #     )

    # estimation.observations.compute_residuals_and_dependent_variables(
    #     observations,
    #     observation_simulators,
    #     system_of_bodies
    #     )

    # residuals_new = np.array(obs_set.residuals)
