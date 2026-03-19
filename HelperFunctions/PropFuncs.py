import numpy as np
import yaml

from tudatpy import numerical_simulation
from tudatpy.dynamics import environment_setup
from tudatpy.dynamics import propagation_setup
import tudatpy.estimation
from tudatpy.estimation.observable_models_setup import links, model_settings
from tudatpy.estimation import observations_setup
from tudatpy.estimation import estimation_analysis
from tudatpy.dynamics import parameters_setup
from tudatpy.numerical_simulation import estimation_setup
from tudatpy.estimation.observable_models_setup import links
from tudatpy import util
from tudatpy.interface import spice

import ObsFunc

def Create_Env(settings_dict):
    # Create default body settings and bodies at Neptune J2000
    start_epoch = settings_dict["start_epoch"]
    end_epoch = settings_dict["end_epoch"]
    global_frame_origin = settings_dict["global_frame_origin"] 
    global_frame_orientation = settings_dict["global_frame_orientation"]
    bodies_to_create = settings_dict["bodies"]
    interpolator_triton_cadance = settings_dict["interpolator_triton_cadance"]

    neptune_extended_gravity = settings_dict["neptune_extended_gravity"]

    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)


    ## Triton ########################################################################################

    body_settings.get("Triton").ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
        start_epoch-100*30, end_epoch+100*30, interpolator_triton_cadance, 
        global_frame_origin, global_frame_orientation)


    ## Neptune 
    if neptune_extended_gravity == "Jacobson2009":
        # Define spherical harmonics from Jacobson 2009
        J2 = 3408.428530717952e-6
        J4 = -33.398917590066e-6
        C20 = -J2 / np.sqrt(5.0)   # C̄20 = -J2 / sqrt(2*2+1)
        C40 = -J4 / 3.0            # C̄40 = -J4 / sqrt(2*4+1) = -J4/3

        # Build coefficient matrices (normalized)
        l_max, m_max = 4, 0
        Cbar = np.zeros((l_max+1, l_max+1))
        Sbar = np.zeros_like(Cbar)
        Cbar[2, 0] = C20
        Cbar[4, 0] = C40

        #Get GM and radius from SPICE
        mu_N = spice.get_body_gravitational_parameter("Neptune")
        radii_km = spice.get_body_properties("Neptune","RADII",3)   # returns [Rx, Ry, Rz] in km in tudatpy >=0.8
        R_eq = radii_km[0] * 1e3   # meters (use equatorial as reference radius)


        body_settings.get("Neptune").gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(
            gravitational_parameter=mu_N,
            reference_radius=R_eq,
            normalized_cosine_coefficients=Cbar,
            normalized_sine_coefficients=Sbar,
            associated_reference_frame="IAU_Neptune",
        )


    # set parameters for defining the rotation between frames
    original_frame = global_frame_orientation
    target_frame = "IAU_Neptune"
    target_frame_spice = "IAU_Neptune" # is this correct?
   

# SELECT ROTATION MODEL NEPTUNE
#---------------------------------------------------------------------------------------------------------------------------------------------------   
    # create rotation model settings and assign to body settings of "Neptune"
    if settings_dict['Neptune_rot_model_type'] == 'simple_from_spice':
        body_settings.get( "Neptune" ).rotation_model_settings = environment_setup.rotation_model.simple_from_spice(
                original_frame, target_frame, target_frame_spice, start_epoch)

    elif settings_dict['Neptune_rot_model_type'] == 'spice':
        body_settings.get( "Neptune" ).rotation_model_settings = environment_setup.rotation_model.spice(
                original_frame, target_frame, target_frame_spice)
    
    # IAU 2015 (2018)
    elif settings_dict['Neptune_rot_model_type'] == 'IAU2015':
        nominal_meridian = np.deg2rad(249.978) # W_0
        nominal_pole = np.deg2rad(np.array([299.36,43.46])) #alpha_0 and delta_0
        rotation_rate=np.deg2rad(541.1397757/24/3600) # W_0_dot (in paper it's multipled by day so /24/3600 should align with tudat check!)
        pole_precession= np.array([0,0]) # alpha_0_dot and delta_0_dot are 0
        merdian_periodic_terms = {np.deg2rad(52.316/36525/24/3600): (np.deg2rad(-0.48), np.deg2rad(357.85))} #w_N_i, W_i, Phi_N_i in that order

        
        # Values for alpha and delta from IAU 2015
        w_n_i = np.deg2rad(52.316/36525/24/3600)
        alpha_i = np.deg2rad(0.7)
        delta_i = np.deg2rad(-0.51)
        phi = np.deg2rad(357.85)


        # Create the numpy array for [alpha_i, delta_i] as a 2x1 column vector
        alpha_delta = np.array([alpha_i, delta_i])

        if 'initial_Pole_Pos' in settings_dict:
            nominal_pole = settings_dict['initial_Pole_Pos']
        if 'initial_Pole_lib_deg1' in settings_dict:
            alpha_delta = settings_dict['initial_Pole_lib_deg1']



        # Create the dictionary
        data = {w_n_i: (alpha_delta, phi)}

        # If you need it in a list (as per the type annotation)
        pole_periodic_terms = data


        body_settings.get( "Neptune" ).rotation_model_settings = environment_setup.rotation_model.iau_rotation_model(
                original_frame, target_frame, nominal_meridian,nominal_pole,rotation_rate,pole_precession,merdian_periodic_terms,pole_periodic_terms,angle_base_frame="J2000")

    elif settings_dict['Neptune_rot_model_type'] == 'Pole_Model_Jacobson2009':
        alpha_r = np.deg2rad(299.4608612607558) #as defined by Jacbson 2009 (check paper for uncertanties)
        delta_r = np.deg2rad(43.4048107907141) #as defined by Jacbson 2009 (check paper for uncertanties)
        epsilon = np.deg2rad(0.4616274249865)
        omega_0 = np.deg2rad(352.1753923868973) # 1989 August 25 needs to be adjusted to J2000
        omega_dot = np.deg2rad(52.3836218446110/36525/24/3600) # rad/sec
        w_dot = np.deg2rad(536.3128492) # rad/day  Not estimated, from Warwick et al. (1989).


        t0 = np.datetime64('1989-08-25T00:00:00')
        t1 = np.datetime64('2000-01-01T12:00:00')

        seconds_to_J2000 = (t1 - t0) / np.timedelta64(1, 's')
        omega_0 = omega_0 + omega_dot*seconds_to_J2000 #adjusted for J2000 !!
 
        alpha_0 = alpha_r
        alpha_1 = epsilon*(1/np.cos(delta_r))
        alpha_2 = -1/2*epsilon**2*np.tan(delta_r)/np.cos(delta_r)

        delta_0 = delta_r - 1/4*epsilon**2*np.tan(delta_r)
        delta_1 = -epsilon
        delta_2 = 1/4*epsilon**2*np.tan(delta_r)

        alpha_delta_1 = np.array([alpha_1, delta_1])
        alpha_delta_2 = np.array([alpha_2,delta_2])

        #Order in a way Tudat accepts
        #-----------------------------------------------------------------------------
        nominal_meridian = np.deg2rad(249.978) # W_0 from IAU (not relevant for this study)
        nominal_pole = np.array([alpha_0,delta_0]) #alpha_0 and delta_0
        rotation_rate=np.deg2rad(541.1397757/24/3600) # W_0_dot from IAU (not relevant for this study)
        pole_precession= np.array([0,0]) # alpha_0_dot and delta_0_dot are 0
        merdian_periodic_terms = {np.deg2rad(52.316/36525/24/3600): (np.deg2rad(-0.48), np.deg2rad(357.85))} #w_N_i, W_i, Phi_N_i in that order from IAU (not relevant)


        data = {omega_dot: (alpha_delta_1, omega_0),2*omega_dot:(alpha_delta_2,2*omega_0)}
        pole_periodic_terms = data



        #Assuming Jacboson 2009 is in J2000 frame and not ECLIPJ2000
        body_settings.get( "Neptune" ).rotation_model_settings = environment_setup.rotation_model.iau_rotation_model(
                original_frame, target_frame, nominal_meridian,nominal_pole,rotation_rate,pole_precession,merdian_periodic_terms,pole_periodic_terms,angle_base_frame="J2000")


    #---------------------------------------------------------------------------------------------------------------------------------------------------

    body_settings.get("Neptune").ephemeris_settings = environment_setup.ephemeris.direct_spice(
            global_frame_origin, global_frame_orientation)

    # body_settings.get("Earth").ephemeris_settings = environment_setup.ephemeris.direct_spice(
    #     global_frame_origin, global_frame_orientation)
        
    precession_nutation_theory = environment_setup.rotation_model.IAUConventions.iau_2006
    #original_frame = "J2000"
    # create rotation model settings and assign to body settings of "Earth"
    body_settings.get( "Earth" ).rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
    precession_nutation_theory, global_frame_orientation)


    # # Create system of selected bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)
    

    return body_settings,bodies


def Create_Acceleration_Models(settings_dict,system_of_bodies):

    # Define bodies that are propagated, and their central bodies of propagation
    bodies_to_propagate = settings_dict['bodies_to_propagate']
    central_bodies = settings_dict['central_bodies']
    bodies_to_simulate = settings_dict['bodies_to_simulate']
    bodies = settings_dict['bodies']
    #system_of_bodies = settings_dict['system_of_bodies']

    neptune_extended_gravity = settings_dict['neptune_extended_gravity']
    


    acts: Dict[str, List] = {}


    for body_name in bodies_to_simulate:
        if body_name == bodies_to_propagate[0]:
            continue  # no self-acceleration
        if body_name not in bodies:
            print("Body ",body_name,"not created in env")
            continue  # not created

        # Special case: Neptune
        if body_name == "Neptune":
            #if neptune_extended_gravity is "None":
            acts.setdefault("Neptune", []).append(
            propagation_setup.acceleration.point_mass_gravity())

            if neptune_extended_gravity != "None":
                acts = SetExtendedGravityNeptune(neptune_extended_gravity,acts)
            continue

        # Special case: Sun
        if body_name == "Sun":
            acts.setdefault("Sun", []).append(
                propagation_setup.acceleration.point_mass_gravity()
            )
            continue

        # Default for "most" other bodies: point-mass gravity
        acts.setdefault(body_name, []).append(
            propagation_setup.acceleration.point_mass_gravity()
        )

    # Create global accelerations settings dictionary.
    acceleration_settings = {bodies_to_propagate[0]: acts}


    # Create acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
        system_of_bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    accelerations_cfg = build_acceleration_config(settings_dict)
    return acceleration_models, accelerations_cfg

def SetExtendedGravityNeptune(neptune_extended_gravity,acts):
    
    if neptune_extended_gravity == "Jacobson2009":
        acts.setdefault("Neptune", []).append(
        propagation_setup.acceleration.spherical_harmonic_gravity(4, 0))
    return acts


def build_acceleration_config(settings_dict):
    bodies_to_propagate = settings_dict['bodies_to_propagate']
    central_bodies = settings_dict['central_bodies']
    bodies_to_simulate = settings_dict['bodies_to_simulate']
    bodies = settings_dict['bodies']
    target = bodies_to_propagate[0]
    neptune_extended_gravity = settings_dict['neptune_extended_gravity']
    
    cfg_acts = {}  # acting_body -> list of model descriptors
    for body_name in bodies_to_simulate:
        if body_name == target:
            continue  # no self-acceleration
        if body_name not in bodies:
            # body not present in the environment; skip
            continue

        models = []

        if body_name == "Neptune":
            models.append({"type": "point_mass_gravity"})
            if neptune_extended_gravity == "Jacobson2009":
                models.append({"type": "spherical_harmonic_gravity_Jacobson_2009", "degree": 4, "order": 0})
        else:
            # Sun, Jupiter, Saturn, etc. → default point-mass gravity
            models.append({"type": "point_mass_gravity"})

        cfg_acts[body_name] = models

    # Final YAML-friendly dict (keeps Tudat's nesting: {target: {...}})
    accel_cfg = {
        "target": target,
        "accelerations": cfg_acts
    }
    return accel_cfg



def Create_Propagator_Settings(settings_dict,acceleration_models):

    simulation_start_epoch = settings_dict['start_epoch'] 
    simulation_end_epoch   = settings_dict['end_epoch']
    simulation_initial_epoch = settings_dict['initial_epoch']
    bodies_to_propagate = settings_dict['bodies_to_propagate'][0]
    central_bodies = settings_dict['central_bodies'][0]
    global_frame_orientation = settings_dict['global_frame_orientation']
    fixed_step_size = settings_dict['fixed_step_size']
    
    #Get initial state OR assign SPICE if not given
    initial_state = settings_dict.get('initial_state', None)

    if initial_state is None:
        initial_state = spice.get_body_cartesian_state_at_epoch(
            target_body_name       = bodies_to_propagate,
            observer_body_name     = central_bodies,
            reference_frame_name   = global_frame_orientation,
            aberration_corrections = "none",
            ephemeris_time         = simulation_initial_epoch,
        )

    #TODO Arrange this better and make expandable
    dependent_variables_to_save = [
        propagation_setup.dependent_variable.rsw_to_inertial_rotation_matrix("Triton", "Neptune"),
        #propagation_setup.dependent_variable.total_acceleration("Triton"),
        propagation_setup.dependent_variable.keplerian_state("Triton", "Neptune"),
        propagation_setup.dependent_variable.latitude("Triton", "Neptune"),
        propagation_setup.dependent_variable.longitude("Triton", "Neptune"),
        
    ]


    # Create termination settings
    termination_condition_end = propagation_setup.propagator.time_termination(simulation_end_epoch)

    termination_condition_start = propagation_setup.propagator.time_termination(simulation_start_epoch)

    termination_settings = propagation_setup.propagator.non_sequential_termination(termination_condition_end,termination_condition_start)

    # Create numerical integrator settings
    #fixed_step_size = Time(60*30) # 30 minutes
    integrator_settings = create_rkf78_integrator_settings(fixed_step_size)
    #integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(Time(fixed_step_size), coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4)

    # Create propagation settings
    propagator_settings = propagation_setup.propagator.translational(
        [central_bodies],
        acceleration_models,
        [bodies_to_propagate],
        initial_state,
        simulation_initial_epoch,
        integrator_settings,
        termination_settings,
        output_variables=dependent_variables_to_save
    )
    return propagator_settings

def create_rkf78_integrator_settings(timestep=60):

    # Create numerical integrator settings
    fixed_step_size = timestep
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_78
    )

    return integrator_settings



def make_relative_position_pseudo_observations(start_epoch,end_epoch, system_of_bodies, settings_dict: dict):
    settings = settings_dict['obs']
    cadence = settings['cadence']

    observation_start_epoch = start_epoch
    observation_end_epoch = end_epoch
    
    observation_times = np.arange(observation_start_epoch + cadence, observation_end_epoch - cadence, cadence)

    observation_times = np.array([t for t in observation_times])
    observation_times_test = observation_times[0]
    bodies_to_observe = settings["bodies"]
    relative_position_observation_settings = []

    for obs_bodies in bodies_to_observe:
        
        # link_ends = {
        #     estimation_setup.observation.observed_body: estimation_setup.observation.body_origin_link_end_id(obs_bodies[0]),
        #     estimation_setup.observation.observer: estimation_setup.observation.body_origin_link_end_id(obs_bodies[1])}
        
        link_ends = dict()
        
        link_ends[links.observed_body] = links.body_origin_link_end_id(obs_bodies[0])
        link_ends[links.observer] = links.body_origin_link_end_id(obs_bodies[1])
        
        
        #link_definition = estimation_setup.observation.LinkDefinition(link_ends)
        link_definition = links.LinkDefinition(link_ends)

        relative_position_observation_settings.append(model_settings.relative_cartesian_position(link_definition))

        observation_simulation_settings = [observations_setup.observations_simulation_settings.tabulated_simulation_settings(
                model_settings.relative_position_observable_type,
                link_definition,
                observation_times,
                reference_link_end_type = links.LinkEndType.observed_body)] #estimation_setup.observation.observed_body


    # Create observation simulators
    ephemeris_observation_simulators = observations_setup.observations_simulation_settings.create_observation_simulators(
        relative_position_observation_settings, system_of_bodies)

    # Get ephemeris states as ObservationCollection
    print('Checking spice for position pseudo observations...')
    simulated_pseudo_observations = observations_setup.observations_wrapper.simulate_observations(
        observation_simulation_settings,
        ephemeris_observation_simulators,
        system_of_bodies)

    return simulated_pseudo_observations, relative_position_observation_settings


def Create_Estimation_Output(settings,system_of_bodies,propagator_settings,pseudo_observations_settings,pseudo_observations):

    #pseudo_observations_settings = settings_dict['pseudo_observations_settings']
    #pseudo_observations = settings_dict['pseudo_observations']

    parameters_to_estimate_settings = parameters_setup.initial_states(propagator_settings, system_of_bodies)
    
    #Only use with simple rotational model!
    if "Rotation_Pole_Position_Neptune" in settings['est']["est_parameters"]:
        parameters_to_estimate_settings.append(parameters_setup.rotation_pole_position('Neptune'))
    
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
    if 'GM_Neptune' in settings['est']['est_parameters']:
        parameters_to_estimate_settings.append(parameters_setup.gravitational_parameter('Neptune'))
    if 'GM_Triton' in settings['est']['est_parameters']:
        parameters_to_estimate_settings.append(parameters_setup.gravitational_parameter('Triton'))

    if 'spherical_harmonics' in settings['est']['est_parameters']:
        block_indices = [
        (2, 0),  # C20 (J2)
        (4, 0)   # C40 (J4)
        ]

        # Create the estimatable parameter for these specific coefficients
        parameters_to_estimate_settings.append(
            parameters_setup.spherical_harmonics_c_coefficients_block(
                body="Neptune",
                block_indices=block_indices
            )
        )



    parameters_to_estimate = parameters_setup.create_parameter_set(
        parameters_to_estimate_settings,
        system_of_bodies,
        propagator_settings)

    original_parameter_vector = parameters_to_estimate.parameter_vector

    print('Running propagation...')

    estimator = estimation_analysis.Estimator(
        system_of_bodies, 
        parameters_to_estimate,
        pseudo_observations_settings,
        propagator_settings)

    convergence_settings = estimation_analysis.estimation_convergence_checker(maximum_iterations=5)

    
    ############################################################################################################
    #CREATE INVERSE A PRIORI COVARIANCE
    ############################################################################################################
    inverse_apriori_cov = []
    if settings['est']['a_priori_covariance'] == True:
        # Get parameter indices
        n_params = parameters_to_estimate.parameter_set_size
        # Parameter identifies IAU (pole/lib are not working yet 
        parameter_identifies = parameters_to_estimate.get_parameter_identifiers()
        pole_pos_identifier = parameter_identifies[0]
        pole_lib_identifier = parameter_identifies[1]
        
        # Create inverse a priori covariance with zeros (no constraint by default)
        inverse_apriori_cov = np.zeros((n_params, n_params))
        
        # Get indices for pole position and librations (rotation model parameters)
        if settings['est']['a_priori_pole'] == True:
            pole_indices = parameters_to_estimate.indices_for_parameter_type(pole_pos_identifier)
            # Apply constraints to pole position
            # Example: uncertainty in pole right ascension and declination
            for idx_range in pole_indices:
                for i in range(idx_range[0], idx_range[0]+idx_range[1]):
                    inverse_apriori_cov[i, i] = 1 / (0.2 * np.pi/180)**2  # Inverse of 0.2 degree uncertainty
        
        if settings['est']['a_priori_lib'] == True:
            libration_indices = parameters_to_estimate.indices_for_parameter_type(pole_lib_identifier)
            
            if settings['est']['a_priori_lib_deg'] == 1:
                # Apply constraints to libration amplitudes
                # Conservative uncertainties: [alpha_1, delta_1]
                libration_sigmas = [0.003, 0.002]
                libration_sigmas[0] = (2 * np.pi/180)  # now 2 deg, prev alpha_1 ~300% of 0.011 rad,  0.01*3
                libration_sigmas[1] = (2 * np.pi/180)  # now 2 deg, prev delta_1 ~300% of 0.008 rad,  0.01*3
                
            if settings['est']['a_priori_lib_deg'] == 2:
                libration_sigmas = [0.003, 0.002, 2e-5, 1e-5]
                libration_sigmas[0] = np.abs(0.01*3)  # alpha_1 ~300% of 0.011 rad 
                libration_sigmas[1] = np.abs(0.01*3)  # delta_1 ~300% of 0.008 rad 
                libration_sigmas[2] = np.abs(4.2e-5*1)   # alpha_2 ~100% of 4.2e-5 rad
                libration_sigmas[3] = np.abs(4.2e-5*1)   # delta_2 ~100% of 1.5e-5 rad 
            
            for idx_range in libration_indices:
                for j, i in enumerate(range(idx_range[0], idx_range[0]+idx_range[1])):
                    param_idx = j % 2  # Cycles through 0,1,2,3 if multiple sets
                    inverse_apriori_cov[i, i] = 1 / libration_sigmas[param_idx]**2        
    ############################################################################################################
    # Create input object for the estimation
    estimation_input = estimation_analysis.EstimationInput(
        observations_and_times=pseudo_observations,
        inverse_apriori_covariance=inverse_apriori_cov,
        convergence_checker=convergence_settings)

    # Set methodological options
    estimation_input.define_estimation_settings(save_state_history_per_iteration=True)

    # Perform the estimation
    print('Performing the estimation...')
    print(f'Original initial states: {original_parameter_vector}')
    parameters_desc = parameters_to_estimate.get_parameter_descriptions()


    estimation_output = estimator.perform_estimation(estimation_input)
    state_history = estimation_output.simulation_results_per_iteration[-1].dynamics_results.state_history_float
    state_history_array = util.result2array(state_history)
    epochs = state_history_array[:,0] 
    #Compute covariances
    propagated_covariances = []
    if settings['obs']['use_weights'] == True:
        propagated_covariances = estimation_analysis.propagate_covariance_from_analysis_objects(
            estimation_output,
            estimator.state_transition_interface,
            epochs
        )
        propagated_covariances = np.array(list(propagated_covariances.values()))
    
    return estimation_output, original_parameter_vector, parameters_desc,propagated_covariances



########################################################################################################################################################
# MANUALLY SIMULATE IAU ROTATION MODEL
########################################################################################################################################################
def PoleModel(time_column,parameter_update=[],model_type='IAU'):
    """
    Simulates a standard IAU rotation model and outputs the pole position (right ascension and declination in rad).
        
    Parameters:
    -----------
    time_column : np.ndarray
        Time in seconds from J2000 epoch, shape (N, 1) or (N,)
    model_type : str = IAU
        model type to select the model parameters either IAU or Jacobson2009
    Returns:
    --------
    alpha_array : np.ndarray
        right ascension in rad shape (N, 1) 
    delta_array : np.ndarray
        declination in rad shape (N, 1) 
    """

    alpha_array = np.array([])
    delta_array = np.array([])

    

    delta_t = time_column #exactly what we need

    
    #parameter_update = [alpha_0,delta_0,alpha_dot,delta_dot,alpha_dot_1,delta_dot_1,alpha_dot_2,delta_dot_2]

    #IAU 2015 values
    if model_type == 'IAU':
        deg = 1
        alpha_0 = np.deg2rad(299.36) + parameter_update[0]
        delta_0 = np.deg2rad(43.46) + parameter_update[1]
        alpha_dot = 0 + parameter_update[2]
        delta_dot = 0 + parameter_update[3]
        alpha_dot_1 = np.deg2rad(0.7) + parameter_update[4]
        delta_dot_1 = np.deg2rad(-0.51) + parameter_update[5]
        omega_1 = np.deg2rad(52.316/36525/24/3600)
        phi_1 =  np.deg2rad(357.85)

    if model_type == 'Jacobson2009':
        deg = 2
        alpha_r = np.deg2rad(299.4608612607558) #as defined by Jacbson 2009 (check paper for uncertanties)
        delta_r = np.deg2rad(43.4048107907141) #as defined by Jacbson 2009 (check paper for uncertanties)
        epsilon = np.deg2rad(0.4616274249865)
        omega_0 = np.deg2rad(352.1753923868973) # 1989 August 25 needs to be adjusted to J2000
        omega_dot = np.deg2rad(52.3836218446110/36525/24/3600) # rad/sec
        w_dot = np.deg2rad(536.3128492) # rad/day  Not estimated, from Warwick et al. (1989).


        t0 = np.datetime64('1989-08-25T00:00:00')
        t1 = np.datetime64('2000-01-01T12:00:00')

        seconds_to_J2000 = (t1 - t0) / np.timedelta64(1, 's')
        omega_0 = omega_0 + omega_dot*seconds_to_J2000

       
 
        #--------------------------------------------------------------
        alpha_0 = alpha_r + parameter_update[0]
        delta_0 = delta_r - 1/4*epsilon**2*np.tan(delta_r) + parameter_update[1]
        
        alpha_dot = 0 + parameter_update[2] + parameter_update[2]
        delta_dot = 0 + parameter_update[3] + parameter_update[3]
        
        alpha_dot_1 = epsilon*(1/np.cos(delta_r)) + parameter_update[4]
        delta_dot_1 = -epsilon + parameter_update[5]
        
        alpha_dot_2 = -1/2*epsilon**2*np.tan(delta_r)/np.cos(delta_r) + parameter_update[6]
        delta_dot_2 = 1/4*epsilon**2*np.tan(delta_r) + parameter_update[7]

        omega_1 = omega_dot
        phi_1 =  omega_0
        omega_2 = 2*omega_dot
        phi_2 = 2*omega_0

    #------------------------------------------------------------------------------------------
    omega_t_phi = omega_1 * delta_t + phi_1
    if deg == 1:
        alpha_array = alpha_0 + alpha_dot*delta_t + alpha_dot_1*np.sin(omega_t_phi)
        delta_array = delta_0 + delta_dot*delta_t + delta_dot_1*np.cos(omega_t_phi)
    if deg == 2:
        omega_2_t_phi = omega_2*delta_t + phi_2
        alpha_array = alpha_0 + alpha_dot*delta_t + alpha_dot_1*np.sin(omega_t_phi) + alpha_dot_2*np.sin(omega_2_t_phi)
        delta_array = delta_0 + delta_dot*delta_t + delta_dot_1*np.cos(omega_t_phi) + delta_dot_2*np.cos(omega_2_t_phi)

    return alpha_array,delta_array