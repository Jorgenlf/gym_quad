from gymnasium.envs.registration import register
import numpy as np

def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return rad * 180 / np.pi

lv_vae_config = {
#General parameters    
    "step_size"                 : 0.01,          # Step size of the physics simulation
    "max_t_steps"               : 8000,          # Maximum number of timesteps in the DRL simulation before it is terminated
    "mesh_path"                 : "./gym_quad/meshes/sphere.obj", # Path to the mesh of the sphere obstacle #TODO idk if this should be here might move it
    "enclose_scene"             : True,          # Enclose the scene with a box thats scaled to the scene size
    "padding"                   : 1.5,           # Padding of the box that encloses the scene [m] #usually 1.5m for indoor training
    "drone_radius_for_collision": 0.10,          # Radius of the drone for collision detection [m] #Actual radius is 0.25m
    "recap_chance"              : 0.1,           # Chance of recapitulating a previous trainig scenario
#Noise parameters #TODO add the noise values here as well?    
    "perturb_sim"               : False,         # Activates all the noise below. Also, the perturb scenarios inside LV_VAE_MESH.py sets this to True
    "perturb_domain"            : False,         # Perturb the domain observation
    "perturb_IMU"               : False,         # Perturb the IMU data
    "perturb_depth_map"         : False,         # Perturb the depth map with noise
    "perturb_camera_pose"       : False,         # Perturb the camera pose
    "perturb_ctrl_gains"        : False,         # Perturb the control gains
    "perturb_latency"           : False,         # Perturb the latency of the sensors
#Depth camera parameters    
    "FOV_horizontal"            : 75,            # Horizontal field of view of the depth camera
    "FOV_vertical"              : 62,            # Vertical field of view of the depth camera
    "depth_map_size"            : (240, 320),    # Size of the depth map Earlier sensor suite
    "max_depth"                 : 10,            # Maximum depth of the depth camera
    "camera_FPS"                : 30,            # Frequency of the sensors
#VAE parameters    
    "compressed_depth_map_size" : 224,           # Size of depth map after compression
    "latent_dim"                : 64,            # Dimension of the latent space
#Path related parameters
    "la_dist"                   : 2.5,           # Look ahead distance aka distance to the point on path to be followed. old:20  
    "accept_rad"                : 1.0,           # Acceptance radius for the quadcopter to consider the end as reached old:5     
    "n_waypoints"               : 6,             # Number of waypoints to be generated
    "segment_length"            : 5,             # Length of the segments between waypoints
    "relevant_dist_to_path"     : 5,             # Distance to the path where the observation will yield values between -1 and 1
#Drone controller parameters
    "s_max"                     : 2,           # Maximum speed of the quadcopter m/s #2.5m/s*3.6 = 9km/h  
    "i_max"                     : deg2rad(65/2), # Maximum inclination angle of commanded velocity wrt x-axis #Approx half of vertical FOV restricts drone to fly where it can see
    "r_max"                     : deg2rad(60),   # Maximum commanded yaw rate rad/s
    "kv"                        : 1.5,           # Velocity proportional gain             All tuned in test_controller.py 2.5, 0.8, 0.8 used a lot
    "kangvel"                   : 0.8,           # Angular velocity damping gain 
    "kR"                        : 0.5,           # Attitude proportional gain             
#Reward parameters
    "min_reward"                : -1.5e4,        # Minimum reward before the simulation is terminated
    
    #Path adherence reward
    'PA_band_edge'              : 3,             # edge of Path adherence band
    'PA_scale'                  : 2.8,           # scale of Path adherence reward [-PA_scale, PA_scale]
    
    #Path progression reward
    'PP_rew_max'                : 2,             # maximum reward for path progression
    'PP_rew_min'                : -1,            # minimum reward for path progression
    
    #Collision reward
    'rew_collision'             : -50,           # reward (penalty) for collision
    
    #reach end reward
    'rew_reach_end'             : 200,           # reward for reaching the end of the path

    #Approach_end reward
    "approach_end_sigma"        : 0.8,           # Sigma of the gaussian for the approach end reward
    "max_approach_end_rew"      : 3,             # Maximum reward for the approach end reward
    
    #Existence reward
    'existence_reward'          : -1.0,          # reward (penalty) for existing
    
    #Collision avoidance                         #Think the new one is superior
    'use_old_CA_rew'            : False,         # Wether to use the old or new collision avoidance reward function
        #Collision avoidance "old"
        'danger_range'              : 10,           # Range between quadcopter and obstacle within which the quadcopter is in danger
        'abs_inv_CA_min_rew'        : 1/16,         # 1/x -> -x is min reward per CA fcn range and angle --> rangefcn + anglefcn = -2*x 
    
        #Collision avoidance "new"
        'CA_scale'                  : 1/1000,        # Scaling of the collision avoidance reward Found via tuning
        'CA_epsilon'                : 0.0001,        # Small number to avoid division by zero
        'TwoDgauss_sigma'           : 60,            # Sigma of the 2D gaussian for the collision avoidance reward
        'TwoDgauss_peak'            : 1.5,           # Peak value at the center of the 2D gaussian
        'min_CA_rew'                : -18,           # Minimum reward for collision avoidance #-20 is too penalizing I think
}

register(
    id='LV_VAE-v0',
    entry_point='gym_quad.envs:LV_VAE',
    kwargs={'env_config': lv_vae_config}
)

register(
    id='LV_VAE_MESH-v0',
    entry_point='gym_quad.envs:LV_VAE_MESH',
    kwargs={'env_config': lv_vae_config}
)


#OLD vv
waypoint_planner_config = {
    "step_size": 0.01,
    "max_t_steps": 60000,
    "min_reward": -20000,
    "n_obs_states": 6,
    "cruise_speed": 2.5,
    "lambda_reward": 0.5,
    "reward_path_following_c": 5,
    "reward_path_following_max": 1,
    "sensor_span": (360, 180), # the horizontal and vertical span of the sensors
    "sensor_suite": (15, 15), # the number of sensors covering the horizontal and vertical span
    "sensor_input_size": (8, 8), # the shape of FLS data passed to the neural network. Max pooling from raw data is used
    "sensor_frequency": 2,
    "sonar_range": 25,
    "n_obs_errors": 8,
    "n_obs_inputs": 0,
    "la_dist": 5,
    "accept_rad": 1,
    "n_waypoints": 7,
    "n_int_obstacles": 5,
    "n_pro_obstacles": 3,
    "n_adv_obstacles": 8,
    "n_fictive_waypoints": 1,
    "fictive_waypoint_span": 10,
    "n_generated_waypoints": 1,
    "simulation_frequency": 10, # How many timesteps to simulate the quadcopter for each new path
    "rl_mode": "path_planning" # [path_planning, desired_acc]
}

register(
    id='WaypointPlanner-v0',
    entry_point='gym_quad.envs:WaypointPlanner',
    kwargs={'env_config': waypoint_planner_config}
)