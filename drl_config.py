import numpy as np

#The master config file for the LV_VAE environment
def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return rad * 180 / np.pi

lv_vae_config = {
#General parameters    
    "step_size"                 : 0.01,             # Step size of the physics simulation
    "max_t_steps"               : 6000,             # Maximum number of timesteps in the DRL simulation before it is terminated
    "mesh_path"                 : "./gym_quad/meshes/sphere.obj", # Path to the mesh of the sphere obstacle #TODO idk if this should be here might move it
    "enclose_scene"             : True,             # Enclose the scene with a box thats scaled to the scene size
    "padding"                   : 1.5,              # Padding of the box that encloses the scene [m] #usually 1.5m for indoor training
    "drone_radius_for_collision": 0.13,             # Radius of the drone for collision detection [m] #Actual armlength to propeller is 0.13m armlength frame extremas 0.25m. height is 0.21m
    "drone_height_for_collision": 0.11,             # Height of the drone for collision detection [m]
    "use_drone_mesh"            : False,            # Use the drone mesh for collision detection if not a cylinder is used using r and h above
    "use_uncaged_drone_mesh"    : False,            # Use the uncaged mesh for collision detection
    "recap_chance"              : 0.1,              # Chance of recapitulating a previous trainig scenario
#Noise parameters #TODO add the noise values here as well?    
    "perturb_sim"               : False,            # Activates all the noise below. Also, the "perturb" scenarios inside LV_VAE_MESH.py sets this to True
    "perturb_domain"            : False,            # Perturb the domain observation
    "perturb_IMU"               : False,            # Perturb the IMU data
    "perturb_depth_map"         : False,            # Perturb the depth map with noise
    "perturb_camera_pose"       : False,            # Perturb the camera pose
    "perturb_ctrl_gains"        : False,            # Perturb the control gains
    "perturb_latency"           : False,            # Perturb the latency of the depth camera sensor
#Depth camera parameters        
    "FOV_horizontal"            : 75,               # Horizontal field of view of the depth camera
    "FOV_vertical"              : 62,               # Vertical field of view of the depth camera
    "depth_map_size"            : (240, 320),       # Size of the depth map Earlier sensor suite (H,W)
    "max_depth"                 : 10,               # Maximum depth of the depth camera
    "camera_FPS"                : 30,               # Frequency of the sensors
#VAE parameters     
    "compressed_depth_map_size" : 224,              # Size of depth map after compression
    "latent_dim"                : 64,               # Dimension of the latent space
#Path related parameters    
    "la_dist"                   : 3.5,              # Look ahead distance aka distance to the point on path to be followed. #LAidst 0.5 is nice in house
    "accept_rad"                : 0.5,              # Acceptance radius for the quadcopter to consider the end as reached    
    "n_waypoints"               : 12,               # Number of waypoints to be generated
    "segment_length"            : 2.5,              # Length of the segments between waypoints [m]
    "line_path_range"           : (3,10),           # Range of the line path length [m] affects scenario line
    "new_3d_path_range"         : (10,20),          # Range of the line path length [m] affects scenario easy, easy_random, intermediate, proficient
    "new_3d_up_down_path_range" : (10,25),          # Range of the line path length [m] affects scenario advanced, expert
    "relevant_dist_to_path"     : 8,                # Distance to the path where the observation will yield values between -1 and 1
#Drone controller parameters    
    "s_max"                     : 2,                # Maximum speed of the quadcopter m/s #2.5m/s*3.6 = 9km/h  #Speed 1 is nice in house
    "i_max"                     : deg2rad(65/2),    # Maximum inclination angle of commanded velocity wrt x-axis #Approx half of vertical FOV restricts drone to fly where it can see
    "r_max"                     : deg2rad(60),      # Maximum commanded yaw rate rad/s
    "kv"                        : 2.5,   #ØD 2,     # Velocity proportional gain             All tuned in test_controller.py 2.5, 0.8, 0.8 used a lot
    "kR"                        : 2.5,   #ØD 2,     # Attitude proportional gain             
    "kangvel"                   : 0.35,  #ØD 0.3,   # Angular velocity damping gain 
#Reward parameters
    "min_reward"                : -4e4,             # Minimum reward before the simulation is terminated 
                                                    #In expert i often accumulates less than 2000 when existence is -8 and min ca is -16
    
    #Path adherence reward
    'PA_band_edge'              : 4,                # edge of Path adherence band
    'PA_scale'                  : 2.5,              # scale of Path adherence reward [-PA_scale, PA_scale]
                                #2.8 might be better? see exp25 proficient_perturbed    
    #Path progression reward
    'PP_rew_max'                : 2,                # maximum reward for path progression
    'PP_rew_min'                : -1,               # minimum reward for path progression
    'PP_rew_scale'              : 0,              # scale of path progression reward
                                #2 might be better? see exp25 proficient_perturbed
    #Collision reward
    'rew_collision'             : -1000,            # reward (penalty) for collision
    
    #reach end reward
    'rew_reach_end'             : 1000,             # reward for reaching the end of the path

    #pass wp reward
    'rew_pass_wp'               : 1000/8,          # reward for passing a waypoint

    #Approach end lambda interpolation
    'approach_end_range'        : 3,                # Dist[m] between goal and drone where Lambda CA and Lambda PA interpolate such that pa>ca

    #Existence reward   
    'existence_reward'          : -7,               # reward (penalty) for existing
    
    #Collision avoidance                         
    'CA_scale'                  : 1/1000,           # Scaling of the collision avoidance reward Found via tuning
    'CA_epsilon'                : 0.0001,           # Small number to avoid division by zero
    'TwoDgauss_sigma'           : 30,               # Sigma of the 2D gaussian for the collision avoidance reward
    'TwoDgauss_peak'            : 1.5,              # Peak value at the center of the 2D gaussian (Amplitude)
    'min_CA_rew'                : -16.0,            # Minimum reward for collision avoidance #-20 is too penalizing I think
    'max_CA_rew'                : 0,                # Max reward for collision avoidance 

    #Lambda interpolation       
    'lambda_PA_max'             : 1,                # Maximum lambda value for path adherence
    'lambda_CA_max'             : 1,                # Maximum lambda value for collision avoidance

    'lambda_PA_min'             : 0.1,              # Minimum lambda value for path adherence
    'lambda_CA_min'             : 0.1,              # Minimum lambda value for collision avoidance
}