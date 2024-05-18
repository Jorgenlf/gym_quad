import numpy as np

#The master config file for the LV_VAE environment
def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return rad * 180 / np.pi

lv_vae_config = {
#General parameters    
    "step_size"                 : 0.01,          # Step size of the physics simulation
    "max_t_steps"               : 6000,          # Maximum number of timesteps in the DRL simulation before it is terminated
    "mesh_path"                 : "./gym_quad/meshes/sphere.obj", # Path to the mesh of the sphere obstacle #TODO idk if this should be here might move it
    "enclose_scene"             : True,          # Enclose the scene with a box thats scaled to the scene size
    "padding"                   : 1.5,           # Padding of the box that encloses the scene [m] #usually 1.5m for indoor training
    "drone_radius_for_collision": 0.10,          # Radius of the drone for collision detection [m] #Actual radius is 0.25m
    "recap_chance"              : 0.1,           # Chance of recapitulating a previous trainig scenario
#Noise parameters #TODO add the noise values here as well?    
    "perturb_sim"               : False,         # Activates all the noise below. Also, the "perturb" scenarios inside LV_VAE_MESH.py sets this to True
    "perturb_domain"            : False,         # Perturb the domain observation
    "perturb_IMU"               : False,         # Perturb the IMU data
    "perturb_depth_map"         : False,         # Perturb the depth map with noise
    "perturb_camera_pose"       : False,         # Perturb the camera pose
    "perturb_ctrl_gains"        : False,         # Perturb the control gains
    "perturb_latency"           : False,         # Perturb the latency of the sensors
#Depth camera parameters    
    "FOV_horizontal"            : 75,            # Horizontal field of view of the depth camera
    "FOV_vertical"              : 62,            # Vertical field of view of the depth camera
    "depth_map_size"            : (240, 320),    # Size of the depth map Earlier sensor suite (H,W)
    "max_depth"                 : 10,            # Maximum depth of the depth camera
    "camera_FPS"                : 30,            # Frequency of the sensors
#VAE parameters    
    "compressed_depth_map_size" : 224,           # Size of depth map after compression
    "latent_dim"                : 64,            # Dimension of the latent space
#Path related parameters
    "la_dist"                   : 2.8,           # Look ahead distance aka distance to the point on path to be followed. old:20  #LAidst 0.5 is nice in house
    "accept_rad"                : 0.8,           # Acceptance radius for the quadcopter to consider the end as reached old:5     
    "n_waypoints"               : 6,             # Number of waypoints to be generated
    "segment_length"            : 5,             # Length of the segments between waypoints
    "relevant_dist_to_path"     : 8,             # Distance to the path where the observation will yield values between -1 and 1
#Drone controller parameters
    "s_max"                     : 2,             # Maximum speed of the quadcopter m/s #2.5m/s*3.6 = 9km/h  #Speed 1 is nice in house
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
    'PP_rew_min'                : -0.8,            # minimum reward for path progression
    
    #Collision reward
    'rew_collision'             : -60,           # reward (penalty) for collision
    
    #reach end reward
    'rew_reach_end'             : 200,           # reward for reaching the end of the path

    #Approach_end reward
    "approach_end_sigma"        : 0.8,           # Sigma of the gaussian for the approach end reward
    "max_approach_end_rew"      : 3,             # Maximum reward for the approach end reward
    
    #Existence reward
    'existence_reward'          : -1,            # reward (penalty) for existing
    
    #Collision avoidance                         #Think the new one is superior
    'use_old_CA_rew'            : False,         # Wether to use the old or new collision avoidance reward function
        #Collision avoidance "old"
        'danger_range'              : 10,           # Range between quadcopter and obstacle within which the quadcopter is in danger
        'abs_inv_CA_min_rew'        : 1/16,         # 1/x -> -x is min reward per CA fcn range and angle --> rangefcn + anglefcn = -2*x 
    
        #Collision avoidance "new"
        'CA_scale'                  : 1/1000,        # Scaling of the collision avoidance reward Found via tuning
        'CA_epsilon'                : 0.0001,        # Small number to avoid division by zero
        'TwoDgauss_sigma'           : 30,            # Sigma of the 2D gaussian for the collision avoidance reward
        'TwoDgauss_peak'            : 1,             # Peak value at the center of the 2D gaussian
        'min_CA_rew'                : -16.5,         # Minimum reward for collision avoidance #-20 is too penalizing I think
}
