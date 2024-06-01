import numpy as np
import torch
import trimesh
import gymnasium as gym
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import gym_quad.utils.geomutils as geom
import gym_quad.utils.state_space as ss
from gym_quad.utils.ODE45JIT import j_Rzyx
from gym_quad.utils.geomutils import enu_to_pytorch3d, enu_to_tri, pytorch3d_to_enu, tri_Rotmat
from gym_quad.objects.quad import Quad
from gym_quad.objects.IMU import IMU
from gym_quad.objects.QPMI import QPMI, generate_random_waypoints
from gym_quad.objects.depth_camera import DepthMapRenderer, FoVPerspectiveCameras, RasterizationSettings, PerspectiveCameras
from gym_quad.objects.mesh_obstacles import Scene, SphereMeshObstacle, CubeMeshObstacle, get_scene_bounds, ImportedMeshObstacle, advanced_create_cylinder

#Helper functions
def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return rad * 180 / np.pi

def m1to1(value, min, max): 
    '''
    Normalizes a value from the range [min,max] to the range [-1,1]
    If value is outside the min max range, it will be clipped to the min or max value (ensuring we return a value in the range [-1,1])
    '''
    value_normalized = 2.0*(value-min)/(max-min) - 1
    return np.clip(value_normalized, -1, 1)

def invm1to1(value, min, max):
    '''
    Inverse normalizes a value from the range [-1,1] to the range [min,max]
    If value that got normalized was outside the min max range it may only be inverted to the min or max value (will not be correct if the clip was used in the normalization)
    '''
    return (value+1)*(max-min)/2.0 + min


class LV_VAE_MESH(gym.Env):
    '''Creates an environment where the actionspace consists of Linear body velocity and yaw rate is passed to a P-PD controller,
    while the observationspace uses a Varial AutoEncoder "plus more" for observations of environment.'''

    def __init__(self, env_config, scenario="line"):
        # print("ENVIRONMENT: LV_VAE_MESH") #Used to verify that the correct environment is being used
        # Set all the parameters from GYM_QUAD/qym_quad/__init__.py as attributes of the class
        for key in env_config:
            setattr(self, key, env_config[key])

        #Actionspace mapped to speed, inclination of velocity vector wrt x-axis and yaw rate
        self.action_space = gym.spaces.Box(
            low = np.array([-1,-1,-1], dtype=np.float32),
            high = np.array([1, 1, 1], dtype=np.float32),
            dtype = np.float32
        )

        #Observationspace
        #Depth camera observation space
        self.perception_space = gym.spaces.Box( 
            low = 0,
            high = 1,
            shape = (1, self.compressed_depth_map_size, self.compressed_depth_map_size),
            dtype = np.float32
        )

        # IMU observation space
        self.IMU_space = gym.spaces.Box(
            low = -1,
            high = 1,
            shape = (6,),
            dtype = np.float32
        )

        #Domain observation space (Angles, distances and coordinates in body frame)
        self.domain_space = gym.spaces.Box(
            low = -1,
            high = 1,
            # shape = (19,),
            shape = (22,),
            dtype = np.float32
        )

        self.observation_space = gym.spaces.Dict({
        'perception': self.perception_space,
        'IMU': self.IMU_space,
        'domain': self.domain_space
        })

        #Scenario set up
        self.scenario = scenario
        self.obstacles = [] #Filled in the scenario functions
        self.scenario_switch = {
            
            # Training scenarios, all functions defined at the bottom of this file
            "line"                  : self.scenario_line,
            "line_up"               : self.scenario_line_up,
            "xy_line"               : self.scenario_xy_line,                
            "squiggly_line_xy_plane": self.scenario_squiggly_line_xy_plane,
            "3d_new"                : self.scenario_3d_new,
            "3d_up_down"            : self.scenario_3d_up_down,
            
            "easy"                  : self.scenario_easy,
            "easy_random"           : self.scenario_random_pos_att_easy, #All training scenarios that are "harder than easy random" also have random pos and att
            "easy_perturbed"        : self.scenario_easy_perturbed_sim,
            
            "intermediate"          : self.scenario_intermediate,
            
            "proficient"            : self.scenario_proficient,
            "proficient_perturbed"  : self.scenario_proficient_perturbed_sim,
            
            "advanced"              : self.scenario_advanced,

            "expert"                : self.scenario_expert,
            "expert_perturbed"      : self.scenario_expert_perturbed_sim,
            
            #Special
            "random_corridor"       : self.scenario_random_corridor,
            # Testing scenarios
            #Agent testing
            "helix"                 : self.scenario_helix,
            "test_path"             : self.scenario_test_path,
            "test"                  : self.scenario_test,
            "house"                 : self.scenario_house,
            "house_easy"            : self.scenario_house_easy,
            "house_hard"            : self.scenario_house_hard,
            "house_easy_obstacles"  : self.scenario_house_easy_obstacles,
            "house_hard_obstacles"  : self.scenario_house_hard_obstacles,
            "horizontal"            : self.scenario_horizontal_test,
            "vertical"              : self.scenario_vertical_test,
            "deadend"               : self.scenario_deadend_test,
            #Dev testing
            "crash"             : self.scenario_dev_test_crash,
            "crash_cube"        : self.scenario_dev_test_cube_crash,
            "obs_at_end"        : self.scenario_dev_test_obs_at_end,
            "outskirt"          : self.scenario_dev_test_obs_in_outskirt,
            "outskirt_and_end"  : self.scenario_dev_test_outskirt_and_end,
            "dev_obsgen_plane"  : self.scenario_dev_test_obs_gen_plane,
        }

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") #Attempt to use GPU if available

        #Init the quadcopter mesh for collison detection only needed to be done once so it is done here

        #Dont use the actual quad for training as it is too detailed and will slow down the collision detection
        self.tri_quad_mesh = None
        if self.use_drone_mesh:
            self.tri_quad_mesh = trimesh.load("gym_quad/meshes/drone_TRI.obj") 
            #Move mesh to origin to rotate it correctly
            self.tri_quad_mesh.apply_translation(np.array([0, 0, 0]))
            #Rotate -90 degrees about trimesh y axis
            self.tri_quad_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [0, 1, 0]))
        else:
            self.tri_quad_mesh = advanced_create_cylinder(radius=self.drone_radius_for_collision, height=self.drone_height_for_collision, sections=8, rot90=True, inverted=False) 
        


        #The 2D gaussian which is multiplied with the depth map to create the collision avoidance reward
        #Only needs to be inited once so it is done here
        peak = self.TwoDgauss_peak
        std =self.TwoDgauss_sigma  #Small -> sharp peak, Large -> wide peak
        TwoD_gaussian = np.zeros((self.depth_map_size[0],self.depth_map_size[1]))
        for i in range(self.depth_map_size[0]):
            for j in range(self.depth_map_size[1]):
                TwoD_gaussian[i,j] = peak*np.exp(-((i-self.depth_map_size[0]/2)**2 + (j-self.depth_map_size[1]/2)**2)/(2*std**2))
        self.torch_TwoD_gaussian = torch.tensor(TwoD_gaussian, device=self.device)

        #Reset environment to init state
        self.reset()


    def reset(self,**kwargs):
        """
        Resets environment to initial state.
        """
        seed = kwargs.get('seed', None)
        super().reset(seed=seed)
        # print("PRINTING SEED WHEN RESETTING:", seed) 
        
        #ONLY UNCOMMENT THIS RANDOM SEEDING IF YOU WANT RANDOMNESS WHEN DOING RUN3D.PY
        #IF YOU WANT TO REPRODUCE THE SAME RESULTS EVERY TIME ADD A NUMBER INTO THE NP.RANDOM.SEED() FUNCTION
        # np.random.seed() 

        #Logging variables:
        #Metrics
        self.cum_e = 0
        self.cum_h = 0
        self.cum_path_progression = 0
        self.cum_total_path_deviance = 0
        self.cum_collision = 0
        self.cum_time = 0
        #Rewards
        self.cum_collision_rew = 0
        self.cum_collision_avoidance_rew = 0
        self.cum_path_adherence_rew = 0
        self.cum_path_progression_rew = 0
        self.cum_reach_end_rew = 0
        self.cum_existence_rew = 0
        self.cum_approach_end_rew = 0
        self.cum_CA_rew = 0
        self.cum_lambda_CA = 0
        self.cum_lambda_PA = 0
        self.cum_pass_wp_rew = 0
        #State
        self.cum_speed = 0


        #General variables being reset
        self.quadcopter = None
        self.path = None
        self.waypoint_index = 0
        self.prog = 0
        self.passed_waypoints = np.zeros((1, 3), dtype=np.float32)

        # Toggle for sparse waypoint passing reward
        self.add_wp_reward = False 

        self.e = None
        self.h = None
        self.chi_error = None
        self.upsilon_error = None

        self.prev_action = [0,0,0]
        self.prev_quad_pos = None

        self.success = False
        self.done = False

        self.LA_at_end = False
        self.cumulative_reward = 0

        self.total_t_steps = 0

        #IMU variables
        self.imu = None
        self.imu = IMU()
        self.imu_measurement = np.zeros((6,), dtype=np.float32) 
        #IMU noise #LET THERE ALWAYS BE SOME NOISE ON THE IMU AS IT IS USED IN THE CONTROLLER (AND ALSO IN THE OBSERVATION)
        self.imu.set_std(0.001, 0.01) #Angular acceleration noise, linear acceleration noise standard deviation a normal dist draws from

        
        #Noise variables
        #If the simulation is not to be perturbed the noise values are set to 0 (except for IMU noise which is always present)
        #Camera pos orient no noise
        camera_look_direction = np.array([1, 0, 0]) #TODO turn these two into hypervariables
        camera_position_body = np.array([0, 0, 0])
        
        self.camera_look_direction_noisy = camera_look_direction
        self.camera_pos_noise = camera_position_body

        #Controller gain noise
        self.kv_noise = 0
        self.kangvel_noise = 0
        self.kR_noise = 0
        
        #Perturbing of camera pose
        if self.perturb_sim or self.perturb_camera_pose:
            self.camera_look_direction_noisy = camera_look_direction + np.random.normal(0, 0.01, 3)
            self.camera_pos_noise = camera_position_body + np.random.normal(0, 0.005, 3)
        
        # IMU boosted noise
        if self.perturb_sim or self.perturb_IMU:
            self.imu.set_std(deg2rad(0.015), 0.015) #Angular acceleration noise, linear acceleration noise standard deviation a normal dist draws from
           # TODO decide on these values and wether it should be set here in reset or in the observation(each step)
        
        #Controller gains
        if self.perturb_sim or self.perturb_ctrl_gains:
            self.kv_noise = np.random.uniform(-0.2, 0.2) #Velocity gain
            self.kR_noise = np.random.uniform(-0.1, 0.1) #Attitude gain
            self.kangvel_noise = np.random.uniform(-0.1, 0.1) #Angular velocity gain

        # TODORandom forces and torques maybe?

        
        #Reward variables
        self.scaled_CA_reward_pre_clip = 0

        #Obstacle variables
        self.obstacles = []
        self.closest_measurement = None
        self.collided = False

        # Depth camera variables
        self.depth_map = torch.zeros((self.depth_map_size[0], self.depth_map_size[1]), dtype=torch.float32, device=self.device)
        self.noisy_depth_map = torch.zeros((self.depth_map_size[0], self.depth_map_size[1]), dtype=torch.float32, device=self.device)
        self.sensor_readings = np.zeros((1, self.compressed_depth_map_size, self.compressed_depth_map_size), dtype=np.float32)

        ## 1. Path and obstacle generation based on scenario
        scenario = self.scenario_switch.get(self.scenario, lambda: print("Invalid scenario"))
        init_state = scenario() #Called such that the obstacles are generated and the init state of the quadcopter is set
        #The function called above sets self.path and self.obstacles(if obstacles are to be generated) and returns the init state of the quadcopter

        #2. Find the obstacles that are close/on the path for the more sophisticated lambda interpolation before the room is generated
        self.obs_near_path_CPPs = [] #Closest points on path for obstacles near path
        if self.obstacles!=[]:
            for obstacle in self.obstacles:
                if obstacle.isDummy:
                    continue
                elif isinstance(obstacle, ImportedMeshObstacle): #Do not add special meshes like the house to the obs_near_path_CPPs
                    continue
                else:
                    obs_pos_np = obstacle.position.cpu().numpy() #Costly so do it here per reset and save important info
                    for wp_index in range(len(self.path.waypoints)-1):
                        obs_cpp = self.path.get_closest_position(obs_pos_np, wp_index)
                        distance = np.linalg.norm(obs_pos_np - obs_cpp) - obstacle.radius #This requires that all obstacles has some form of radius (I made sure the cubes have a "radius")
                        if distance < self.drone_radius_for_collision:
                            self.obs_near_path_CPPs.append(obs_cpp)
        self.obs_near_path_CPPs = [np.round(obs_cpp, 0) for obs_cpp in self.obs_near_path_CPPs]
        self.obs_near_path_CPPs = np.array(self.obs_near_path_CPPs)
        self.obs_near_path_CPPs = np.unique(self.obs_near_path_CPPs, axis=0)

        ## 3. Generate room
        if self.enclose_scene:
            #For some reason the collison manager detects immeadiate collision if only the room is present
            #Hacky workaround is checking if there are no obstacles and adding a far away dummy obstacle
            bounds = None
            if self.obstacles == []:
                dummy_obstacle = SphereMeshObstacle(device=self.device, path = self.mesh_path, radius=1, center_position=torch.tensor([100,100,100]), isDummy=True)
                self.obstacles.append(dummy_obstacle) #This will be used when joining the collision scene later
                bounds, _ = get_scene_bounds([], self.path, padding=self.padding)
            else:
                bounds, _ = get_scene_bounds(self.obstacles, self.path, padding=self.padding)
            #calculate the size of the room
            width = bounds[1] - bounds[0] #z in tri and pt3d, x in enu
            height = bounds[3] - bounds[2] #y in tri and pt3d, z in enu
            depth = bounds[5] - bounds[4] #x in tri and pt3d y in enu
            room_center = torch.tensor([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
            cube = CubeMeshObstacle(device=self.device,width=width, height=height, depth=depth, center_position=room_center,inverted=True) 
            self.obstacles.append(cube)

        ## 4. Generate camera, scene and renderer
        camera = None
        raster_settings = None
        scene = None
        if self.obstacles!=[]:
            #s = min(self.depth_map_size)
            #f_ndc = -0.5*self.depth_map_size[1]/np.tan(self.FOV_horizontal/2) * 2.0 / s
            #px_ndc = - (self.depth_map_size[1] / 2 - self.depth_map_size[1] / 2.0) * 2.0 / s
            #py_ndc = - (self.depth_map_size[0] / 2 - self.depth_map_size[0] / 2.0) * 2.0 / s
            
            #print(f_ndc, px_ndc, py_ndc)


            #focal_length = (-0.5*self.depth_map_size[1]/np.tan(self.FOV_horizontal/2), )
            #principal_point = ((self.depth_map_size[1] / 2, self.depth_map_size[0] / 2), ) # Assuming perfect cam TODO: get K from real cam for exact values(?)
            #img_size = (self.depth_map_size,)
            #camera = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point, image_size=img_size, device=self.device, in_ndc=False)

            #camera = PerspectiveCameras(focal_length=(f_ndc, ), principal_point=((px_ndc, py_ndc), ), device=self.device, in_ndc=True)
            #print(camera.get_projection_transform().get_matrix().cpu().numpy())
            # Change the sign of the last row of K
            #K = camera.get_projection_transform().get_matrix().cpu().numpy()
            #K[0, 3, :] = -K[0, 3, :]
            #camera = PerspectiveCameras(K=K, device=self.device, in_ndc=True)
            #camera = FoVPerspectiveCameras(device=self.device, K=K)

            camera = FoVPerspectiveCameras(device=self.device, fov=self.FOV_horizontal, znear=0.01, zfar=self.max_depth)
            #print(camera.get_projection_transform().get_matrix().cpu().numpy())            

            raster_settings = RasterizationSettings(
                    image_size=self.depth_map_size, 
                    blur_radius=0.0, 
                    faces_per_pixel=1, # Keep at 1, dont change
                    perspective_correct=True, # Doesn't do anything(??), but seems to improve speed
                    cull_backfaces=True #TODO Temp fix for the camera going inside obstacles before collision is detected Find a better less runtimeconsuming solution
                )
            if self.obstacles[0].isDummy: #Means theres just the room and the path
                scene = Scene(device = self.device, obstacles=[self.obstacles[1]])
            else:
                scene = Scene(device = self.device, obstacles=self.obstacles)
            
            self.renderer = DepthMapRenderer(device=self.device, 
                                            raster_settings=raster_settings, 
                                            camera=camera, 
                                            scene=scene, 
                                            MAX_MEASURABLE_DEPTH=self.max_depth, 
                                            img_size=self.depth_map_size)
        else: 
            self.depth_map.fill_(self.max_depth)    # IF there are no obstacles then we know that the depthmap always will display max_depth
                                                    # Aditionally we dont need the rasterizer and renderer if there are no obstacles


        ## 5. Init the trimesh meshes for collision detection 
        #IMPORTANT TO DO THIS AFTER THE CAMERA INIT AS CAMERA NEEDS INVERTED CUBES COLLISION DETECTION NEEDS NORMAL CUBES
        obs_meshes = None
        tri_obs_meshes = None
        tri_joined_obs_mesh = None
        if self.obstacles!=[]:
            obs_meshes = [obstacle.mesh for obstacle in self.obstacles] #Extracting the mesh of the obstacles
            tri_obs_meshes = [trimesh.Trimesh(vertices=mesh.verts_packed().cpu().numpy(), faces=mesh.faces_packed().cpu().numpy()) for mesh in obs_meshes] #Converting pt3d meshes to trimesh meshes
            tri_joined_obs_mesh = trimesh.util.concatenate(tri_obs_meshes) #Create one mesh for obstacles
            tri_joined_obs_mesh.fix_normals() #Fixes the normals of the mesh
            self.collision_manager = trimesh.collision.CollisionManager() #Creating the collision manager
            self.collision_manager.add_object("obstacles", tri_joined_obs_mesh) #Adding the obstacles to the collision manager (Stationary objects)
            #Do not add quadcopter to collision manager as it is moving and will be checked in the step function
            if self.obstacles[0].isDummy: #Means theres just the room and the path
                self.obstacles.pop(0) #Remove the dummy obstacle from the list of obstacles

        # Generate Quadcopter
        self.quadcopter = Quad(self.step_size, init_state)
        self.prev_quad_pos = self.quadcopter.position
        #Move the mesh to the position of the quadcopter
        tri_quad_init_pos = enu_to_tri(self.quadcopter.position)
        #First move the tri_quad_mesh to the origin and then apply Rotation, then Translation (Iguess one could do both at once if concated R and t to T)
        self.tri_quad_mesh.apply_translation(-self.tri_quad_mesh.centroid)
        #Rotate the quadcopter mesh to the orientation of the quadcopter
        self.tri_quad_mesh.apply_transform(tri_Rotmat(*self.quadcopter.attitude))
        self.initial_tri_quad_mesh = self.tri_quad_mesh.copy() #Save the Mesh when it is at the origin and has been rotated to the initial orientation, will be used to update the mesh in the step function
        #Move the quadcopter mesh to the position of the quadcopter
        self.tri_quad_mesh.apply_translation(tri_quad_init_pos)
        

        ###
        self.info = {}
        self.observation = self.observe() 
        return (self.observation,self.info)


    def observe(self):
        """
        Returns the observations of the environment.
        """
        
        #IMU observation
        self.imu_measurement = self.imu.measure(self.quadcopter)
        #Both linear acceleration and angvel is not in [-1,1] clipping it using the max speed of the quadcopter
        self.imu_measurement[0:3] = m1to1(self.imu_measurement[0:3], -self.s_max*2, self.s_max*2)
        self.imu_measurement[3:6] = m1to1(self.imu_measurement[3:6], -self.r_max*2, self.r_max*2)
        self.imu_measurement = self.imu_measurement.astype(np.float32)

        #Depth camera observation
        temp_depth_map = None
        if self.obstacles!=[]:
            pos = self.quadcopter.position
            pos_noisy = pos + self.camera_pos_noise
            orientation = self.quadcopter.attitude
            Rcam,Tcam = self.renderer.camera_R_T_from_quad_pos_orient(pos_noisy, orientation, self.camera_look_direction_noisy)
            self.renderer.update_R(Rcam)
            self.renderer.update_T(Tcam)
            self.depth_map = self.renderer.render_depth_map()
            temp_depth_map = self.depth_map
        else:
            temp_depth_map = self.depth_map #Handles the case where there are no obstacles, Is equal to the init of self.depth_map which is all pixels at max_depth

        #These 5 lines are for the collision avoidance reward but done here to save time as we now can bunch GPU to CPU moves together
        #Use the "pure" depth map for the collision avoidance reward
        self.closest_measurement = torch.min(temp_depth_map)
        non_singular_depth_map = temp_depth_map + self.CA_epsilon #Adding 0.0001 to avoid singular matrix
        div_by_one_depth_map = 1 / non_singular_depth_map
        reward_collision_avoidance_pre_clip = -torch.sum((self.torch_TwoD_gaussian * div_by_one_depth_map)) 


        # Add Gaussian noise to depth map (naive noise model)
        if self.perturb_sim or self.perturb_depth_map:
            sigma = 0.1 # [m] Standard deviation of the Gaussian noise added to the depth map
            noise = torch.normal(mean=0, std=sigma, size=temp_depth_map.size(), device=self.device)
            temp_depth_map += noise
            self.noisy_depth_map = temp_depth_map #For saving the noisy depth map for debugging

        normalized_depth_map = temp_depth_map / self.max_depth
        
        normalized_depth_map_PIL = transforms.ToPILImage()(normalized_depth_map)

        resize_transform = transforms.Compose([
            transforms.Resize((self.compressed_depth_map_size, self.compressed_depth_map_size)),
            transforms.ToTensor(),  # Convert back to tensor
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
        ])        

        resized_depth_map = resize_transform(normalized_depth_map_PIL)
        
        #All moves from GPU to CPU are done here to save time
        self.scaled_CA_reward_pre_clip = reward_collision_avoidance_pre_clip.item()*self.CA_scale #This item operation is costly as moves from GPU to CPU 
        
        self.closest_measurement = self.closest_measurement.item()
        # print("The closeset measurement from the depthmap is: ", np.round(self.closest_measurement,3))  
        self.sensor_readings = resized_depth_map.detach().cpu().numpy()
        #.item() and .detach.cpu.numpy Moves the data from GPU to CPU so we do it here close to the tensor to numpy conversion as
        

        #Domain observation
        self.update_errors() #Updates the errors chi_error and upsilon_error (also crosstrack and vertical track error which is used to calc IAE during run3d.py)

        chi_error_noise = 0
        upsilon_error_noise = 0
        quad_pos_noise = np.zeros(3)
        quad_att_noise = np.zeros(3)
        if self.perturb_sim or self.perturb_domain:
            deg_as_rad = deg2rad(3)
            chi_error_noise = np.random.uniform(-deg_as_rad, deg_as_rad)
            upsilon_error_noise = np.random.uniform(-deg_as_rad, deg_as_rad)
            quad_pos_noise = np.random.normal(0, 0.025, 3) #+-2.5cm noise as std dev #Assume IMU accl integrated into pos but filtered somehow
            quad_att_noise = np.random.normal(0, deg2rad(1), 3) #+-1 degree noise as std dev
        
        quad_pos = self.quadcopter.position + quad_pos_noise #Do this once here and use them in the domain_observation below
        quad_att = self.quadcopter.attitude + quad_att_noise

        # domain_obs = np.zeros(19, dtype=np.float32)
        domain_obs = np.zeros(22, dtype=np.float32)

        # Heading angle error wrt. the path
        domain_obs[0] = np.sin(self.chi_error+chi_error_noise)
        domain_obs[1] = np.cos(self.chi_error+chi_error_noise)
        # Elevation angle error wrt. the path
        domain_obs[2] = np.sin(self.upsilon_error+upsilon_error_noise)
        domain_obs[3] = np.cos(self.upsilon_error+upsilon_error_noise)
         
        # x y z of closest point on path in body frame
        relevant_distance = self.relevant_dist_to_path #For this value and lower the observation will be changing i.e. giving info if above or below its clipped to -1 or 1 
        closest_point = self.path(self.prog)
        closest_point_body = np.transpose(geom.Rzyx(*quad_att)).dot(closest_point - quad_pos)
        domain_obs[4] = m1to1(closest_point_body[0], -relevant_distance,relevant_distance) 
        domain_obs[5] = m1to1(closest_point_body[1], -relevant_distance, relevant_distance) 
        domain_obs[6] = m1to1(closest_point_body[2], -relevant_distance,relevant_distance) 
    
        # Two angles to describe direction of the vector between the drone and the closeset point on path
        x_b_cpp = closest_point_body[0]
        y_b_cpp = closest_point_body[1]
        z_b_cpp = closest_point_body[2]
        ele_closest_p_point_vec = np.arctan2(z_b_cpp, np.sqrt(x_b_cpp**2 + y_b_cpp**2))
        azi_closest_p_point_vec = np.arctan2(y_b_cpp, x_b_cpp)
        
        domain_obs[7] = np.sin(ele_closest_p_point_vec)
        domain_obs[8] = np.cos(ele_closest_p_point_vec)
        domain_obs[9] = np.sin(azi_closest_p_point_vec)
        domain_obs[10] = np.cos(azi_closest_p_point_vec)

        #euclidean norm of the distance from drone to next waypoint
        relevant_distance = (self.path.length / self.n_waypoints-1)*2 #Should be n-1 waypoints to get m segments
        distance_to_next_wp = 0
        if self.waypoint_index+1 < len(self.path.waypoints):
            distance_to_next_wp = np.linalg.norm(self.path.waypoints[self.waypoint_index+1] - quad_pos)
        else:
            distance_to_next_wp = 0 # np.linalg.norm(self.path.waypoints[-1] - quad_pos) With passwp reward we dont want the agent to look for next wp reward when only the end wp is left

        domain_obs[11] = m1to1(distance_to_next_wp, -relevant_distance, relevant_distance)
        # print("dist_nxt_wp", np.round(distance_to_next_wp),"  normed", np.round(domain_obs[18],2))

        #euclidean norm of the distance from drone to the final waypoint
        distance_to_end = np.linalg.norm(self.path.get_endpoint() - quad_pos)
        domain_obs[12] = m1to1(distance_to_end, -self.path.length*2, self.path.length*2)
        # print("DISTANCE TO END:", distance_to_end)

        #body coordinates of the look ahead point
        lookahead_world = self.path.get_lookahead_point(quad_pos, self.la_dist, self.waypoint_index)
        
        #If lookahead point is the end point lock it to the end point
        if not self.LA_at_end and np.abs(lookahead_world[0] - self.path.get_endpoint()[0]) < 1 and np.abs(lookahead_world[1] - self.path.get_endpoint()[1]) < 1 and np.abs(lookahead_world[2] - self.path.get_endpoint()[2]) < 1:
            self.LA_at_end = True
        if self.LA_at_end:
            lookahead_world = self.path.get_endpoint()    

        lookahead_body = np.transpose(geom.Rzyx(*quad_att)).dot(lookahead_world - quad_pos)
        relevant_distance = self.la_dist*2 if self.la_dist > 1 else self.la_dist + self.la_dist
        # select_house_path - 1 if select_house_path is not None else np.random.randint(len(paths))

        domain_obs[13] = m1to1(lookahead_body[0], -relevant_distance,relevant_distance)
        domain_obs[14] = m1to1(lookahead_body[1], -relevant_distance, relevant_distance)
        domain_obs[15] = m1to1(lookahead_body[2], -relevant_distance,relevant_distance)
        # print("LOOKAHEAD BODY:", np.round(lookahead_body))

        #Give the previous action as an observation
        domain_obs[16] = self.prev_action[0]    
        domain_obs[17] = self.prev_action[1]
        domain_obs[18] = self.prev_action[2]
        
        #The velocity of quadcopter in body frame #TODO add noise to this if letting the agent observe body velocity helps
        vel_body = np.transpose(geom.Rzyx(*quad_att)).dot(self.quadcopter.velocity) + quad_pos_noise/2 #Temp noise fix
        domain_obs[19] = m1to1(vel_body[0], -self.s_max, self.s_max) 
        domain_obs[20] = m1to1(vel_body[1], -self.s_max, self.s_max) 
        domain_obs[21] = m1to1(vel_body[2], -self.s_max, self.s_max) 


        #List of the observations before min-max scaling
        pure_obs = [ 
            *self.imu_measurement,
            self.chi_error,
            self.upsilon_error,
            *closest_point_body,
            ele_closest_p_point_vec,
            azi_closest_p_point_vec,
            distance_to_next_wp,
            distance_to_end,
            *lookahead_body,
            *self.prev_action,
            # *vel_body
        ]

        self.info['pure_obs'] = pure_obs

        #The min max normalized domain observation
        self.info['domain_obs'] = domain_obs

        return {'perception':self.sensor_readings,      #Noise from camera 
                'IMU':self.imu_measurement,             #Noise from IMU
                'domain':domain_obs}                    #Noise perturbations


    def step(self, action):
        """
        Simulates the drl environment one time-step. And the physics environment multiple time-steps. 
        Such that the drl env is in sync with the occurence of new depth maps.
        """
        sensor_latency = 0
        if self.perturb_sim or self.perturb_latency:
            decide_if_latency_hits = np.random.uniform(0, 1)
            if decide_if_latency_hits < 0.1: #10% chance of latency
                sensor_latency = np.random.uniform(-1, 1)   #When running 15fps camera and 100Hz physics sim the steps are 6.67ms long so with the sensor latency the steps will range from 4.67ms to 8.67ms
                                                            #When running 10fps camera and 100Hz physics sim the steps are 10ms long so with the sensor latency the steps will range from 8ms to 12ms

        #The quadcopter steps until a new depth map is available
        sim_hz = 1/self.step_size #100 Hz usually
        cam_hz = self.camera_FPS  #15 Hz usually
        steps_before_new_depth_map = (sim_hz//cam_hz) + sensor_latency
        for i in range(int(steps_before_new_depth_map)):           
            F = self.geom_ctrlv2(action)
            #TODO maybe need some translation between input u and thrust F i.e translate u to propeller speed omega? 
            #We currently skip this step for simplicity
            self.quadcopter.step(F)


        #Check for collisions Done in drl sim to save time not in physics sim
        if self.obstacles != []: #Only check collision if there are obstacles
            quad_pos = self.quadcopter.position
            quad_att = self.quadcopter.attitude
            #Reset the mesh to the ORIGIN and initial orientation
            self.tri_quad_mesh = self.initial_tri_quad_mesh.copy() 
            # Generate the rotation matrix using Euler angles
            R = tri_Rotmat(quad_att[0], quad_att[1], quad_att[2])
            # Apply the rotation matrix
            self.tri_quad_mesh.apply_transform(R)
            # Translate mesh to the desired position
            self.tri_quad_mesh.apply_translation(enu_to_tri(quad_pos))
            #Check for collision
            collision_manager_detect = self.collision_manager.in_collision_single(self.tri_quad_mesh)
            if collision_manager_detect: 
                self.collided = True
                #TODO maybe find what obstacle the quadcopter collided with to log
        # self.prev_quad_pos = self.quadcopter.position #For translation of the tri_quad_mesh in the next step


        # Check if a waypoint is passed
        self.prog = self.path.get_closest_u(self.quadcopter.position, self.waypoint_index)
        k = self.path.get_u_index(self.prog)
        if k > self.waypoint_index:
            print("Passed waypoint {:d}".format(k+1), np.round(self.path.waypoints[k],3), "\tquad position:", np.round(self.quadcopter.position,3))
            print("At timestep: ",self.total_t_steps, "  Which equates to: ", self.total_t_steps*self.step_size, "s")
            self.passed_waypoints = np.vstack((self.passed_waypoints, self.path.waypoints[k]))
            self.waypoint_index = k
            self.add_wp_reward = True


        end_cond_1 = np.linalg.norm(self.path.get_endpoint() - self.quadcopter.position) < self.accept_rad
        end_cond_2 = self.total_t_steps >= self.max_t_steps
        end_cond_3 = self.cumulative_reward < self.min_reward
        timeout = False
        reach_min_rew = False
        if end_cond_1 or end_cond_2 or self.collided or end_cond_3:
            if end_cond_1:
                print("Quadcopter reached target!")
                print("Endpoint position", self.path.waypoints[-1], "\tquad position:", self.quadcopter.position)
                print("At timestep: ",self.total_t_steps, "  Which equates to: ", self.total_t_steps*self.step_size, "s")
                self.success = True
            elif self.collided:
                print("Quadcopter collided!")
                print ("At position: ", self.quadcopter.position)
                print("At timestep: ",self.total_t_steps, "  Which equates to: ", self.total_t_steps*self.step_size, "s")
                self.success = False
            elif end_cond_2:
                print("Exceeded time limit")
                print("At timestep: ",self.total_t_steps, "  Which equates to: ", self.total_t_steps*self.step_size, "s")
                self.success = False
                timeout = True
            elif end_cond_3:
                print("Acumulated reward less than", self.min_reward)
                print("At timestep: ",self.total_t_steps, "  Which equates to: ", self.total_t_steps*self.step_size, "s")
                self.success = False
                reach_min_rew = True
            print("TERMINAL STATE REACHED IN SCENARIO: ", self.scenario)
            self.done = True

        # Save sim time info
        self.total_t_steps += 1
        
        #Save interesting info
        self.info['env_steps'] = self.total_t_steps
        self.info['time'] = self.total_t_steps*self.step_size
        self.info['progression'] = self.prog/self.path.length
        self.info['state'] = np.copy(self.quadcopter.state)
        self.info['errors'] = np.array([self.e, self.h])
        self.info['total_path_deviance'] = np.sqrt(self.e**2 + self.h**2)
        self.info['cmd_thrust'] = self.quadcopter.input
        self.info['action'] = action
        self.info['collision_rate'] = int(self.collided)
        self.info['timeout'] = int(timeout)
        self.info['min_rew_reached'] = int(reach_min_rew)
        self.info['success'] = int(self.success)
        
        #For average over episode tensorboardlogging divides by number of steps in the episode in logger.py
        self.cum_e += self.e
        self.cum_h += self.h
        self.cum_path_progression += self.prog
        self.cum_total_path_deviance += np.sqrt(self.e**2 + self.h**2)
        self.cum_collision += int(self.collided)
        self.cum_time += self.total_t_steps*self.step_size
        
        self.info['cum_e_error'] = self.cum_e
        self.info['cum_h_error'] = self.cum_h
        self.info['cum_path_progression'] = self.cum_path_progression
        self.info['cum_total_path_deviance'] = self.cum_total_path_deviance
        self.info['cum_collision'] = self.cum_collision
        self.info['cum_time'] = self.cum_time
        
        #Quadcopter states for logging
        self.cum_speed += np.linalg.norm(self.quadcopter.velocity)
        self.info['cum_speed'] = self.cum_speed

        # Calculate reward
        step_reward = self.reward()
        self.cumulative_reward += step_reward
        self.info['cumulative_reward'] = self.cumulative_reward


        # Make next observation
        #Such that the oberservation has access to the previous action
        self.prev_action = action
        self.observation = self.observe()

        truncated = False
        return self.observation, step_reward, self.done, truncated, self.info


    def reward(self):
        """
        Calculates the reward function for one time step. 
        """
        tot_reward = 0
        lambda_PA = self.lambda_PA_max
        lambda_CA = self.lambda_CA_max

        #Path adherence reward
        dist_from_path = np.linalg.norm(self.path(self.prog) - self.quadcopter.position)
        reward_path_adherence = -(2*(np.clip(dist_from_path, 0, self.PA_band_edge) / self.PA_band_edge) - 1)*self.PA_scale 
        # print("reward_path_adherence", np.round(reward_path_adherence,2),\
        #       "  dist_from_path", np.round(dist_from_path,2))

        ####Collision avoidance reward#### (continuous)
        reward_collision_avoidance = 0
        if self.obstacles != []: #If there are no obstacles, no need to calculate the reward
            danger_range = self.max_depth
            drone_closest_obs_dist = self.closest_measurement 

            #This does not take into account the FOV of the camera -> #naive fast check is simply checking if the obstacle position is along the positive x body axis of the quadcopter
            #Can find the volume of the FOV of the camera and check if the obstacle is in that volume #TODO volume approach needs more work/testing
            # y1 = np.tan(self.FOV_horizontal/2)*self.max_depth #These could be calculated once and saved as a variable
            # y2 = -y1
            # z1 = np.tan(self.FOV_horizontal/2)*self.max_depth
            # z2 = -z1
            #This should work to check if point inside volume: if 0 < obs_cpp_in_body[0] <= self.max_depth and y2 < obs_cpp_in_body[1] < y1 and z2 < obs_cpp_in_body[2] < z1:
            for obs_cpp in self.obs_near_path_CPPs:
                obs_cpp_in_body = np.transpose(geom.Rzyx(*self.quadcopter.attitude)).dot(obs_cpp - self.quadcopter.position)
                if np.linalg.norm(obs_cpp - self.quadcopter.position) < self.max_depth and obs_cpp_in_body[0] > 0:
                    lambda_PA = (drone_closest_obs_dist/danger_range)/2
                    if lambda_PA < self.lambda_PA_min : lambda_PA = self.lambda_PA_min
                    lambda_CA = 1-lambda_PA

            if (drone_closest_obs_dist < danger_range):
                reward_collision_avoidance = np.clip(self.scaled_CA_reward_pre_clip, self.min_CA_rew, self.max_CA_rew) #This value is calculated in observe to save time moving from GPU to CPU

            # print("Collision avoidance reward clipped:", np.round(reward_collision_avoidance,2),                 
            #         "    Collision avoidance reward unclipped:", np.round(scaled_reward_pre_clip,2),\
            #         "    old_reward_collision_avoidance:", np.round(old_reward_collision_avoidance,2),\
            #         "    Distance to closest obstacle:", drone_closest_obs_dist)
            ####Collision avoidance reward done####


        #Path progression reward 
        reward_path_progression = 0
        reward_path_progression1 = np.cos(self.chi_error)
        reward_path_progression2 = np.cos(self.upsilon_error)
        reward_path_progression = (reward_path_progression1 + reward_path_progression2)*self.PP_rew_scale/2
        reward_path_progression = np.clip(reward_path_progression, self.PP_rew_min, self.PP_rew_max) 

        #Approach end reward 
        #Easy to exploit so rather tune CA and PA when near goal
        # approach_end_reward = 0
        # if self.waypoint_index == len(self.path.waypoints)-2: 
        #     dist_to_end = np.linalg.norm(self.quadcopter.position - self.path.get_endpoint())
        #     approach_end_reward = np.exp(-((dist_to_end**2)/(2*self.approach_end_sigma**2)))*self.max_approach_end_rew
        
        #Rather do this: #TODO or maybe not.. decide...
        # if self.waypoint_index == len(self.path.waypoints)-2:
        #     dist_to_end = np.linalg.norm(self.quadcopter.position - self.path.get_endpoint())
        #     if dist_to_end < self.approach_end_range:
        #         lambda_CA = (dist_to_end/self.approach_end_range)/2
        #         if lambda_CA < self.lambda_CA_min : lambda_CA = self.lambda_CA_min
        #         lambda_PA = 1-lambda_CA
                # print("Lambda_CA:", lambda_CA, "  Lambda_PA:", lambda_PA)

        #Collision reward (sparse)
        reward_collision = 0
        if self.collided:
            reward_collision = self.rew_collision
            # print("Collision Reward:", reward_collision)

        #Passed waypoint reward (sparse)
        reward_pass_wp = 0
        '''Note: Does not yield reward for passing the final wp. Which is desierable :)'''
        if self.add_wp_reward:
            reward_pass_wp = self.rew_pass_wp
            # print("Passed waypoint reward:", reward_pass_wp) 
            self.add_wp_reward = False

        #Reach end reward (sparse)
        reach_end_reward = 0
        if self.success:
            reach_end_reward = self.rew_reach_end

        #Existential reward (penalty for being alive to encourage the quadcopter to reach the end of the path quickly) (continous)
        ex_reward = self.existence_reward 

        tot_reward = reward_path_adherence*lambda_PA + reward_collision_avoidance*lambda_CA + reward_collision + reward_path_progression + reach_end_reward + ex_reward + reward_pass_wp

        self.info['reward'] = tot_reward
        self.info['collision_avoidance_reward'] = reward_collision_avoidance*lambda_CA
        self.info['path_adherence'] = reward_path_adherence*lambda_PA
        self.info["path_progression"] = reward_path_progression
        self.info['collision_reward'] = reward_collision
        self.info['reach_end_reward'] = reach_end_reward
        self.info['existence_reward'] = ex_reward
        self.info['lambda_CA'] = lambda_CA
        self.info['lambda_PA'] = lambda_PA
        self.info['pass_wp_reward'] = reward_pass_wp
        # self.info['approach_end_reward'] = approach_end_reward

        #Cumulative reward tensorboard logging # Could do moving averages if we want
        # self.cum_approach_end_rew += approach_end_reward
        self.cum_existence_rew += ex_reward
        self.cum_reach_end_rew += reach_end_reward
        self.cum_collision_rew += reward_collision
        self.cum_path_progression_rew += reward_path_progression
        self.cum_path_adherence_rew += reward_path_adherence*lambda_PA
        self.cum_CA_rew += reward_collision_avoidance*lambda_CA
        self.cum_lambda_CA += lambda_CA
        self.cum_lambda_PA += lambda_PA
        self.cum_pass_wp_rew += reward_pass_wp

        self.info['cum_collision_rew'] = self.cum_collision_rew
        self.info['cum_CA_rew'] = self.cum_CA_rew        
        self.info['cum_path_adherence_rew'] = self.cum_path_adherence_rew
        self.info['cum_path_progression_rew'] = self.cum_path_progression_rew
        self.info['cum_reach_end_rew'] = self.cum_reach_end_rew
        self.info['cum_existence_rew'] = self.cum_existence_rew
        self.info['cum_approach_end_rew'] = self.cum_approach_end_rew
        self.info['cum_lambda_CA'] = self.cum_lambda_CA
        self.info['cum_lambda_PA'] = self.cum_lambda_PA
        self.info['cum_pass_wp_rew'] = self.cum_pass_wp_rew

        return tot_reward


    def geom_ctrlv2(self, action):
        #Translate the action to the desired velocity and yaw rate
        cmd_v_x = self.s_max * ((action[0]+1)/2)*np.cos(action[1]*self.i_max)
        cmd_v_y = 0
        cmd_v_z = self.s_max * ((action[0]+1)/2)*np.sin(action[1]*self.i_max)
        cmd_r = self.r_max * action[2]
        self.cmd = np.array([cmd_v_x, cmd_v_y, cmd_v_z, cmd_r]) #For plotting

        #Gains, z-axis-basis=e3 and rotation matrix 
        kv = self.kv + self.kv_noise
        kR = self.kR + self.kR_noise
        kangvel = self.kangvel + self.kangvel_noise

        imu_meas = self.imu.measure(self.quadcopter)
        imu_quad_angvel = np.array([imu_meas[3], imu_meas[4], imu_meas[5]])
        #Pseudo integration of linear velocity and angular rate
        #assuming that a filtered version of the integrated rates will result in approx these values
        imu_quad_vel = self.quadcopter.velocity + self.imu.lin_noise*0.3 #0.3 is a guess
        imu_quad_att = self.quadcopter.attitude + self.imu.ang_noise*0.3 #0.3 is a guess

        R = j_Rzyx(*imu_quad_att) # Rotation matrix from body to world frame

        #Body frame velocity control #THIS GIVES THE BEHAVIOUR WE WANT
        ev = np.array([cmd_v_x, cmd_v_y, cmd_v_z]) - imu_quad_vel #self.quadcopter.velocity Old "pure measurements"
        drag_force_body = np.array([ss.d_u*imu_quad_vel[0], ss.d_v*imu_quad_vel[1], ss.d_w*imu_quad_vel[2]])
        gravity_in_body = np.array([-ss.m*ss.g*np.sin(imu_quad_att[1]), 
                                    ss.m*ss.g*np.cos(imu_quad_att[1])*np.sin(imu_quad_att[0]), 
                                    ss.m*ss.g*np.cos(imu_quad_att[1])*np.cos(imu_quad_att[0])])
        #Thrust command
        f = kv*ev + drag_force_body + gravity_in_body
        f_world = R @ f
        thrust_command = np.dot(f_world,R[2]) #Projection of f in world on z-axis of body frame to avoid large f if quadcopter very is tilted

        #Rd 
        f_x = f[0]
        f_y = -f[1]
        f_z = f[2]
        pitch_setpoint = np.arctan2(f_x, f_z)
        roll_setpoint = np.arctan2(f_y, np.sqrt(f_z**2 + f_x**2))
        yaw_setpoint = imu_quad_att[2] #self.quadcopter.attitude[2]       
        Rd = j_Rzyx(roll_setpoint, pitch_setpoint, yaw_setpoint) 

        eR = 1/2*(Rd.T @ R - R.T @ Rd)
        eatt = geom.vee_map(eR)
        eatt = np.reshape(eatt, (3,))

        des_angvel = np.array([0.0, 0.0, cmd_r])

        s_pitch = np.sin(imu_quad_att[1])
        c_pitch = np.cos(imu_quad_att[1])
        s_roll = np.sin(imu_quad_att[0])                  
        c_roll = np.cos(imu_quad_att[0]) 
        R_euler_to_body = np.array([[1, 0, -s_pitch],
                                    [0, c_roll, s_roll*c_pitch],
                                    [0, -s_roll, c_roll*c_pitch]]) #Essentially the inverse of Tzyx from geomutils

        des_angvel_body = R_euler_to_body @ des_angvel

        eangvel = imu_quad_angvel - R.T @ (Rd @ des_angvel_body) 

        torque = -kR*eatt - kangvel*eangvel + np.cross(imu_quad_angvel,ss.Ig@imu_quad_angvel)
        
        u = np.zeros(4)
        u[0] = thrust_command
        u[1:] = torque

        F = np.linalg.inv(ss.B()[2:]).dot(u)
        F = np.clip(F, ss.thrust_min, ss.thrust_max)
        return F

      #### UPDATE FUNCTION####
    
    def update_errors(self):
        '''Updates the cross track and vertical track errors, as well as the course and elevation errors.'''
        self.e = 0.0 #Cross track error
        self.h = 0.0 #Vertical track error
        self.chi_error = 0.0 #Course angle error xy-plane
        self.upsilon_error = 0.0 #Elevation angle error between z and xy-plane

        s = self.prog

        chi_p, upsilon_p = self.path.get_direction_angles(s) #Path direction angles also denoted by pi in some literature
        # Calculate tracking errors Serret Frenet frame
        SF_rotation = geom.Rzyx(0, upsilon_p, chi_p)

        epsilon = np.transpose(SF_rotation).dot(self.quadcopter.position - self.path(s))
        self.e = epsilon[1] #Cross track error
        self.h = epsilon[2] #Vertical track error

        # Calculate course and elevation errors from tracking errors
        chi_r = np.arctan2(self.e, self.la_dist) 
        upsilon_r = np.arctan2(self.h, np.sqrt(self.e**2 + self.la_dist**2))
        
        #Desired course and elevation angles
        chi_d = chi_p - chi_r 
        upsilon_d = upsilon_p - upsilon_r 

        self.chi_error = geom.ssa(chi_d - self.quadcopter.chi) #Course angle error xy-plane 
        self.upsilon_error = geom.ssa(upsilon_d - self.quadcopter.upsilon) #Elevation angle error between z and xy-plane

        # print("upsilon_d", np.round(upsilon_d*180/np.pi), "upsilon_quad", np.round(self.quadcopter.upsilon*180/np.pi), "upsilon_error", np.round(self.upsilon_error*180/np.pi),\
        #       "\n\nchi_d", np.round(chi_d*180/np.pi), "chi_quad", np.round(self.quadcopter.chi*180/np.pi), "chi_error", np.round(self.chi_error*180/np.pi))

    #### PLOTTING ####
    def axis_equal3d(self, ax):
        """
        Shifts axis in 3D plots to be equal. Especially useful when plotting obstacles, so they appear spherical.

        Parameters:
        ----------
        ax : matplotlib.axes
            The axes to be shifted.
        """
        extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:,1] - extents[:,0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        # plt.show()
        return ax

    def plot3D(self, wps_on=True, leave_out_first_wp=True):
        """
        Returns 3D plot of path and obstacles.
        """
        ax = self.path.plot_path(wps_on, leave_out_first_wp=leave_out_first_wp)
        for obstacle in self.obstacles:
            if isinstance(obstacle, SphereMeshObstacle):
                ax.plot_surface(*obstacle.return_plot_variables(), color='r', zorder=1)
                ax.set_aspect('equal', adjustable='datalim')
        return ax#self.axis_equal3d(ax)

    def plot_section3d(self):
        """
        Returns 3D plot of path, obstacles and quadcopter.
        """
        plt.rc('lines', linewidth=3)
        ax = self.plot3D(wps_on=False)
        ax.set_xlabel(xlabel="North [m]", fontsize=14)
        ax.set_ylabel(ylabel="East [m]", fontsize=14)
        ax.set_zlabel(zlabel="Down [m]", fontsize=14)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.zaxis.set_tick_params(labelsize=12)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([-50, 0, 50])
        ax.set_zticks([-50, 0, 50])
        ax.view_init(elev=-165, azim=-35)
        ax.scatter3D(*self.quadcopter.position, label="Initial Position", color="y")

        self.axis_equal3d(ax)
        ax.legend(fontsize=14)
        plt.show()


    #### SCENARIOS #### 
    #Utility function for scenarios
    def recap_previous_scenario(self,n_prev_scenarios): #TODO low pri make this update based on the curriculum_config
        '''
        Inputs:
        n_prev_scenarios: number of previous scenarios
        Returns:
        initial_state: initial state of the previous scenario

        This function must match the scenario dictionary from the train3d.py file such that the scenarios are correctly recapitulated.
        Also change the n_prev_scenarios in the scenario functions to appear in the correct order.
        '''
        
        chance = np.random.uniform(0,1)
        if chance < 1/n_prev_scenarios:
            initial_state = self.scenario_line()
        elif chance < 2/n_prev_scenarios:
            initial_state = self.scenario_easy()
        elif chance < 3/n_prev_scenarios:
            initial_state = self.scenario_random_pos_att_easy()
        elif chance < 4/n_prev_scenarios:
            initial_state = self.scenario_intermediate()
        elif chance < 5/n_prev_scenarios:
            initial_state = self.scenario_proficient()
        elif chance < 6/n_prev_scenarios:
            initial_state = self.scenario_advanced()
        elif chance < 7/n_prev_scenarios:
            initial_state = self.scenario_expert()    
        else:
            initial_state = self.scenario_proficient_perturbed_sim()
        
        print("RECAPPING A PREVIOUS SCENARIO")

        return initial_state
    
    def generate_obstacles(self, n, rmin, rmax , path:QPMI, mean, std, onPath=False, quad_pos = None, safety_margin = 2, obstacle_type = None):
        '''
        Inputs:
        n: number of obstacles
        rmin: minimum radius of obstacles
        rmax: maximum radius of obstacles
        path: path object
        mean: mean distance from path
        std: standard deviation of distance from path
        onPath: if True, obstacles will be placed on the path
        quad_pos: position of the quadcopter
        safety_margin: minimum distance between obstacles and quadcopter upon initialization
        obstacle_type: if None, obstacles will be either spheres or cubes with equal probability. If 'sphere', all obstacles will be spheres. If 'cube', all obstacles will be cubes.
        Returns:
        None. Adds obstacles directly to the environment.
        '''
        num_obstacles = 0
        path_lenght = path.length
        while num_obstacles < n:
            #uniform distribution of length along path
            u_obs = np.random.uniform(0.20*path_lenght,0.90*path_lenght)

            # Get the tangent, normal, and binormal vectors at u_obs
            t_hat, n_hat, b_hat = path.calculate_vectors(u_obs)

            #Draw a normal distributed random number for the distance from the path
            dist = np.random.normal(mean, std)
            #get x,y,z coordinates of the obstacle if it were placed on the path
            x,y,z = path.__call__(u_obs)
            obs_on_path_pos = np.array([x,y,z])
            
            obs_pos = np.zeros(3)

            #Offset the obstacle a distance d from the path at a random angle in the yz path plane (x-axis in path plane points along path)
            # Generate random angle theta in the yz-plane (local to the path)
            theta = np.random.uniform(0, 2 * np.pi)
            # Compute position offset in local yz-plane
            local_offset = dist * np.array([0, np.sin(theta), np.cos(theta)])
            # Transform local offset to world coordinates
            world_offset = n_hat * local_offset[1] + b_hat * local_offset[2]
            obs_pos = obs_on_path_pos + world_offset

            obstacle_radius = np.random.uniform(rmin,rmax) #uniform distribution of size
            #50/50 of it being a sphere or a cube unless overridden by the obstacle_type (can add more meshes if wanted)
            obstacle_type_choice = np.random.uniform(0,1)

            if not onPath and (np.linalg.norm(obs_pos - obs_on_path_pos) < obstacle_radius + safety_margin):
                continue #We check if the obstacle is too close to the path when its not allowed to be on the path if so we skip this obstacle
            elif (np.linalg.norm(obs_pos - quad_pos) < obstacle_radius + safety_margin + self.drone_radius_for_collision):  
                continue #We check if the obstacle is too close to the quadcopter if so we skip this obstacle
            elif np.linalg.norm(obs_pos - path.get_endpoint()) < obstacle_radius + safety_margin:
                continue #We check if the obstacle is too close to the endpoint if so we skip this obstacle
            else:
                obstacle_coords = torch.tensor(obs_pos,device=self.device).float().squeeze()
                pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
                if obstacle_type == 'sphere': #First check if anything specific is requested if not leave it to chance
                    self.obstacles.append(SphereMeshObstacle(radius = obstacle_radius,center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))
                elif obstacle_type == 'cube':
                    self.obstacles.append(CubeMeshObstacle(device=self.device, width=obstacle_radius*1.15, height=obstacle_radius*1.15, depth=obstacle_radius*1.15, center_position=pt3d_obs_coords, inverted=False))
                elif obstacle_type_choice > 0.5:
                    self.obstacles.append(SphereMeshObstacle(radius = obstacle_radius,center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path)) 
                elif obstacle_type_choice < 0.5:
                    self.obstacles.append(CubeMeshObstacle(device=self.device, width=obstacle_radius*1.15, height=obstacle_radius*1.15, depth=obstacle_radius*1.15, center_position=pt3d_obs_coords, inverted=False))
                num_obstacles += 1

    #No obstacles
    def scenario_line(self):
        initial_state = np.zeros(6)
        
        desired_length = np.random.uniform(self.line_path_range[0],self.line_path_range[1])
        n_wps = int(desired_length//2 + 1)
        segment_length = desired_length/(n_wps-1)
        # print("desired lenght",desired_length ,"nwp",n_wps, "seglen",segment_length)

        waypoints = generate_random_waypoints(n_wps,'line',segmentlength=segment_length)
        self.path = QPMI(waypoints)
        init_pos = [0, 0, 0]
        init_attitude=np.array([0,0,self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state
    
    def scenario_line_y(self):
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(self.n_waypoints,'line_y',segmentlength=self.segment_length)
        self.path = QPMI(waypoints)
        init_pos = [0, 0, 0]
        init_attitude=np.array([0,0,self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state
    
    def scenario_line_up(self):
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(self.n_waypoints,'line_up',segmentlength=self.segment_length)
        self.path = QPMI(waypoints)
        init_pos = [0, 0, 0]
        init_attitude=np.array([0,0,self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state

    def scenario_xy_line(self):
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(self.n_waypoints,'xy_line', segmentlength=self.segment_length)
        self.path = QPMI(waypoints)
        init_pos = [0, 0, 0]
        init_attitude=np.array([0,0,self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state

    def scenario_squiggly_line_xy_plane(self):
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(self.n_waypoints,'squiggly_line_xy_plane' , segmentlength=self.segment_length)
        self.path = QPMI(waypoints)
        init_pos = [0, 0, 0]
        init_attitude=np.array([0,0,self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state

    def scenario_3d_new(self,random_pos=False,random_attitude=False):
        initial_state = np.zeros(6)

        desired_length = np.random.uniform(self.new_3d_path_range[0],self.new_3d_path_range[1])
        n_wps = int(desired_length//2 + 1)
        segment_length = desired_length/(n_wps-1)
        # print("desired lenght",desired_length ,"nwp",n_wps, "seglen",segment_length)

        waypoints = generate_random_waypoints(n_wps,'3d_new', segmentlength=segment_length)
        self.path = QPMI(waypoints)
        
        if random_pos:
            init_pos = [np.random.uniform(-self.padding+2,self.padding-2), np.random.uniform(-self.padding+2,self.padding-2), np.random.uniform(-self.padding+2,self.padding-2)]
        else:    
            init_pos=[0, 0, 0]
        
        if random_attitude:
            init_attitude=np.array([np.random.uniform(-np.pi/6,np.pi/6), np.random.uniform(-np.pi/6,np.pi/6), np.random.uniform(-np.pi,np.pi)])
        else:
            init_attitude=np.array([0, 0, self.path.get_direction_angles(0)[0]])
            
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state
    
    def scenario_3d_up_down(self, random_pos=False, random_attitude=False):
        initial_state = np.zeros(6)
        if np.random.uniform(0,1) < 0.5:
            waypoints = generate_random_waypoints(self.n_waypoints,'3d_up', segmentlength=self.segment_length)
        else:
            waypoints = generate_random_waypoints(self.n_waypoints,'3d_down', segmentlength=self.segment_length)
        self.path = QPMI(waypoints)

        if random_pos:
            init_pos = [np.random.uniform(-self.padding+2,self.padding-2), np.random.uniform(-self.padding+2,self.padding-2), np.random.uniform(-self.padding+2,self.padding-2)]
        else:    
            init_pos=[0, 0, 0]
        
        if random_attitude:
            init_attitude=np.array([np.random.uniform(-np.pi/6,np.pi/6), np.random.uniform(-np.pi/6,np.pi/6), np.random.uniform(-np.pi,np.pi)])
        else:
            init_attitude=np.array([0, 0, self.path.get_direction_angles(0)[0]])
            
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state
    
    def scenario_3d_up_down_plane(self, random_pos=False, random_attitude=False, e_angle_range = (np.pi/3,np.pi/2)):

        initial_state = np.zeros(6)
        desired_length = np.random.uniform(self.new_3d_up_down_path_range[0],self.new_3d_up_down_path_range[1])
        n_wps = int(desired_length//2 + 1)
        segment_length = desired_length/(n_wps-1)
        # print("desired lenght",desired_length ,"nwp",n_wps, "seglen",segment_length)
        choice = np.random.uniform(0,1)
        if choice < 0.33:
            waypoints = generate_random_waypoints(n_wps,'3d_new', segmentlength=segment_length)
        elif choice < 0.66:
            waypoints = generate_random_waypoints(n_wps,'3d_up', segmentlength=segment_length, e_angle_range=e_angle_range)
        else:
            waypoints = generate_random_waypoints(n_wps,'3d_down', segmentlength=segment_length, e_angle_range=e_angle_range)
        self.path = QPMI(waypoints)

        if random_pos:
            init_pos = [np.random.uniform(-self.padding+2,self.padding-2), np.random.uniform(-self.padding+2,self.padding-2), np.random.uniform(-self.padding+2,self.padding-2)]
        else:    
            init_pos=[0, 0, 0]
        
        if random_attitude:
            init_attitude=np.array([np.random.uniform(-np.pi/6,np.pi/6), np.random.uniform(-np.pi/6,np.pi/6), np.random.uniform(-np.pi,np.pi)])
        else:
            init_attitude=np.array([0, 0, self.path.get_direction_angles(0)[0]])
            
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state
    

    #With obstacles
    def scenario_easy(self): #Surround the path with 1-4 obstacles But ensure no obstacles on path
        initial_state = self.scenario_3d_new()
        
        n_obstacles = np.random.randint(1,5)
        self.generate_obstacles(n = n_obstacles, rmin=0.2, rmax=2, path = self.path, mean = 0, std = 3, onPath=False, quad_pos=initial_state[0:3])
        
        if np.random.uniform(0,1) < self.recap_chance:
            initial_state = self.recap_previous_scenario(n_prev_scenarios=1)

        return initial_state
    
    def scenario_random_pos_att_easy(self):
        initial_state = self.scenario_3d_new(random_pos=True,random_attitude=True)
        
        n_obstacles = np.random.randint(1,5)
        self.generate_obstacles(n = n_obstacles, rmin=0.2, rmax=2, path = self.path, mean = 0, std = 3, onPath=False, quad_pos=initial_state[0:3])

        n_prev_scenarios = 2
        if np.random.uniform(0,1) < self.recap_chance:
            initial_state = self.recap_previous_scenario(n_prev_scenarios)

        return initial_state

    def scenario_intermediate(self):
        initial_state = self.scenario_3d_new(random_attitude=True,random_pos=True)
        self.generate_obstacles(n = 1, rmin=0.5, rmax=2, path = self.path, mean = 0, std = 0.2, onPath=True, quad_pos=initial_state[0:3])
        
        n_prev_scenarios = 3
        if np.random.uniform(0,1) < self.recap_chance:
            initial_state = self.recap_previous_scenario(n_prev_scenarios)

        return initial_state
    
    def scenario_proficient(self): #NOTE THAT PROFICENT HAS BEEN RUN BEFORE INTERMEDIATE FOR MANY TRAINING SESSIONS TRY TO FLIP IT NOW
        initial_state = self.scenario_3d_new(random_attitude=True,random_pos=True)
        #One obs near/ on path:
        self.generate_obstacles(n = 1, rmin=0.5, rmax=2, path = self.path, mean = 0, std = 0.01, onPath=True, quad_pos=initial_state[0:3])
        # One away from path
        self.generate_obstacles(n = 1, rmin=1, rmax=3, path = self.path, mean = 0, std = 3, onPath=False, quad_pos=initial_state[0:3])

        n_prev_scenarios = 4
        if np.random.uniform(0,1) < self.recap_chance:
            initial_state = self.recap_previous_scenario(n_prev_scenarios)

        return initial_state

    def scenario_advanced(self):
        initial_state = self.scenario_3d_up_down_plane(random_attitude=True,random_pos=True,e_angle_range=(np.pi/6,np.pi/3))        
        self.generate_obstacles(n = 1, rmin=0.5, rmax=2, path = self.path, mean = 0, std = 0.01, onPath=True, quad_pos=initial_state[0:3])
        self.generate_obstacles(n = 3, rmin=1, rmax=3, path = self.path, mean = 0, std = 4, onPath=False, quad_pos=initial_state[0:3])

        n_prev_scenarios = 5
        if np.random.uniform(0,1) < self.recap_chance:
            initial_state = self.recap_previous_scenario(n_prev_scenarios)

        return initial_state

    def scenario_expert(self):        
        initial_state = self.scenario_3d_up_down_plane(random_attitude=True,random_pos=True,e_angle_range=(np.pi/4,deg2rad(75)))
        self.generate_obstacles(n = 1, rmin=0.4, rmax=2, path = self.path, mean = 0, std = 0.2, onPath=True, quad_pos=initial_state[0:3])
        self.generate_obstacles(n = 5, rmin=0.4, rmax=3, path = self.path, mean = 0, std = 4.5, onPath=False, quad_pos=initial_state[0:3])

        n_prev_scenarios = 6
        if np.random.uniform(0,1) < self.recap_chance:
            initial_state = self.recap_previous_scenario(n_prev_scenarios)

        return initial_state
    
    def scenario_easy_perturbed_sim(self):
        initial_state = self.scenario_3d_new(random_attitude=True,random_pos=True)
        self.perturb_sim = True

        n_prev_scenarios = None
        if np.random.uniform(0,1) < self.recap_chance:
            initial_state = self.recap_previous_scenario(n_prev_scenarios)

        return initial_state

    def scenario_proficient_perturbed_sim(self):
        initial_state=self.scenario_proficient()
        self.perturb_sim = True

        n_prev_scenarios = 7
        if np.random.uniform(0,1) < self.recap_chance:
            initial_state = self.recap_previous_scenario(n_prev_scenarios) 

        return initial_state
    
    def scenario_expert_perturbed_sim(self):
        initial_state = self.scenario_expert()
        self.perturb_sim = True

        n_prev_scenarios = 8
        if np.random.uniform(0,1) < self.recap_chance:
            initial_state = self.recap_previous_scenario(n_prev_scenarios)  

        return initial_state
    
#Specials    
    def scenario_random_corridor(self): #This can be used as a test scenario if we set the seed to be consistent
        print("RANDOM CORRIDOR")
        initial_state = self.scenario_3d_new()
        #Many cubes 1m away from path
        #TODO Make this not random while still allowing the other tests that are running in parallell be random
        #TOOD Turn this into "CAVE" scenario
        np.random.seed(0)
        self.generate_obstacles(n = 40, rmin=0.8, rmax=1, path = self.path, mean = 2, std = 0.01, onPath=False, quad_pos=initial_state[0:3], safety_margin=0.2, obstacle_type='cube')
        self.generate_obstacles(n = 40, rmin=0.8, rmax=1, path = self.path, mean = -2, std = 0.01, onPath=False, quad_pos=initial_state[0:3], safety_margin=0.2, obstacle_type='cube')
        return initial_state

    def scenario_rotate_to_escape(self): #TODO
        #Start inside "deadend" and rotate to escape. 
        #Let path go through the middle of the obstacles
        #Randomize where the opening is.
        #Mybe not needed as random attitude in other train scenarios make agent learn rotation about itself.
        self.padding = 5
        initial_state = self.scenario_line()
        #move quad pos to be in the middle of the path offset by 2.5 m in y
        initial_state[0] = 5
        #Surround the point [0,2.5,0] with n sphere obstacles encapsulating the quadcopter
        n_obstacles = 10

        return initial_state


#Testing scenarios #TODO make the drone spawn at a larger random area at the start for more varied paths
    def scenario_test_path(self):
        test_waypoints = np.array([np.array([0,0,0]), np.array([10,1,0]), np.array([20,0,0]), np.array([70,0,0])])
        self.n_waypoints = len(test_waypoints)
        self.path = QPMI(test_waypoints)
        init_pos = [0,0,0]
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])

        obstacle_radius = 10
        obstacle_coords = self.path(20)
        obstacle_coords = torch.tensor(obstacle_coords,device=self.device).float().squeeze()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
        self.obstacles.append(SphereMeshObstacle(radius = obstacle_radius,center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))
        return initial_state

    def scenario_test(self):
        initial_state = self.scenario_test_path()
        obstacle_radius = 10
        obstacle_coords = self.path(self.path.length/2)
        obstacle_coords = torch.tensor(obstacle_coords,device=self.device).float().squeeze()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
        self.obstacles.append(SphereMeshObstacle(radius = obstacle_radius,center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))
        return initial_state

    def scenario_horizontal_test(self):
        print("HORIZONTAL")
        waypoints = [(0,0,0), (2.5,0,0), (5,0,0), (7.5,0,0), (10,0,0)]
        self.path = QPMI(waypoints)
        self.obstacles = []
        for i in range(7):
            y = (-3+1*i)
            obstacle_coords = torch.tensor([5,y,0],device=self.device)
            pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
            self.obstacles.append(SphereMeshObstacle(radius = 0.5, center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))
        self.padding = 3
        init_pos = np.array([0, 0, 0]) + np.random.uniform(low=-2, high=2, size=(1,3))
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos[0], init_attitude])
        return initial_state

    def scenario_vertical_test(self):
        print("VERTICAL")
        waypoints = [(0,0,0), (2.5,0,0), (5,0,0), (7.5,0,0), (10,0,0)]
        self.path = QPMI(waypoints)
        self.obstacles = []
        for i in range(7):
            z = -3+1*i
            obstacle_coords = torch.tensor([5,0,z],device=self.device)
            pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
            self.obstacles.append(SphereMeshObstacle(radius = 0.5, center_position=pt3d_obs_coords, device=self.device,path=self.mesh_path))
        self.padding = 3    
        init_pos = np.array([0, 0, 0]) + np.random.uniform(low=-2, high=2, size=(1,3))
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos[0], init_attitude])
        return initial_state

    def scenario_deadend_test(self): 
        print("DEADEND")
        waypoints = [
            (0.0, 0.0, 0.0),
            (2.5, 0.0, 0.0),
            (5.0, 0.0, 0.0),
            (7.5, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (12.5, 0.0, 0.0),
            (15.0, 0.0, 0.0),
            (17.5, 0.0, 0.0),
            (20.0, 0.0, 0.0),
            (22.5, 0.0, 0.0),
            (25.0, 0.0, 0.0),
            (27.5, 0.0, 0.0),
            (30.0, 0.0, 0.0),
            (32.5, 0.0, 0.0),
            (35.0, 0.0, 0.0),
            (37.5, 0.0, 0.0),
            (40.0, 0.0, 0.0),
            (42.5, 0.0, 0.0),
            (45.0, 0.0, 0.0),
            (47.5, 0.0, 0.0),
            (50.0, 0.0, 0.0)]
        self.path = QPMI(waypoints)
        radius = 10
        angles = np.linspace(-90, 90, 10)*np.pi/180
        obstacle_radius = (angles[1]-angles[0])*radius/2
        for ang1 in angles:
            for ang2 in angles:
                x = 25 + radius*np.cos(ang1)*np.cos(ang2)
                y = radius*np.cos(ang1)*np.sin(ang2)
                z = -radius*np.sin(ang1)
                
                obstacle_coords = torch.tensor([x,y,z],device=self.device).float().squeeze()
                pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
                self.obstacles.append(SphereMeshObstacle(radius = obstacle_radius*1.3,center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))
                # self.obstacles.append(CubeMeshObstacle(device=self.device, width=obstacle_radius*1.41, height=obstacle_radius*1.41, depth=obstacle_radius*1.41, center_position=pt3d_obs_coords, inverted=False))

        self.padding = 3
        init_pos = np.array([0, 0, 0]) + np.random.uniform(low=-2, high=2, size=(1,3))
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos[0], init_attitude])
        return initial_state

    def scenario_helix(self):
        print("HELIX")
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(self.n_waypoints,'helix', segmentlength=self.segment_length)
        self.path = QPMI(waypoints)
        init_pos = np.array([11, 0, -26/2]) + np.random.uniform(low=-0.5, high=0.5, size=(1,3))
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos[0], init_attitude])
        obstacle_coords = torch.tensor([0,0,0],device=self.device).float()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
        self.obstacles.append(SphereMeshObstacle(radius = 10,center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))

        return initial_state
    

    def scenario_house(self):
        print("HOUSE")
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(self.n_waypoints,'house',select_house_path=1) #TODO change select_house_path to what we want, None for random
        self.path = QPMI(waypoints)

        init_pos = waypoints[0]# + np.random.uniform(low=-0.25, high=0.25, size=(1,3))

        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([np.array(init_pos), init_attitude])

        obstacle_coords = torch.tensor([0,0,0],device=self.device).float()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
        self.obstacles.append(ImportedMeshObstacle(device=self.device, path = "./gym_quad/meshes/house_TRI_new.obj", center_position=pt3d_obs_coords))
        return initial_state
        

    def scenario_house_easy(self):
        print("HOUSE EASY")
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(self.n_waypoints,'house',select_house_path=6) 
        waypoints = waypoints[::-1]
        self.path = QPMI(waypoints)
        init_pos = waypoints[0] + np.random.uniform(low=-0.75, high=0.75, size=(1,3))
        init_pos = init_pos[0]

        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([np.array(init_pos), init_attitude])

        obstacle_coords = torch.tensor([0,0,0],device=self.device).float()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
        self.obstacles.append(ImportedMeshObstacle(device=self.device, path = "./gym_quad/meshes/house_TRI_new.obj", center_position=pt3d_obs_coords))
        return initial_state
    
    def scenario_house_easy_obstacles(self):
        print("HOUSE EASY OBSTACLES")
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(self.n_waypoints,'house',select_house_path=6) 
        waypoints = waypoints[::-1]
        self.path = QPMI(waypoints)
        init_pos = waypoints[0] + np.random.uniform(low=-0.75, high=0.75, size=(1,3))
        init_pos = init_pos[0]

        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([np.array(init_pos), init_attitude])

        obstacle_coords = torch.tensor([0,0,0],device=self.device).float()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
        self.obstacles.append(ImportedMeshObstacle(device=self.device, path = "./gym_quad/meshes/house_TRI_new.obj", center_position=pt3d_obs_coords))

        # Add small cube at (1.03, 0.14, 1.12)
        obstacle_coords = torch.tensor([1.03, 0.14, 1.12],device=self.device).float()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
        self.obstacles.append(CubeMeshObstacle(device=self.device, width=0.25, height=2, depth=0.25, center_position=pt3d_obs_coords, inverted=False))

        # Add small sphere at ( 3.075, -2.555,   1.12)
        obstacle_coords = torch.tensor([3.075, -2.555, 1.12],device=self.device).float()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
        self.obstacles.append(SphereMeshObstacle(radius = 0.25,center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))
        return initial_state

    def scenario_house_hard(self):
        print("HOUSE HARD")
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(self.n_waypoints,'house',select_house_path=1) 
        self.path = QPMI(waypoints)

        init_pos = waypoints[0] + np.random.uniform(low=-0.75, high=0.75, size=(1,3))
        init_pos = init_pos[0]

        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([np.array(init_pos), init_attitude])

        obstacle_coords = torch.tensor([0,0,0],device=self.device).float()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
        self.obstacles.append(ImportedMeshObstacle(device=self.device, path = "./gym_quad/meshes/house_TRI_new.obj", center_position=pt3d_obs_coords))
        return initial_state
    
    def scenario_house_hard_obstacles(self):
        print("HOUSE HARD OBSTACLES")
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(self.n_waypoints,'house',select_house_path=1)
        self.path = QPMI(waypoints)

        init_pos = waypoints[0] + np.random.uniform(low=-0.75, high=0.75, size=(1,3))
        init_pos = init_pos[0]

        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([np.array(init_pos), init_attitude])

        obstacle_coords = torch.tensor([0,0,0],device=self.device).float()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
        self.obstacles.append(ImportedMeshObstacle(device=self.device, path = "./gym_quad/meshes/house_TRI_new.obj", center_position=pt3d_obs_coords))

        # Add small cube at (1.973361, -1.19834, -1.20533)
        obstacle_coords = torch.tensor([1.973361, -1.19834, -1.20533],device=self.device).float()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
        self.obstacles.append(CubeMeshObstacle(device=self.device, width=0.25, height=2, depth=0.25, center_position=pt3d_obs_coords, inverted=False))

        # Add small sphere at (2.25, -1.95, 3.49)
        obstacle_coords = torch.tensor([2.25, -2.5, 3.49],device=self.device).float()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
        self.obstacles.append(SphereMeshObstacle(radius = 0.25,center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))

        return initial_state



#Development scenarios #TODO update to scale
    def scenario_dev_test_crash(self):
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(3,'line',segmentlength=self.segment_length)
        self.path = QPMI(waypoints)
        init_pos = [0, 0, 0]
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        #Place one large obstacle at the second waypoint
        obstacle_coords = torch.tensor(self.path.waypoints[1],device=self.device).float().squeeze()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
        self.obstacles.append(SphereMeshObstacle(radius = 2,center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))
        # Decrease the room padding such that a crash is likely
        self.padding = 3
        return initial_state
    
    def scenario_dev_test_cube_crash(self):
        initial_state = np.zeros(6)
        waypoints = generate_random_waypoints(3,'line',segmentlength=self.segment_length)
        self.path = QPMI(waypoints)
        init_pos = [0, 0, 0]
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        #Place one large obstacle at the second waypoint
        obstacle_coords = torch.tensor(self.path.waypoints[1],device=self.device).float().squeeze()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
        self.obstacles.append(CubeMeshObstacle(width=2, height=2, depth=2, center_position=pt3d_obs_coords, device=self.device, inverted=False))
        self.padding = 3
        return initial_state
    
    def scenario_dev_test_obs_at_end(self):
        init_state = self.scenario_line()
        #Place an obstacle at the end of the path
        obstacle_coords = torch.tensor(self.path.waypoints[-1],device=self.device).float().squeeze()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
        self.obstacles.append(SphereMeshObstacle(radius = 1, center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))
        return init_state
    
    def scenario_dev_test_obs_in_outskirt(self):
        init_state = self.scenario_line()
        #Place an obstacle at the outskirt of the fov at the center of the path
        obstacle_coords = torch.tensor([6,6,0],device=self.device).float()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
        self.obstacles.append(SphereMeshObstacle(radius = 1, center_position=pt3d_obs_coords,device=self.device,path=self.mesh_path))
        return init_state
    
    def scenario_dev_test_outskirt_and_end(self):
        init_state = self.scenario_dev_test_obs_in_outskirt()
        #Place an obstacle at the end of the path
        obstacle_coords = torch.tensor(self.path.waypoints[-1],device=self.device).float().squeeze()
        pt3d_obs_coords = enu_to_pytorch3d(obstacle_coords,device=self.device)
        self.obstacles.append(CubeMeshObstacle(width=1, height=1, depth=1, inverted=False, center_position=pt3d_obs_coords,device=self.device))
        return init_state
    
    def scenario_dev_test_obs_gen_plane(self):
        initial_state = self.scenario_squiggly_line_xy_plane()
        #Place many obstacles along the path
        self.generate_obstacles(n = 40, rmin=0.8, rmax=1, path = self.path, mean = 2, std = 0.01, onPath=False, quad_pos=initial_state[0:3], safety_margin=0.2, obstacle_type='cube')
        # self.generate_obstacles(n = 40, rmin=0.8, rmax=1, path = self.path, mean = -2, std = 0.01, onPath=False, quad_pos=initial_state[0:3], safety_margin=0.2, obstacle_type='cube')
        return initial_state

