import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_quad
import os
import glob
import re

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Rectangle
from PIL import Image
from RTvisualizer import *
from pv_plotting_3d import Plotter3D, Plotter3DMultiTraj

from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from train_run_utils import *
from argparse import Namespace

from gym_quad import register_lv_vae_envs
from drl_config import lv_vae_config


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Tensflow logging level
# '0': (default) logs all messages.
# '1': logs messages with level INFO and above.
# '2': logs messages with level WARNING and above.
# '3': logs messages with level ERROR and above.

"""
To run a trained agent, run the following command in terminal, exchange x for the experiment id you want to run:

python run3d.py --env "" --exp_id x --run_scenario "" --trained_scenario "" --agent x --episodes x --manual_control False --RT_vis True

python run3d.py --exp_id 3 --run_scenario "line" --trained_scenario "line" --agent 800000 --episodes 1 --save_depth_maps True

--manual_control and --RT_vis are False by default
--env "" is set to LV_VAE-v0 by default

If RT_vis and manual_control are both set to False,
The simulation will be ran and results and plots will be saved to a test folder in the experiment directory
"""

if __name__ == "__main__":

    run_config = lv_vae_config.copy()
    run_config["recap_chance"] = 0.0 # No recapitulation when running
    run_config["max_t_steps"] = 10 #6000 # Maximum number of timesteps in the DRL simulation before it is terminated

    register_lv_vae_envs(run_config)
    
    experiment_dir, agent_path, args = parse_experiment_info()

    experiment_dir = os.path.join(experiment_dir, args.run_scenario)
    experiment_dir = os.path.join(experiment_dir, "tests")
    os.makedirs(experiment_dir, exist_ok=True)

    #----#----#For running of file without the need of command line arguments#----#----#

    # args = Namespace(manual_control=True, env = "LV_VAE_MESH-v0", save_depth_maps=False) 
    manual_scenario = "proficient" # "line", "horizontal", "3d", "helix", "intermediate", "proficient", "expert", "crash", "easy"
    
    #Temp variables for debugging
    quad_pos_log = []
    quad_mesh_pos_log = []
    #----#----#NB uncomment when running actual agents#----#----#

    if args.manual_control == False:
        tests = glob.glob(os.path.join(experiment_dir, "test*"))
        if tests == []:
            test = "test1"
        else:
            last_test = max([int(*re.findall(r'\d+', os.path.basename(os.path.normpath(file)))) for file in tests])
            test = f"test{last_test + 1}"
        test_dir = os.path.join(experiment_dir, test)
        os.mkdir(test_dir)
        os.mkdir(os.path.join(test_dir, "plots"))

        summary = pd.DataFrame(columns=["Episode", "Timesteps", "Avg Absolute Path Error", "IAE Cross", "IAE Vertical", "Progression", "Success", "Collision"])
        summary.to_csv(os.path.join(test_dir, 'test_summary.csv'), index=False)

        env = gym.make(args.env, scenario=args.run_scenario)
        agent = PPO.load(agent_path)

        # print("Feature extractor:\n", agent.policy.features_extractor) # Uncomment to see the feature extractor being used

        if args.RT_vis == False:
            cum_rewards = {}
            all_drone_trajs = {} # Maps episode number to a trajectory
            all_init_pos = {} # Maps episode number to initial position
            for episode in range(args.episodes):
                try:
                    episode_df, env = simulate_environment(episode, env, agent, test_dir, args.save_depth_maps)
                    sim_df = pd.concat([sim_df, episode_df], ignore_index=True) #TODO make it work with several episodes
                except NameError:
                    sim_df = episode_df

                #Creates folders for plots
                create_plot_folders(test_dir)
                
                path = env.unwrapped.path

                #sim_df[sim_df['Episode']==episode]
                all_drone_trajs[episode] = np.stack((episode_df[r"$X$"], episode_df[r"$Y$"], episode_df[r"$Z$"]), axis=-1)
                all_init_pos[episode] = all_drone_trajs[episode][0]
                cum_rewards[episode] = episode_df['Reward'].sum()
                
                
                # Per episide stuff
                drone_traj = np.stack((episode_df[r"$X$"], episode_df[r"$Y$"], episode_df[r"$Z$"]), axis=-1)
                init_pos = drone_traj[0]
                obstacles = env.unwrapped.obstacles

                #Make the interactive 3D plot            
                plotter = Plotter3D(obstacles=obstacles, 
                                    path=path, 
                                    drone_traj=drone_traj,
                                    initial_position=init_pos,
                                    nosave=True) 
                plotter.plot_scene_and_trajs(save_path=os.path.join(test_dir, "plots", f"episode{episode}.png"),
                                            azimuth=90, # 90 or 0 is best angle for the 3D plot 
                                            elevation=None,
                                            see_from_plane=None)
                
                del plotter
                
                # Save the 3D plot: #Ghetto fix calling the same class twice but works.
                # plotter = Plotter3D(obstacles=obstacles, 
                #                     path=path, 
                #                     drone_traj=drone_traj,
                #                     initial_position=init_pos,
                #                     nosave=False) 
                #plotter.plot_scene_and_trajs(save_path=os.path.join(test_dir, "plots", f"episode{episode}.png"),
                #                            azimuth=90, # 90 or 0 is best angle for the 3D plot 
                #                            elevation=None,
                #                            see_from_plane=None)
                
                # For plotting the scenarios
                savename = "expert"
                azis = [0, 90, 180, 270]
                for azi in azis:
                    plotter = Plotter3D(obstacles=obstacles, 
                                    path=path, 
                                    drone_traj=drone_traj,
                                    initial_position=init_pos,
                                    nosave=False) 
                    plotter.plot_only_scene(save_path=os.path.join(test_dir, "plots", f"{savename}_azi{azi}_{episode}.png"),
                                            azimuth=azi, 
                                            elevation=None,
                                            see_from_plane=None)
                    del plotter
            
            if args.episodes > 1:
                pass
                multiplotter = Plotter3DMultiTraj(obstacles=obstacles,
                                                path=path,
                                                drone_trajectories=all_drone_trajs,
                                                cum_rewards=cum_rewards,
                                                nosave=False)
                multiplotter.plot_scene_and_trajs(save_path=os.path.join(test_dir, "plots", f"multiplot.png"),
                                                azimuth=90,
                                                elevation=None,
                                                see_from_plane=None)
                
                # This loop plots the from all four azimuths and all three planes # TODO This is buggy for some view angles. Axis labels dissapear (i think this is bc. they are essentially meshes and are view from behind or something)
                # azis = [0, 90, 180, 270]
                # planes = ["xy", "xz", "yz"]
                # for azi in azis:
                #     plotter = Plotter3D(obstacles=obstacles, 
                #                     path=path, 
                #                     drone_traj=drone_traj,
                #                     initial_position=init_pos,
                #                     nosave=False)
                #     plotter.plot_scene_and_trajs(save_path=os.path.join(test_dir, "plots", f"episode{int(sim_df['Episode'].iloc[0])}_azi{azi}.png"),
                #                                 azimuth=azi, 
                #                                 elevation=None,
                #                                 see_from_plane=None)
                #     del plotter
                # for plane in planes:
                #     plotter = Plotter3D(obstacles=obstacles, 
                #                     path=path, 
                #                     drone_traj=drone_traj,
                #                     initial_position=init_pos,
                #                     nosave=False)
                #     plotter.plot_scene_and_trajs(save_path=os.path.join(test_dir, "plots", f"episode{int(sim_df['Episode'].iloc[0])}_{plane}.png"),
                #                                 see_from_plane=plane)
                #     del plotter
                
                


                #Path and quadcopter travel
                #plot_3d(env, sim_df[sim_df['Episode']==episode], test_dir)
                #plt.show()

                # Observations
                #normalized
                # plot_all_normed_domain_observations(sim_df,test_dir)
                # #original (pure)
                # plot_observation_body_accl(sim_df,test_dir)
                # plot_observation_body_angvel(sim_df,test_dir)
                # plot_observation_cpp(sim_df,test_dir)
                # plot_observation_cpp_azi_ele(sim_df,test_dir)
                # plot_observation_e_azi_ele(sim_df,test_dir)
                # plot_observation_dists(sim_df,test_dir)
                # plot_observation_LA(sim_df,test_dir)
                
                # # States
                # plot_angular_velocity(sim_df,test_dir)
                # plot_attitude(sim_df,test_dir)
                # plot_velocity(sim_df,test_dir)
                
                # plt.close('all')

            write_report(test_dir, sim_df, env, episode)



        elif args.RT_vis == True: 
            for episode in range(args.episodes):
                obs,info = env.reset()
                visualizer = EnvironmentVisualizer(env.unwrapped.obstacles, env.unwrapped.quadcopter.position, env.unwrapped.quadcopter.attitude)
                visualizer.draw_path(env.unwrapped.path.waypoints)
                while True:
                    action = agent.predict(env.unwrapped.observation, deterministic=True)[0] #[a,dtype,None] so action[0] is the action
                    _, _, done, _, _ = env.step(action)
                    
                    quad_pos = env.unwrapped.quadcopter.position
                    quad_att = env.unwrapped.quadcopter.attitude
                    visualizer.update_quad_visual(quad_pos, quad_att)

                    world_LApoint = env.unwrapped.path.get_lookahead_point(quad_pos, 5, env.unwrapped.waypoint_index) #Maybe faster to use env.wolrd_LApoint and env.closest_path_point
                    closest_path_point = env.unwrapped.path.get_closest_position(quad_pos,env.unwrapped.waypoint_index)
                    velocity_world = env.unwrapped.quadcopter.position_dot
                    b1 = geom.Rzyx(*quad_att) @ np.array([1,0,0])
                    b2 = geom.Rzyx(*quad_att) @ np.array([0,1,0])
                    b3 = geom.Rzyx(*quad_att) @ np.array([0,0,1])

                    #Body axis vectors
                    visualizer.update_vector(quad_pos,quad_pos+b1, [1, 0, 0],"b1")
                    visualizer.update_vector(quad_pos,quad_pos+b2, [0, 1, 0],"b2")
                    visualizer.update_vector(quad_pos,quad_pos+b3, [0, 0, 1],"b3")
                    
                    #orange body velocity vector
                    visualizer.update_vector(quad_pos,quad_pos+velocity_world, [1, 0.5, 0],"world_velocity")

                    #purple LA point and vector
                    visualizer.update_vector(quad_pos,world_LApoint, [160/255, 32/255, 240/255],"LA_vec") 
                    visualizer.update_point(world_LApoint, [160/255, 32/255, 240/255],"LA_point")

                    #green closest point on path
                    visualizer.update_point(closest_path_point, [0, 1, 0],"Closest_p_point")        
                    app.process_events()
                    if done:
                        break
                env.close()    


    elif args.manual_control == True:
        def _manual_control(env):
            """ Manual control function.
                Reads keyboard inputs and maps them to valid inputs to the environment.
            
            Infinite environment loop:
            - Map keyboard inputs to valid actions
            - Reset environment once done is True
            - Exits upon closing the window or pressing ESCAPE.
            """
            obs,info = env.reset()
            input = [0, 0, 0] #speed -1 = 0, inclination of velocity vector wrt x-axis and yaw rate 
            update_text = False
            visualizer = EnvironmentVisualizer(env.unwrapped.obstacles, env.unwrapped.quadcopter.position, env.unwrapped.quadcopter.attitude)
            visualizer.draw_path(env.unwrapped.path.waypoints)
            @visualizer.scene.events.key_press.connect
            def on_key(event):
                nonlocal input
                nonlocal update_text
                if event.key == 'd': # Rotate right
                    input = [-1, 0, -1]
                elif event.key == 'a': # Rotate left
                    input = [-1, 0,1]
                elif event.key == 'w': # Forward
                    input = [1, 0.3, 0]
                elif event.key == 'Space': # Up 
                    input = [1, 1, 0]
                elif event.key == 's': # down (might not be possible as geometric ctrl holds quad hovering)
                    input = [1, -1, 0]
                elif event.key == 'escape':
                    env.close()
                elif event.key == 'u':
                    update_text = True

            visualizer.add_text("Distance to end") 
            visualizer.add_text("x y z of closest point on path")
            visualizer.add_text("Heading angle error deg")  
            visualizer.add_text("elevation angle error deg")
            visualizer.add_text("phi angle vec drone closest point on path deg")
            visualizer.add_text("psi angle vec drone closest point on path deg")
            
            done = False
            while True:
                obs, rew, done, _, info = env.step(action=input)
                quad_pos = env.unwrapped.quadcopter.position
                quad_att = env.unwrapped.quadcopter.attitude

                #Saving of depthmaps:
                if args.save_depth_maps:
                    save_depth_maps(env,"debug_manual_depthmaps")

                #Update quadcopter visual every third step
                if env.unwrapped.total_t_steps % 3 == 0:
                    visualizer.update_quad_visual(quad_pos, quad_att)
                    world_LApoint = env.unwrapped.path.get_lookahead_point(quad_pos, 5, env.unwrapped.waypoint_index)
                    closest_path_point = env.unwrapped.path.get_closest_position(quad_pos,env.unwrapped.waypoint_index)
                    velocity_world = env.unwrapped.quadcopter.position_dot
                    # vel_world_from_transform = geom.Rzyx(*quad_att) @ env.quadcopter.velocity  #equivalent to the pos_dot :)
                    b1 = geom.Rzyx(*quad_att) @ np.array([1,0,0])
                    b2 = geom.Rzyx(*quad_att) @ np.array([0,1,0])
                    b3 = geom.Rzyx(*quad_att) @ np.array([0,0,1])

                    #Body axis vectors
                    visualizer.update_vector(quad_pos,quad_pos+b1, [1, 0, 0],"b1")
                    visualizer.update_vector(quad_pos,quad_pos+b2, [0, 1, 0],"b2")
                    visualizer.update_vector(quad_pos,quad_pos+b3, [0, 0, 1],"b3")

                    #orange world velocity vector
                    visualizer.update_vector(quad_pos,quad_pos+velocity_world, [1, 0.5, 0],"world_velocity")
                    # visualizer.update_vector(quad_pos,quad_pos+vel_world_from_transform*3, [1, 0, 0],"transd_world_velocity")

                    #purple LA point and vector
                    visualizer.update_vector(quad_pos,world_LApoint, [160/255, 32/255, 240/255],"LA_vec") 
                    visualizer.update_point(world_LApoint, [160/255, 32/255, 240/255],"LA_point")

                    #green closest point on path
                    visualizer.update_point(closest_path_point, [0, 1, 0],"Closest_p_point")

                    #For plotting the mesh and quad pos after
                    # quad_pos_log.append(quad_pos)
                    # quad_mesh_pos_log.append(env.quad_mesh_pos)

                    #blue point for realtime plotting of the quadcopter mesh 
                    # visualizer.update_point(env.unwrapped.quad_mesh_pos, color=[0,0,1], id="Quad_mesh")

                    if update_text: #Relate values to the added text above
                        print("Updating values of text")  
                        rad2deg = 180/np.pi
                        normalized_dist_to_end = info['domain_obs'][19]
                        x_y_z_closepath = info['domain_obs'][8:11] 
                        headingangleerr = np.arcsin(info['domain_obs'][0])*rad2deg
                        elevationangleerr = np.arcsin(info['domain_obs'][1])*rad2deg
                        anglesclosestppath_phi = np.arcsin(info['domain_obs'][11])*rad2deg
                        agnleclosestppath_psi = np.arcsin(info['domain_obs'][13])*rad2deg
                        values_related_to_text = [normalized_dist_to_end,
                                                    x_y_z_closepath,
                                                    headingangleerr,
                                                    elevationangleerr,
                                                    anglesclosestppath_phi,
                                                    agnleclosestppath_psi]
                        visualizer.update_text(values_related_to_text)
                        update_text = False

                    app.process_events()
                if done:
                    done = False
                    env.reset()
                    env.close()
                    visualizer.close()

        env = gym.make(id=args.env, scenario=manual_scenario)
        _manual_control(env)
        # #plot the path of the quadcopter and its mesh use done as while condition
        # quad_pos_log = np.array(quad_pos_log)
        # quad_mesh_pos_log = np.array(quad_mesh_pos_log)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(quad_pos_log[:,0], quad_pos_log[:,1], quad_pos_log[:,2], label='Quadcopter')
        # ax.plot(quad_mesh_pos_log[:,0], quad_mesh_pos_log[:,1], quad_mesh_pos_log[:,2], label='Quadcopter mesh')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.legend()
        # plt.show()
        exit()




''' To move around in the 3D plot from vispy:
LMB: orbits the view around its center point.
RMB or scroll: change scale_factor (i.e. zoom level)
SHIFT + LMB: translate the center point
SHIFT + RMB: change FOV
'''

'''Possible scenarios: (Exist more now check LV_VAE_MESH.py dict in init() fcn)
Training scenarios
    "line": 
    "line_new": 
    "horizontal": 
    "horizontal_new": 
    "3d": 
    "3d_new": 
    "helix": 
    "intermediate": 
    "proficient": 
    #"advanced": 
    "expert": 

# Testing scenarios
    "test_path": 
    "test": 
    "test_current": 
    "horizontal": 
    "vertical": 
    "deadend": 

#Dev testing scenarios
    "crash"
''' 
