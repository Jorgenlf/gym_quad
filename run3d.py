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

from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from utils import *
from argparse import Namespace

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Tensflow logging level
# '0': (default) logs all messages.
# '1': logs messages with level INFO and above.
# '2': logs messages with level WARNING and above.
# '3': logs messages with level ERROR and above.

"""
To run a trained agent, run the following command in terminal, exchange x for the experiment id you want to run:

python run3d.py --env "" --exp_id x --run_scenario "" --trained_scenario "" --agent x --episodes x --manual_control False --RT_vis True

python run3d.py --exp_id 3 --run_scenario "line" --trained_scenario "line" --agent 800000 --episodes 1 --RT_vis False --save_depth_maps True

--manual_control and --RT_vis are False by default
--env "" is set to LV_VAE-v0 by default

If RT_vis and manual_control are both set to False,
The simulation will be ran and results and plots will be saved to a test folder in the experiment directory
"""

if __name__ == "__main__":
    
    experiment_dir, agent_path, args = parse_experiment_info()

    #----#----#For running of file without the need of command line arguments#----#----#

    # args = Namespace(manual_control=True) 
    manual_scenario = "proficient" # "line", "horizontal", "3d", "helix", "intermediate", "proficient", "expert"

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

        print("Feature extractor:\n", agent.policy.features_extractor) #TODO would be nice if we could get the compressed depth maps from the VAE

        if args.RT_vis == False:
            for episode in range(args.episodes):
                try:
                    episode_df, env = simulate_environment(episode, env, agent, test_dir, args.save_depth_maps)
                    sim_df = pd.concat([sim_df, episode_df], ignore_index=True)
                except NameError:
                    sim_df = episode_df
                
                #Creates folders for plots
                create_plot_folders(test_dir)

                #Path and quadcopter travel
                plot_3d(env, sim_df[sim_df['Episode']==episode], test_dir)
                plt.show()

                # Observations
                #normalized
                plot_all_normed_domain_observations(sim_df,test_dir)
                #original (pure)
                plot_observation_body_accl(sim_df,test_dir)
                plot_observation_body_angvel(sim_df,test_dir)
                plot_observation_cpp(sim_df,test_dir)
                plot_observation_cpp_azi_ele(sim_df,test_dir)
                plot_observation_e_azi_ele(sim_df,test_dir)
                plot_observation_dists(sim_df,test_dir)
                plot_observation_LA(sim_df,test_dir)
                
                # States
                plot_angular_velocity(sim_df,test_dir)
                plot_attitude(sim_df,test_dir)
                plot_velocity(sim_df,test_dir)
                
                plt.close('all')

                write_report(test_dir, sim_df, env, episode)



        elif args.RT_vis == True: 
            for episode in range(args.episodes):
                obs,info = env.reset()
                visualizer = EnvironmentVisualizer(env.obstacles, env.quadcopter.position, env.quadcopter.attitude)
                visualizer.draw_path(env.path.waypoints)
                while True:
                    action = agent.predict(env.observation, deterministic=True)[0] #[a,dtype,None] so action[0] is the action
                    _, _, done, _, _ = env.step(action)
                    
                    quad_pos = env.quadcopter.position
                    quad_att = env.quadcopter.attitude
                    visualizer.update_quad_visual(quad_pos, quad_att)

                    world_LApoint = env.path.get_lookahead_point(quad_pos, 5, env.waypoint_index) #Maybe faster to use env.wolrd_LApoint and env.closest_path_point
                    closest_path_point = env.path.get_closest_position(quad_pos,env.waypoint_index)
                    velocity_world = env.quadcopter.position_dot
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
            visualizer = EnvironmentVisualizer(env.obstacles, env.quadcopter.position, env.quadcopter.attitude)
            visualizer.draw_path(env.path.waypoints)
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
            
            while True:
                obs, rew, done, _, info = env.step(action=input)
                quad_pos = env.quadcopter.position
                quad_att = env.quadcopter.attitude
                visualizer.update_quad_visual(quad_pos, quad_att)

                world_LApoint = env.path.get_lookahead_point(quad_pos, 5, env.waypoint_index)
                closest_path_point = env.path.get_closest_position(quad_pos,env.waypoint_index)
                velocity_world = env.quadcopter.position_dot
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

        env = gym.make("LV_VAE-v0", scenario=manual_scenario, seed=0)
        _manual_control(env)
        exit()

''' To move around in the 3D plot from vispy:
LMB: orbits the view around its center point.
RMB or scroll: change scale_factor (i.e. zoom level)
SHIFT + LMB: translate the center point
SHIFT + RMB: change FOV
'''

'''Possible scenarios:
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
'''        