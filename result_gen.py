import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_quad
import os
import glob
import re
import pandas as pd

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Rectangle
from PIL import Image
from RTvisualizer import *
from pv_plotting_3d import Plotter3D, Plotter3DMultiTraj

from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from train_run_utils import *
from argparse import Namespace

if __name__ == "__main__":
    # experiment_dir, args = parse_experiment_info()
    
    #TODO as the house and the other scenarios require different LV_VAE_config param we need to figure out
    #To switch config between house and the other scenarios. Can do manually, but auto would be nice.
    test_scenarios = ["horizontal","vertical","deadend","helix","random_corridor"]
    trained_scenarios_to_run = ["expert","proficient_perturbed","expert_perturbed"]    
    
    # test_scenarios = ["horizontal","vertical"]
    # trained_scenarios_to_run = ["expert","proficient_perturbed"]
    
    #---# For debugging #---#
    args = Namespace(env="LV_VAE_MESH-v0", episodes=10, save_depth_maps=False, exp_id=19) 
    base_experiment_dir = os.path.join(r"./log", r"{}".format(args.env), r"Experiment {}".format(args.exp_id))
    #---# For debugging #---#

    for trained_scen in trained_scenarios_to_run:
        agent_path = os.path.join(base_experiment_dir, trained_scen)
        available_agents = glob.glob(os.path.join(agent_path, "agents", "*.zip"))
        
        for agent in available_agents:
            agent_name = os.path.splitext(os.path.basename(agent))[0]
            for test_scen in test_scenarios:
                # Construct the directory paths
                results_gen_dir = os.path.join(agent_path, "results_gen")
                test_scen_dir = os.path.join(results_gen_dir, test_scen)
                os.makedirs(test_scen_dir, exist_ok=True)

                # Generate the test directory name based on agent name
                test_dir = os.path.join(test_scen_dir, f"test_agent_{agent_name}")
                if os.path.exists(test_dir):
                    # If the directory already exists, find a unique name
                    index = 1
                    while os.path.exists(test_dir):
                        test_dir = os.path.join(test_scen_dir, f"test_agent_{agent_name}_{index}")
                        index += 1

                os.makedirs(test_dir, exist_ok=True)
                os.makedirs(os.path.join(test_dir, "plots"), exist_ok=True)

                summary = pd.DataFrame(columns=["Episode", "Timesteps", "Avg Absolute Path Error", "IAE Cross", "IAE Vertical", "Progression", "Success", "Collision"])
                summary.to_csv(os.path.join(test_dir, 'test_summary.csv'), index=False)

                env = gym.make(args.env, scenario=test_scen)
                agent_model = PPO.load(agent)

                cum_rewards = {}
                all_drone_trajs = {}  # Maps episode number to a trajectory
                all_init_pos = {}  # Maps episode number to initial position
                for episode in range(args.episodes):
                    try:
                        episode_df, env = simulate_environment(episode, env, agent_model, test_dir, args.save_depth_maps)
                        sim_df = pd.concat([sim_df, episode_df], ignore_index=True)  # TODO make it work with several episodes
                    except NameError:
                        sim_df = episode_df

                    # Creates folders for plots
                    # create_plot_folders(test_dir)
                    # Dont need to call the fcn if only the trajectory plots are plotted
                    
                    path = env.unwrapped.path

                    all_drone_trajs[episode] = np.stack((episode_df[r"$X$"], episode_df[r"$Y$"], episode_df[r"$Z$"]), axis=-1)
                    all_init_pos[episode] = all_drone_trajs[episode][0]
                    cum_rewards[episode] = episode_df['Reward'].sum()
                    
                    # Per episode stuff
                    drone_traj = np.stack((episode_df[r"$X$"], episode_df[r"$Y$"], episode_df[r"$Z$"]), axis=-1)
                    init_pos = drone_traj[0]
                    obstacles = env.unwrapped.obstacles
                
                    plotter = Plotter3D(obstacles=obstacles, 
                                        path=path, 
                                        drone_traj=drone_traj,
                                        initial_position=init_pos,
                                        nosave=False)  # TODO make it both save and display interactive plot, needs to fix resolution thing
                    plotter.plot_scene_and_trajs(save_path=os.path.join(test_dir, "plots", f"episode{episode}.png"),
                                                 azimuth=90,  # 90 or 0 is best angle for the 3D plot 
                                                 elevation=None,
                                                 see_from_plane=None)
                    del plotter

                    write_report(test_dir, sim_df, env, episode)
                
                if args.episodes > 1:
                    multiplotter = Plotter3DMultiTraj(obstacles=obstacles,
                                                      path=path,
                                                      drone_trajectories=all_drone_trajs,
                                                      cum_rewards=cum_rewards,
                                                      nosave=False)
                    multiplotter.plot_scene_and_trajs(save_path=os.path.join(test_dir, "plots", f"multiplot.png"),
                                                      azimuth=90,
                                                      elevation=None,
                                                      see_from_plane=None)

# import numpy as np
# import matplotlib.pyplot as plt
# import gymnasium as gym
# import gym_quad
# import os
# import glob
# import re

# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.patches import Rectangle
# from PIL import Image
# from RTvisualizer import *
# from pv_plotting_3d import Plotter3D, Plotter3DMultiTraj

# from mpl_toolkits.mplot3d import Axes3D
# from stable_baselines3 import PPO
# from train_run_utils import *
# from argparse import Namespace


# if __name__ == "__main__":
#     # experiment_dir, args = parse_experiment_info()
    
#     # test_scenarios = ["house","horizontal","vertical","deadend","helix","random_corridor"]
#     # trained_scenarios_to_run = ["expert","proficient_perturbed","expert_perturbed"]    
    
#     test_scenarios = ["horizontal","vertical"]
#     trained_scenarios_to_run = ["expert","proficient_perturbed"]
    
#     #---# For debugging #---#
#     args = Namespace(env = "LV_VAE_MESH-v0", episodes = 2,save_depth_maps = False, exp_id = 19) 
#     experiment_dir = os.path.join(r"./log", r"{}".format(args.env), r"Experiment {}".format(args.exp_id))
#     #---# For debugging #---#



#     for trained_scen in trained_scenarios_to_run:
#         agent_path = os.path.join(experiment_dir, trained_scen)
#         available_agents = glob.glob(agent_path + "/agents/*.zip")
#         experiment_dir = os.path.join(experiment_dir, trained_scen)
#         experiment_dir = os.path.join(experiment_dir, "results_gen")
#         experiment_dir = os.path.join(experiment_dir, test_scen)
#         os.makedirs(experiment_dir, exist_ok=True)
#         for agent_path in available_agents:
            
#             for test_scen in test_scenarios:

#                 tests = glob.glob(os.path.join(experiment_dir, "test*"))
#                 if tests == []:
#                     test = "test1"
#                 else:
#                     last_test = max([int(*re.findall(r'\d+', os.path.basename(os.path.normpath(file)))) for file in tests])
#                     test = f"test{last_test + 1}"
#                 test_dir = os.path.join(experiment_dir, test)
#                 os.mkdir(test_dir)
#                 os.mkdir(os.path.join(test_dir, "plots"))
                
#                 #save the agent number and training scenario name to the test folder #TODO make more elegant
#                 with open(os.path.join(test_dir, "agent_number.txt"), "w") as f:
#                     f.write(agent_path) 

#                 summary = pd.DataFrame(columns=["Episode", "Timesteps", "Avg Absolute Path Error", "IAE Cross", "IAE Vertical", "Progression", "Success", "Collision"])
#                 summary.to_csv(os.path.join(test_dir, 'test_summary.csv'), index=False)

#                 env = gym.make(args.env, scenario=test_scen)
#                 agent = PPO.load(agent_path)

#                 # print("Feature extractor:\n", agent.policy.features_extractor) # Uncomment to see the feature extractor being used

#                 cum_rewards = {}
#                 all_drone_trajs = {} # Maps episode number to a trajectory
#                 all_init_pos = {} # Maps episode number to initial position
#                 for episode in range(args.episodes):
#                     try:
#                         episode_df, env = simulate_environment(episode, env, agent, test_dir, args.save_depth_maps)
#                         sim_df = pd.concat([sim_df, episode_df], ignore_index=True) #TODO make it work with several episodes
#                     except NameError:
#                         sim_df = episode_df

#                     #Creates folders for plots
#                     create_plot_folders(test_dir)
                    
#                     path = env.unwrapped.path

#                     #sim_df[sim_df['Episode']==episode]
#                     all_drone_trajs[episode] = np.stack((episode_df[r"$X$"], episode_df[r"$Y$"], episode_df[r"$Z$"]), axis=-1)
#                     all_init_pos[episode] = all_drone_trajs[episode][0]
#                     cum_rewards[episode] = episode_df['Reward'].sum()
                    
                    
#                     # Per episide stuff
#                     drone_traj = np.stack((episode_df[r"$X$"], episode_df[r"$Y$"], episode_df[r"$Z$"]), axis=-1)
#                     init_pos = drone_traj[0]
#                     obstacles = env.unwrapped.obstacles
                
#                     plotter = Plotter3D(obstacles=obstacles, 
#                                         path=path, 
#                                         drone_traj=drone_traj,
#                                         initial_position=init_pos,
#                                         nosave=False) #TODO make it both save and display interactive plot, needs to fix resolution thing
#                     plotter.plot_scene_and_trajs(save_path=os.path.join(test_dir, "plots", f"episode{episode}.png"),
#                                                 azimuth=90, # 90 or 0 is best angle for the 3D plot 
#                                                 elevation=None,
#                                                 see_from_plane=None)
#                     del plotter

#                     write_report(test_dir, sim_df, env, episode)
                
#                 if args.episodes > 1:
#                     multiplotter = Plotter3DMultiTraj(obstacles=obstacles,
#                                                     path=path,
#                                                     drone_trajectories=all_drone_trajs,
#                                                     cum_rewards=cum_rewards,
#                                                     nosave=False)
#                     multiplotter.plot_scene_and_trajs(save_path=os.path.join(test_dir, "plots", f"multiplot.png"),
#                                                     azimuth=90,
#                                                     elevation=None,
#                                                     see_from_plane=None)