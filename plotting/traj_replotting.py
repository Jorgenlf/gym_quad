import pandas as pd
import numpy as np

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from pv_plotting_3d import Plotter3D, Plotter3DMultiTraj
from gym_quad.envs.LV_VAE_MESH import LV_VAE_MESH
from drl_config import lv_vae_config



high_level_agent_name = "locked_conv" # "locked_conv", "random", "unlocked"

if high_level_agent_name == "locked_conv":
    exp_dir = f'Best_agent_res_gen_1 exp 10005'
    trained_scen = "proficient_perturbed"
    model_name = "test_agent_model_180000"
elif high_level_agent_name == "random":
    exp_dir = f'Best_agent_res_gen_2 exp 32'
    trained_scen = "advanced"
    model_name = "test_agent_model_10000"


test_scen_list = ["horizontal", "vertical", "deadend", "helix", "cave", "house_hard", "house_hard_obstacles", "house_easy", "house_easy_obstacles"]
episode = 0 #Only used in single_ep mode
mode = "single_ep"
mode = "all_eps" # "single_ep" or "all_eps"

output_path = os.path.join('plotting', 'replotting_results', high_level_agent_name)


replot_config = lv_vae_config.copy()
replot_config["max_t_steps"] = 0

for test_scen in test_scen_list:

    retrieve_data_path = os.path.join('log', 'LV_VAE_MESH-v0', exp_dir , trained_scen, 'results_gen', test_scen, model_name)
    sim_data = pd.read_csv(retrieve_data_path + '/sim_df.csv')
    metrics_data = pd.read_csv(retrieve_data_path + '/test_summary.csv')

    env = LV_VAE_MESH(env_config=replot_config,scenario=test_scen)
    obstacles = env.unwrapped.obstacles
    path = env.unwrapped.path


    if mode == "single_ep":
        print(f"Replotting episode {episode}...")
        episode_df = sim_data[sim_data["Episode"]==episode]
        drone_traj = np.stack((episode_df[r"$X$"], episode_df[r"$Y$"], episode_df[r"$Z$"]), axis=-1)
        init_pos = drone_traj[0]

        #Make sure the output path exists
        os.makedirs(os.path.join(output_path, "plots"), exist_ok=True)

        plotter = Plotter3D(obstacles=obstacles, 
                                    path=path, 
                                    drone_traj=drone_traj,
                                    initial_position=init_pos,
                                    save=True,
                                    scene=test_scen) 
        plotter.plot_scene_and_trajs(save_path=os.path.join(output_path, "plots", f"{high_level_agent_name}_{test_scen}_episode{episode}.png"), hv=1, only_scene=False)
        print(f"Episode {episode} replotting done.")    

    elif mode == "all_eps":
        print("Replotting all episodes...")
        #Make sure the output path exists
        os.makedirs(os.path.join(output_path, "plots"), exist_ok=True)
        cum_rewards = {}
        all_drone_trajs = {}
        all_init_pos = {}

        for episode in sim_data["Episode"].unique():
            episode_df = sim_data[sim_data["Episode"]==episode]
            drone_traj = np.stack((episode_df[r"$X$"], episode_df[r"$Y$"], episode_df[r"$Z$"]), axis=-1)
            init_pos = drone_traj[0]
            all_drone_trajs[episode] = drone_traj
            all_init_pos[episode] = all_drone_trajs[episode][0]
            cum_rewards[episode] = episode_df["Reward"].sum()

        if test_scen in ["house_hard", "house_hard_obstacles"]:
            for hv in [1,2]: # View from two angles, must have separate instances. Solved with hv variable in plotter func. "hv=HouseView"
                multiplotter = Plotter3DMultiTraj(obstacles=obstacles,
                                                    path=path,
                                                    drone_trajs=all_drone_trajs,
                                                    initial_position=init_pos,
                                                    cum_rewards=cum_rewards,
                                                    scene=test_scen,
                                                    save=True)
                multiplotter.plot_scene_and_trajs(save_path=os.path.join(output_path, "plots", f"{high_level_agent_name}_{test_scen}_multiplot_hv{hv}.png"),
                                                azimuth=90,
                                                hv=hv)
                del multiplotter
        else:
            multiplotter = Plotter3DMultiTraj(obstacles=obstacles,
                                            path=path,
                                            drone_trajs=all_drone_trajs,
                                            initial_position=init_pos,
                                            cum_rewards=cum_rewards,
                                            scene=test_scen,
                                            save=True)
            multiplotter.plot_scene_and_trajs(save_path=os.path.join(output_path, "plots", f"{high_level_agent_name}_{test_scen}_multiplot.png"),
                                            azimuth=90)
        print("All episodes replotting done.")
    else:
        print("Invalid mode. Choose either 'single_ep' or 'all_eps'")