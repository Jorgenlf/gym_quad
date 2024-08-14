import numpy as np
import gymnasium as gym
import os
import glob
import re
import time
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from pv_plotting_3d import Plotter3D, Plotter3DMultiTraj

from stable_baselines3 import PPO
from train_run_utils import *
from gym_quad import register_lv_vae_envs
from drl_config import lv_vae_config

import warnings
# Filter out the specific warning
#NB this is a temporary fix to avoid the warning from pytorch3d #Need the mtl file if we want actual images.
warnings.filterwarnings("ignore", message="No mtl file provided", category=UserWarning, module="pytorch3d.io.obj_io")
warnings.filterwarnings("ignore", message="Attempting to set window_size on an unavailable render widow.", category=UserWarning, module="pyvista.plotting.plotter")
warnings.filterwarnings("ignore", message="This plotter is closed and cannot be scaled. Using the last saved image. Try using the `image_scale` property directly.", category=UserWarning, module="pyvista.plotting.plotter")

def unregister_env(env_id):
    if env_id in gym.envs.registry:
        del gym.envs.registry[env_id]

def run_test(trained_scen, agent, test_scen, result_config, args, base_experiment_dir):
    agent_name = os.path.splitext(os.path.basename(agent))[0]
    print(f"Running test for agent {agent_name} in scenario {test_scen}")

    if test_scen == "house_easy" or test_scen == "house_hard":
        result_config["la_dist"] = 0.5
        result_config["s_max"] = 2
        result_config["s_max"] = 2
        result_config["max_t_steps"] = 6000 #Needs more time in the house
    elif test_scen == "house_easy_obstacles" or test_scen == "house_hard_obstacles":
        result_config["la_dist"] = 1
        result_config["s_max"] = 2
        result_config["max_t_steps"] = 6000 #Needs more time in the house
    elif test_scen == "helix":
        result_config["max_t_steps"] = 6000 #Needs more time in the helix        
    else:
        result_config["la_dist"] = lv_vae_config["la_dist"]
        result_config["s_max"] = lv_vae_config["s_max"]

    unregister_env('LV_VAE_MESH-v0') # To avoid overwriting the envs
    register_lv_vae_envs(result_config)

    # Construct the directory paths
    agent_path = os.path.join(base_experiment_dir, trained_scen)
    results_gen_dir = os.path.join(agent_path, "results_gen")
    test_scen_dir = os.path.join(results_gen_dir, test_scen)
    os.makedirs(test_scen_dir, exist_ok=True)

    # Generate the test directory name based on agent name
    test_dir = os.path.join(test_scen_dir, f"test_agent_{agent_name}")
    completed_episodes = 0
    resuming = False

    if os.path.exists(test_dir):
        report_path = os.path.join(test_dir, 'report.txt')
        if os.path.exists(report_path):
            with open(report_path, 'r') as file:
                content = file.read()
                match = re.search(r'LAST (\d+) EPISODES AVG', content)
                if match:
                    completed_episodes = int(match.group(1))

            if completed_episodes >= args.episodes:
                print(f"Test directory for agent {agent_name} in scenario {test_scen} already tested all episodes. Skipping...")
                return
            else:
                remaining_episodes = args.episodes - completed_episodes
                print(f"Test directory for agent {agent_name} in scenario {test_scen} already tested {completed_episodes} episodes. Continuing with {remaining_episodes} more episodes...")
                args.episodes = remaining_episodes
                resuming = True

    #If no report exists, we need to start from scratch so let rest of fcn run

    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(test_dir, "plots"), exist_ok=True)

    summary_path = os.path.join(test_dir, 'test_summary.csv')
    if resuming and os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
    else:
        summary = pd.DataFrame(columns=["Episode", "Timesteps", "Avg Absolute Path Error", "Speed", "IAE Cross", "IAE Vertical", "Progression", "Success", "Collision"])
        summary.to_csv(summary_path, index=False)

    env = gym.make(args.env, scenario=test_scen)
    agent_model = PPO.load(agent)

    cum_rewards = {}
    all_drone_trajs = {}
    all_init_pos = {}

    for episode in range(completed_episodes, completed_episodes + args.episodes):
        try:
            episode_df, env = simulate_environment(episode, env, agent_model, test_dir, args.save_depth_maps)
            if resuming:
                sim_df = pd.read_csv(os.path.join(test_dir, 'sim_df.csv'))
                sim_df = pd.concat([sim_df, episode_df], ignore_index=True)
            else:
                sim_df = pd.concat([sim_df, episode_df], ignore_index=True) if 'sim_df' in locals() else episode_df
        except NameError:
            sim_df = episode_df

        path = env.unwrapped.path

        all_drone_trajs[episode] = np.stack((episode_df[r"$X$"], episode_df[r"$Y$"], episode_df[r"$Z$"]), axis=-1)
        all_init_pos[episode] = all_drone_trajs[episode][0]
        cum_rewards[episode] = episode_df['Reward'].sum()

        drone_traj = np.stack((episode_df[r"$X$"], episode_df[r"$Y$"], episode_df[r"$Z$"]), axis=-1)
        init_pos = drone_traj[0]
        obstacles = env.unwrapped.obstacles
    
        plotter = Plotter3D(obstacles=obstacles, 
                                    path=path, 
                                    drone_traj=drone_traj,
                                    initial_position=init_pos,
                                    save=True,
                                    scene=args.run_scenario) 
        plotter.plot_scene_and_trajs(save_path=os.path.join(test_dir, "plots", f"episode{episode}.png"), hv=1, only_scene=False)
        del plotter

        write_report(test_dir, sim_df, env, episode) #This also writes to the summary df containing the report stats per episode

        sim_df.to_csv(os.path.join(test_dir, 'sim_df.csv'), index=False)

    if args.episodes > 1:
        if args.run_scenario in ["house_hard", "house_hard_obstacles"]:
            for hv in [1,2]: # View from two angles, must have separate instances. Solved with hv variable in plotter func. "hv=HouseView"
                multiplotter = Plotter3DMultiTraj(obstacles=obstacles,
                                                  path=path,
                                                  drone_trajs=all_drone_trajs,
                                                  initial_position=init_pos,
                                                  cum_rewards=cum_rewards,
                                                  scene=args.run_scenario,
                                                  save=True)
                multiplotter.plot_scene_and_trajs(save_path=os.path.join(test_dir, "plots", f"multiplot_hv{hv}.png"),
                                                azimuth=90,
                                                hv=hv)
                del multiplotter
        else:
            multiplotter = Plotter3DMultiTraj(obstacles=obstacles,
                                            path=path,
                                            drone_trajs=all_drone_trajs,
                                            initial_position=init_pos,
                                            cum_rewards=cum_rewards,
                                            scene=args.run_scenario,
                                            save=True)
            multiplotter.plot_scene_and_trajs(save_path=os.path.join(test_dir, "plots", f"multiplot.png"),
                                            azimuth=90)


'''
To run this scrip do e.g.:
python result_gen.py --exp_id 19 --episodes 10 --trained_list expert expert_perturbed --test_list horizontal vertical deadend random_corridor house --test_all_agents True
in terminal.

For the final resultsgen do similar to this:
python result_gen.py --exp_id 10006 --episodes 100 --trained_list expert_perturbed --test_list helix house_hard house_hard_obstacles deadend cave house_easy house_easy_obstacles horizontal vertical --test_all_agents True
'''

#Define the config for the results generation #Could import from experiment config file
result_config = lv_vae_config.copy()
result_config["max_t_steps"] = 3500
result_config["recap_chance"] = 0
result_config["perturb_sim"] = True
result_config["min_reward"] = -100e4 
result_config["use_uncaged_drone_mesh"] = True #Decide if we want to use the uncaged drone mesh for collision detection during testing if false uses cylinder (faster)
 
if __name__ == "__main__":
    _s = time.time() #For tracking training time
    _, _, args = parse_experiment_info()
    test_scenarios = args.test_list
    trained_scenarios_to_run = args.trained_list
    
    expdir_string = r"Experiment {}".format(args.exp_id)
    expdir_string = r"A_Filter stage 2 exp {}".format(args.exp_id) #NB This is temp for final res gen
    expdir_string = r"Best_agent_res_gen_1 exp {}".format(args.exp_id) #NB This is temp for final res gen
    expdir_string = r"A_maybe_best_pt_unlocked {}".format(args.exp_id) #NB This is temp for final res gen

    base_experiment_dir = os.path.join(r"./log", r"{}".format(args.env), expdir_string)

    tasks = []
    for trained_scen in trained_scenarios_to_run:
        agent_path = None
        if args.test_all_agents:
            agent_path = os.path.join(base_experiment_dir, trained_scen)
            available_agents = glob.glob(os.path.join(agent_path, "agents", "*.zip"))
        else:
            agent_path = os.path.join(base_experiment_dir, trained_scen)
            available_agents = [f"{base_experiment_dir}/{trained_scen}/agents/model_{args.agent}.zip" if args.agent != None else f"{base_experiment_dir}/{trained_scen}/agents/last_model.zip"]
        
        for agent in available_agents:
            for test_scen in test_scenarios:
                tasks.append((trained_scen, agent, test_scen, result_config.copy(), args, base_experiment_dir))

    # Define batch size and split tasks into batches
    batch_size = 9  # Adjust the batch size based on your system's capacity
    num_batches = len(tasks) // batch_size + int(len(tasks) % batch_size > 0)

    for batch_idx in tqdm(range(num_batches), desc="Total Progress"):
        batch_tasks = tasks[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        print(f"Running batch {batch_idx + 1}/{num_batches} with {len(batch_tasks)} tasks")
        Parallel(n_jobs=-1)(
            delayed(run_test)(*task) for task in batch_tasks
        )
        print(f"Completed batch {batch_idx + 1}/{num_batches}")
    
    print(f"WHOLE RESULTGEN TOOK {time.strftime('%H:%M:%S', time.gmtime(time.time() - _s))}")