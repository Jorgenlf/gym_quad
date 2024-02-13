import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_quad
import os
import glob
import re

from gym_quad.utils.controllers import PI, PID
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from utils import *
from argparse import Namespace

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Tensflow logging level
# '0': (default) logs all messages.
# '1': logs messages with level INFO and above.
# '2': logs messages with level WARNING and above.
# '3': logs messages with level ERROR and above.


if __name__ == "__main__":
    experiment_dir, agent_path, args = parse_experiment_info()

    # Uncomment and run in debug mode for debugging---------
    # experiment_dir = "./log\WaypointPlanner-v0\Experiment 100"
    # agent_path = "./log/WaypointPlanner-v0/Experiment 100/intermediate/agents/model_820000.pkl"
    # args = Namespace(env='WaypointPlanner-v0', exp_id=100, scenario='3d', controller_scenario='intermediate', controller=820000, episodes=1)
    #-------------------------

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

    env = gym.make(args.env, scenario=args.scenario)
    agent = PPO.load(agent_path)

    for episode in range(args.episodes):
        try:
            episode_df, env = simulate_environment(episode, env, agent)
            sim_df = pd.concat([sim_df, episode_df], ignore_index=True)
        except NameError:
            sim_df = episode_df
        
        write_report(test_dir, sim_df, env, episode)
        # plot_attitude(sim_df)
        #plot_velocity(sim_df)
        #plot_angular_velocity(sim_df)
        #plot_control_inputs([sim_df])
        #plot_control_errors([sim_df])
        plot_3d(env, sim_df[sim_df['Episode']==episode], test_dir)
        #plot_current_data(sim_df)

