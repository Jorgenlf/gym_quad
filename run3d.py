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
To run a trained agent, run the following command in terminal exchange x for the experiment id you want to train:
python run3d.py --env "" --exp_id x --run_scenario "" --trained_scenario "" --agent x --episodes x --manual_control False --RT_vis True
"""

if __name__ == "__main__":
    experiment_dir, agent_path, args = parse_experiment_info()

    # Uncomment and run in debug mode for debugging of code---------
    # experiment_dir = "./log\LV_VAE-v0\Experiment 1"
    # agent_path = "./log/LV_VAE-v0/Experiment 1/intermediate/agents/model_150000.pkl"
    # args = Namespace(env='LV_VAE-v0', exp_id=1, scenario='3d', controller_scenario='intermediate', controller=150000, episodes=1) #OLD
    #------------------------------------------------------
    # args = Namespace(manual_control=True)
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

        if args.RT_vis == False:
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
        elif args.RT_vis == True: #TODO add the stoarge of variables to this loop below such as above
            for episode in range(args.episodes):
                try:
                    obs,info = env.reset()
                    visualizer = EnvironmentVisualizer(env.obstacles, env.quadcopter.position, env.quadcopter.attitude)
                    visualizer.draw_path(env.path.waypoints)
                    while True:
                        action = agent.predict(env.observation, deterministic=True)[0]
                        _, _, done, _, _ = env.step(action)
                        print(action)
                        visualizer.update_quad_visual(env.quadcopter.position, env.quadcopter.attitude)           
                        app.process_events()
                        if done:
                            break
                    env.close()    
                except NameError:
                    pass

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

            visualizer = EnvironmentVisualizer(env.obstacles, env.quadcopter.position, env.quadcopter.attitude)
            visualizer.draw_path(env.path.waypoints)
            @visualizer.scene.events.key_press.connect
            def on_key(event):
                nonlocal input
                if event.key == 'd': # Right #TODO make these actually make the quadcopter move wrt to x axis
                    input = [1, 0, -1]
                elif event.key == 'a': # Left
                    input = [1, 0,1]
                elif event.key == 'w': # Up
                    input = [1, 1, -1]
                elif event.key == 's': # Down
                    input = [-1, -1, -1]
                elif event.key == 'escape':
                    env.close()
            
            while True:
                obs, rew, done, info, _ = env.step(action=input)
                visualizer.update_quad_visual(env.quadcopter.position, env.quadcopter.attitude)           
                app.process_events()
                if done:
                    done = False
                    env.reset()

        env = gym.make("LV_VAE-v0", scenario="intermediate")
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
    # "advanced": 
    "expert": 

# Testing scenarios
    "test_path": 
    "test": 
    "test_current": 
    "horizontal": 
    "vertical": 
    "deadend": 
'''        