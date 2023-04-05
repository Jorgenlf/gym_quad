
import os
import gym
import gym_quad
import stable_baselines3.common.results_plotter as results_plotter
import numpy as np
import torch
import multiprocessing
from typing import Callable
import glob
import re

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from lidarCNN import *
from utils import parse_experiment_info

print('CPU COUNT:', multiprocessing.cpu_count())

# scenarios = ["line","line_new","horizontal_new", "3d_new","intermediate"]
scenarios = ["3d_new"]

hyperparams = {
    'n_steps': 1024,
    'learning_rate': 2.5e-4,
    'batch_size': 64,
    'gae_lambda': 0.95,
    'gamma': 0.99,
    'n_epochs': 4,
    'clip_range': 0.2,
    'ent_coef': 0.001,
    'verbose': 2,
    'device':'cuda'
}

policy_kwargs = dict(
                        features_extractor_class = PerceptionNavigationExtractor,
                        features_extractor_kwargs = dict(sensor_dim_x=15,sensor_dim_y=15,features_dim=32),
                        net_arch=[dict(pi=[128, 64, 32], vf=[128, 64, 32])]
                    )

class StatsCallback(BaseCallback):
    def __init__(self):
        self.n_steps = 0
        self.n_calls=0
        self.prev_stats=None
        self.ob_names=["u","v","w","roll","pitch","yaw","p","q","r","nu_c0","nu_c1","nu_c2","chi_err","upsilon_err","chi_err_1","upsilon_err_1","chi_err_5","upsilon_err_5"]
        self.state_names=["x","y","z","roll","pitch","yaw","u","v","w","p","q","r"]
        self.error_names=["e", "h"]
    
    def _on_step(self):
        done_array=np.array(self.locals.get("dones") if self.locals.get("dones") is not None else self.locals.get("dones"))
        stats=self.locals.get("self").get_env().env_method("get_stats")
        global n_steps
        
        for i in range(len(done_array)):
            if done_array[i]:
                if  self.prev_stats is not None:
                    for stat in self.prev_stats[i].keys():
                        self.logger.record('stats/' + stat, self.prev_stats[i][stat])
        self.prev_stats=stats

        if (n_steps + 1) % 50000 == 0:
            _self = self.locals.get("self")
            _self.save(os.path.join(agents_dir, "model_" + str(n_steps+1) + ".pkl"))
        n_steps += 1
        return True


if __name__ == '__main__':
    experiment_dir, _, args = parse_experiment_info()
    
    seed=np.random.randint(0,10000)
    with open('seed.txt', 'w') as file:
        file.write(str(seed))
    print("set seed"+" "+ experiment_dir)

    for i, scen in enumerate(scenarios):
        agents_dir = os.path.join(experiment_dir, scen, "agents")
        tensorboard_dir = os.path.join(experiment_dir, scen, "tensorboard")
        os.makedirs(agents_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        hyperparams["tensorboard_log"] = tensorboard_dir

        if os.path.exists(os.path.join(experiment_dir, scen, "agents", "last_model.pkl")):
            print(experiment_dir, "ALREADY FINISHED TRAINING IN,", scen.upper(), "SKIPPING TO THE NEXT STAGE")
            if scen!="intermediate":
                continue

        num_envs = 1
        # num_envs = multiprocessing.cpu_count() - 1
        print("INITIALIZING", num_envs, scen.upper(), "ENVIRONMENTS...", end="")
        if num_envs > 1:
            env = SubprocVecEnv(
                [lambda: Monitor(gym.make(args.env, scenario=scen), agents_dir, allow_early_resets=True)
                for i in range(num_envs)]
            )
        else:
            env = DummyVecEnv(
                [lambda: Monitor(gym.make(args.env, scenario=scen), agents_dir,allow_early_resets=True)]
            )
        print("DONE")
        print("INITIALIZING AGENT...", end="")

        agents = glob.glob(os.path.join(experiment_dir, scen, "agents", "model_*.pkl"))
        if agents == []:
            continual_step = 0
        else:
            continual_step = max([int(*re.findall(r'\d+', os.path.basename(os.path.normpath(file)))) for file in agents])

        if scen == "3d_new" and continual_step == 0:
            agent = PPO('MultiInputPolicy', env, **hyperparams,policy_kwargs=policy_kwargs,seed=seed)
        elif continual_step == 0:
            continual_model = os.path.join(experiment_dir, scenarios[i-1], "agents", "last_model.pkl")
            agent = PPO.load(continual_model, _init_setup_model=True, env=env, **hyperparams)
        else:
            continual_model = os.path.join(experiment_dir, scen, "agents", f"model_{continual_step}.pkl")
            agent = PPO.load(continual_model, _init_setup_model=True, env=env, **hyperparams)
        print("DONE")

        best_mean_reward, n_steps, timesteps = -np.inf, continual_step, int(15e6) - num_envs*continual_step
        print("TRAINING FOR", timesteps, "TIMESTEPS")
        agent.learn(total_timesteps=timesteps, tb_log_name="PPO2",callback=StatsCallback())
        print("FINISHED TRAINING AGENT IN", scen.upper())
        save_path = os.path.join(agents_dir, "last_model.pkl")
        agent.save(save_path)
        print("SAVE SUCCESSFUL")
