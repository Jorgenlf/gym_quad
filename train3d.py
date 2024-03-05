
import os
import gymnasium as gym
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
from stable_baselines3.common.callbacks import BaseCallback #Can remove if using tensorboard logger #TODO
# from logger import TensorboardLogger #TODO uncomment when global n_steps is fixed

from lidarCNN import *
from utils import parse_experiment_info

print('CPU COUNT:', multiprocessing.cpu_count())

# scenarios = ["line","line_new","horizontal_new", "3d_new","intermediate"]
scenarios = ["line_new"]

#From kulkarni paper:
'''
The neural network is trained with an adaptive learning rate initialized at lr = 10−4. 
The discount factor is set to γ = 0.98. 
The neural network is trained with 1024 environments simulated in parallel with an average time step of 0.1s 
and rollout buffer size set to 32. 
We train this policy for approximately 26 × 10^6 environment steps aggregated over all agents.
'''
#TODO implement the above hyperparameters


hyperparams = {
    'n_steps': 1024, #TODO double check what is reasobale when considered against the time steps of the environment
    'learning_rate': 10e-4, #2.5e-4,old 
    'batch_size': 64,
    'gae_lambda': 0.95,
    'gamma': 0.98, #old:0.99,
    'n_epochs': 4,
    'clip_range': 0.2,
    'ent_coef': 0.01, 
    'verbose': 2,
    # "optimizer_class":torch.optim.Adam, #Throws error 
    # "optimizer_kwargs":{"lr": 10e-4}
    # 'optimizer_class': torch.optim.Adam, #Throws error
    # 'device':'cuda' #unsure if cuda wanted as default as dont have nvidia gpu
}

policy_kwargs = dict(
    features_extractor_class = PerceptionNavigationExtractor,
    features_extractor_kwargs = dict(sensor_dim_x=15,sensor_dim_y=15,features_dim=32),
    net_arch = [dict(pi=[128, 64, 32], vf=[128, 64, 32])]
)

#-----#------#-----#Temp fix to make the global n_steps variable work pasting the tensorboardlogger class here#-----#------#-----#
class TensorboardLogger(BaseCallback):
    '''
     A custom callback for tensorboard logging.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    To open tensorboard after training run the following command in terminal:
    tensorboard --logdir Path/to/tensorboard_dir
    example path: 'C:/Users/jflin/Code/Drone3D/gym_quad/log/LV_VAE-v0/Experiment 6/line/tensorboard'
    '''

    def __init__(self, agents_dir=None, verbose=0,):
        super().__init__(verbose)
        #From tensorboard logger
        self.n_episodes = 0
        self.ep_reward = 0
        self.ep_length = 0
        #from stats callback
        self.agents_dir = agents_dir
        self.n_steps = 0
        self.n_calls=0
        self.prev_stats=None
        self.ob_names=["u","v","w","roll","pitch","yaw","p","q","r","nu_c0","nu_c1","nu_c2","chi_err","upsilon_err","chi_err_1","upsilon_err_1","chi_err_5","upsilon_err_5"]
        self.state_names=["x","y","z","roll","pitch","yaw","u","v","w","p","q","r"]
        self.error_names=["e", "h"]

    # Those variables will be accessible in the callback
    # (they are defined in the base class)
    # The RL model
    # self.model = None  # type: BaseAlgorithm
        
    # An alias for self.model.get_env(), the environment used for training
    # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        
    # Number of time the callback was called
    # self.n_calls = 0  # type: int
        
    # self.num_timesteps = 0  # type: int
    # local and global variables
    # self.locals = None  # type: Dict[str, Any]
    # self.globals = None  # type: Dict[str, Any]
        
    # The logger object, used to report things in the terminal
    # self.logger = None  # stable_baselines3.common.logger
    # # Sometimes, for event callback, it is useful
    # # to have access to the parent object
    # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        ###From tensorboard logger###
        # Logging data at the end of an episode - must check if the environment is done
        done_array = self.locals["dones"]
        n_done = np.sum(done_array).item()
    
        # Only log if any workers are actually at the end of an episode
        ###From tensorboard logger end###

        ###From stats callbacks###
        stats  = {"path_adherence": [],             #self.reward_path_following_sum, 
                "collision_avoidance_reward": [],   #self.reward_collision_avoidance_sum,
                "collision_reward": [],             #self.reward_collision,
                "obs":[],                           #self.past_obs,
                "states":[],                        #self.past_states,
                "errors":[]}                        #self.past_errors
        
        for info in self.locals["infos"]:
                stats["path_adherence"].append(info["path_adherence"])
                stats["collision_avoidance_reward"].append(info["collision_avoidance_reward"])
                stats["collision_reward"].append(info["collision_reward"])
                # stats["obs"].append(info["obs"])
                stats["states"].append(info["state"])
                stats["errors"].append(info["errors"])

        global n_steps
        ###From stats callback end###

        if n_done > 0:
            # Record the cumulative number of finished episodes
            self.n_episodes += n_done
            self.logger.record('time/episodes', self.n_episodes)

            # Fetch data from the info dictionary in the environment (convert tuple->np.ndarray for easy indexing)
            infos = np.array(self.locals["infos"])[done_array]

            avg_reward = 0
            avg_length = 0
            avg_collision_reward = 0
            avg_collision_avoidance_reward = 0
            avg_path_adherence = 0
            avg_path_progression = 0
            avg_reach_end_reward = 0
            avg_existence_reward = 0
            for info in infos:
                avg_reward += info["reward"]
                avg_length += info["env_steps"]
                avg_collision_reward += info["collision_reward"]
                avg_collision_avoidance_reward += info["collision_avoidance_reward"]
                avg_path_adherence += info["path_adherence"]
                avg_path_progression += info["path_progression"]
                avg_reach_end_reward += info['reach_end_reward'] 
                avg_existence_reward += info['existence_reward']

            avg_reward /= n_done
            avg_length /= n_done
            avg_collision_reward /= n_done
            avg_collision_avoidance_reward /= n_done
            avg_path_adherence /= n_done
            avg_path_progression /= n_done
            avg_reach_end_reward /= n_done
            avg_existence_reward /= n_done

            # Write to the tensorboard logger
            self.logger.record("episodes/avg_reward", avg_reward)
            self.logger.record("episodes/avg_length", avg_length)
            self.logger.record("episodes/avg_collision_reward", avg_collision_reward)
            self.logger.record("episodes/avg_collision_avoidance_reward", avg_collision_avoidance_reward)
            self.logger.record("episodes/avg_path_adherence_reward", avg_path_adherence)
            self.logger.record("episodes/avg_path_progression_reward", avg_path_progression)
            self.logger.record("episodes/avg_reach_end_reward", avg_reach_end_reward)
            self.logger.record("episodes/avg_existence_reward", avg_existence_reward)

            #From stats callback

            #Got this error when trying to use the stats callback code below
            # File "C:\Users\jflin\Code\Drone3D\gym_quad\train3d.py", line 218, in _on_step
            # for stat in self.prev_stats[i].keys():
            # KeyError: 13

            # for i in range(len(done_array)): #TODO somewhat fixed, cant find the variables in tensorboard though
            #     if done_array[i]:
            if self.prev_stats is not None:
                # for stat in self.prev_stats[i].keys():
                for stat in self.prev_stats.keys():
                    self.logger.record('stats/' + stat, self.prev_stats[stat])
                    # for stat in stats[i].keys():
                    #     self.logger.record('stats/' + stat, stats[i][stat])
        
        #From stats callback
        self.prev_stats = stats

        if (n_steps + 1) % 10000 == 0:
            _self = self.locals.get("self")
            _self.save(os.path.join(self.agents_dir, "model_" + str(n_steps+1) + ".zip"))
        n_steps += 1
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
#-----#------#-----#Temp fix to make the global n_steps variable work pasting the tensorboardlogger class above#-----#------#-----#
#TODO make it work without a global variable please

"""
To train the agent, run the following command in terminal exchange x for the experiment id you want to train:
python train3d.py --exp_id x
"""

if __name__ == '__main__':
    
    experiment_dir, _, args = parse_experiment_info()
        
    for i, scen in enumerate(scenarios):
        agents_dir = os.path.join(experiment_dir, scen, "agents")
        tensorboard_dir = os.path.join(experiment_dir, scen, "tensorboard")
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(agents_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        hyperparams["tensorboard_log"] = tensorboard_dir
        seed=np.random.randint(0,10000)
        try:
            with open(f'{experiment_dir}/{scen}/seed.txt', 'r') as file:
                seed = int(file.read())
        except FileNotFoundError: 
            with open(f'{experiment_dir}/{scen}/seed.txt', 'w') as file:
                file.write(str(seed))
        print("set seed"+" "+ experiment_dir) 

        if os.path.exists(os.path.join(experiment_dir, scen, "agents", "last_model.zip")):
            print(experiment_dir, "ALREADY FINISHED TRAINING IN,", scen.upper(), "SKIPPING TO THE NEXT STAGE")
            if scen!="intermediate":
                continue

    num_envs = multiprocessing.cpu_count() - 2
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

    agents = glob.glob(os.path.join(experiment_dir, scen, "agents", "model_*.zip"))
    if agents == []:
        continual_step = 0
    else:
        continual_step = max([int(*re.findall(r'\d+', os.path.basename(os.path.normpath(file)))) for file in agents])

    if scen == "line_new" and continual_step == 0:
        agent = PPO('MultiInputPolicy', env, **hyperparams,policy_kwargs=policy_kwargs,seed=seed)
    elif continual_step == 0:
        continual_model = os.path.join(experiment_dir, scenarios[i-1], "agents", "last_model.zip")
        agent = PPO.load(continual_model, _init_setup_model=True, env=env, **hyperparams)
    else:
        continual_model = os.path.join(experiment_dir, scen, "agents", f"model_{continual_step}.zip")
        agent = PPO.load(continual_model, _init_setup_model=True, env=env, **hyperparams)
    print("DONE")

    best_mean_reward, n_steps, timesteps = -np.inf, continual_step, int(15e6) - num_envs*continual_step
    print("TRAINING FOR", timesteps, "TIMESTEPS")
    agent.learn(total_timesteps=timesteps, tb_log_name="PPO",callback=TensorboardLogger(agents_dir=agents_dir),progress_bar=True)
    print("FINISHED TRAINING AGENT IN", scen.upper())
    save_path = os.path.join(agents_dir, "last_model.zip")
    agent.save(save_path)
    print("SAVE SUCCESSFUL")