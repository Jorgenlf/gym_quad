import os
import json
import gymnasium as gym
import gym_quad
import stable_baselines3.common.results_plotter as results_plotter
import numpy as np
import torch
import multiprocessing
import glob
import re
from typing import Callable

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback #Can remove if using tensorboard logger #TODO
from gym_quad import lv_vae_config
# from logger import TensorboardLogger #TODO uncomment when global n_steps is fixed

from PPO_feature_extractor import *
from utils import parse_experiment_info

import warnings
# Filter out the specific warning
#NB this is a temporary fix to avoid the warning from pytorch3d
#Need the mtl file if we want actual images.
warnings.filterwarnings("ignore", message="No mtl file provided", category=UserWarning, module="pytorch3d.io.obj_io")

def event_with_probability(x):
    # Generate a random number between 0 and 1
    random_number = np.random.rand()
    
    # Check if the random number is less than or equal to the probability threshold x
    if random_number <= x:
        return True
    else:
        return False


# scenarios = ["line","line_new","horizontal_new", "3d_new","intermediate"]
mode = None
scenarios = ["proficient"]

mode = "curriculum"
stages = ["line_new","3d_new","intermediate","proficient","expert"]
#TODO add a scenario where theres one obstacle close to path, but not on path which we insert after 3d_new before intermediate


'''
From kulkarni paper:
The neural network is trained with an adaptive learning rate initialized at lr = 10−4. 
The discount factor is set to γ = 0.98. 
The neural network is trained with 1024 environments simulated in parallel with an average time step of 0.1s 
and rollout buffer size set to 32. 
We train this policy for approximately 26 × 10^6 environment steps aggregated over all agents.
'''
#TODO implement the above hyperparameters

PPO_hyperparams = {
    'n_steps': 1024, # lv_vae_config["max_t_steps"] #TODO double check what is reasobale when considered against the time steps of the environment
    #'learning_rate': 2.5e-4, #10e-4, #2.5e-4,old # Try default (3e-4)
    'batch_size': 64,
    'gae_lambda': 0.95,
    'gamma': 0.99, #old:0.99,
    'n_epochs': 4,
    #'clip_range': 0.2,
    'ent_coef': 0.001, 
    'verbose': 2,
    'device':'cuda', #Will be used for both feature extractor and PPO
    #"optimizer_class":torch.optim.Adam, #Throws error (not hos Eirik :)) Now it does idk why sorry man
    #"optimizer_kwargs":{"lr": 10e-4}
}
'''
Kulkarni paper:
We define a neural network architecture containing 3 fullyconnected layers consisting of 
512, 256 and 64 neurons each with an ELU activation layer, followed by a GRU with a hidden layer size of 64. 
Given an observation vector ot, the policy outputs a 3-dimensional action command at = [at,1, at,2, at,3] with values in [-1, 1]
'''

#encoder_path = f"{os.getcwd()}/VAE_encoders/encoder_conv1_experiment_73_seed0_dim32.json"
encoder_path = None

policy_kwargs = dict(
    features_extractor_class = PerceptionIMUDomainExtractor,
    features_extractor_kwargs = dict(img_size=lv_vae_config["compressed_depth_map_size"],
                                     features_dim=lv_vae_config["latent_dim"],
                                     device = PPO_hyperparams['device'],
                                     lock_params=False,
                                     pretrained_encoder_path = encoder_path),
    net_arch = dict(pi=[64, 64], vf=[64, 64])#The PPO network architecture policy and value function
)
#From Ørjan:    net_arch = dict(pi=[128, 64, 32], vf=[128, 64, 32])
#SB3 default:   net_arch = dict(pi=[64, 64], vf=[64, 64])
#From Kulkarni: net_arch = dict(pi=[512, 256, 64], vf=[512, 256, 64]) #NB: GRU is not included in this
#There exists a recurrent PPO using LSTM which could be used as replacement for the GRU

#-----#------#-----#Temp fix to make the global n_steps variable work pasting the tensorboardlogger class here#-----#------#-----#
class TensorboardLogger(BaseCallback):
    '''
     A custom callback for tensorboard logging.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    
    To open tensorboard after/during training, run the following command in terminal:
    tensorboard --logdir 'log/LV_VAE-v0/Experiment x'
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

    # def _on_training_start(self):
    #     self._log_freq = 1000  # log every 1000 calls

    #     output_formats = self.logger.output_formats
    #     # Save reference to tensorboard formatter object
    #     # note: the failure case (not formatter found) is not handled here, should be done with try/except.
    #     self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

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
        # Logging data at the end of an episode - must check if the environment is done
        done_array = self.locals["dones"]
        n_done = np.sum(done_array).item()
    
        # Only log if any workers are actually at the end of an episode

        global n_steps
        global avg_reach_end_reward        
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

            #Can log error and state here if wanted

        if (n_steps + 1) % 20000 == 0:
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
python train3d.py --exp_id x --n_cpu x
"""

if __name__ == '__main__':

    print('\nTOTAL CPU CORE COUNT:', multiprocessing.cpu_count(),"\n")

    experiment_dir, _, args = parse_experiment_info()
    avg_reach_end_reward = 0

    if mode == "curriculum":
        #Folder setup
        agents_dir = os.path.join(experiment_dir, mode, "agents")
        tensorboard_dir = os.path.join(experiment_dir, mode, "tensorboard")
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(agents_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        PPO_hyperparams["tensorboard_log"] = tensorboard_dir
        seed=np.random.randint(0,10000)
        try:
            with open(f'{experiment_dir}/{mode}/seed.txt', 'r') as file:
                seed = int(file.read())
        except FileNotFoundError: 
            with open(f'{experiment_dir}/{mode}/seed.txt', 'w') as file:
                file.write(str(seed))
        print("set seed"+" "+ experiment_dir) 
        
        scen = stages.pop(0)

        prev_scen = []
        sprinkle_in_prev_scen = False
        last_stage = stages[-1]
        firstiter = True

        while stages != []:            
            if firstiter: 
                firstiter = False
                #Initial setup of the first scenario
                #N core training setup
                num_envs = args.n_cpu
                assert num_envs > 0, "Number of cores must be greater than 0"
                assert num_envs <= multiprocessing.cpu_count(), "Number of cores must be less than or equal to the number of cores available"
                print("USING", num_envs, "CORES FOR TRAINING") 
                print("INITIALIZING", num_envs, scen.upper(), "ENVIRONMENTS...")
                env = SubprocVecEnv(
                    [lambda: Monitor(gym.make(args.env, scenario=scen), agents_dir, allow_early_resets=True)
                    for i in range(num_envs)]
                )
                print("DONE INITIALIZING ENVIRONMENTS")

                #Agent setup
                print("INITIALIZING AGENT...")
                agent = PPO('MultiInputPolicy', env, **PPO_hyperparams, policy_kwargs=policy_kwargs, seed=seed) #Policykwargs To use homemade feature extractor and architecture
                print("DONE INITIALIZING AGENT")

                #Training
                timesteps = int(15e6) 
                n_steps = 0
                print("TRAINING FOR", timesteps, "TIMESTEPS")
                agent.learn(total_timesteps=timesteps, tb_log_name="PPO",callback=TensorboardLogger(agents_dir=agents_dir),progress_bar=True)
        

            # if prev_scen != []: #If there is a previous scenario #TODO decide wether to use this or not
            #     sprinkle_in_prev_scen = event_with_probability(0.1) #20% chance of sprinkling in a previous scenario 
            #     #TODO decide if this should be timestep dependepent or random

            print("avg_reach_end_reward", avg_reach_end_reward)

            if (avg_reach_end_reward > lv_vae_config["rew_reach_end"]*0.8 and scen != last_stage):
                print("REACHED TARGET REWARD AT TIMESTEP:", continual_step,"\nIN STAGE", scen, "\nMOVING ON TO NEXT STAGE:", stages[0].upper())
                scen = stages.pop(0)
                prev_scen.append(scen)

                #N core training setup
                num_envs = args.n_cpu
                assert num_envs > 0, "Number of cores must be greater than 0"
                assert num_envs <= multiprocessing.cpu_count(), "Number of cores must be less than or equal to the number of cores available"
                print("USING", num_envs, "CORES FOR TRAINING") 
                print("INITIALIZING", num_envs, scen.upper(), "ENVIRONMENTS...")
                env = SubprocVecEnv(
                    [lambda: Monitor(gym.make(args.env, scenario=scen), agents_dir, allow_early_resets=True)
                    for i in range(num_envs)]
                )
                print("DONE INITIALIZING ENVIRONMENTS")

                #Agent setup
                print("INITIALIZING AGENT...")
                agents = glob.glob(os.path.join(experiment_dir, mode, "agents", "model_*.zip"))
                if agents == []:
                    continual_step = 0
                else:
                    continual_step = max([int(*re.findall(r'\d+', os.path.basename(os.path.normpath(file)))) for file in agents])

                if continual_step == 0: 
                    agent = PPO('MultiInputPolicy', env, **PPO_hyperparams, policy_kwargs=policy_kwargs, seed=seed) #Policykwargs To use homemade feature extractor and architecture
                else:
                    continual_model = os.path.join(experiment_dir, mode, "agents", f"model_{continual_step}.zip")
                    agent = PPO.load(continual_model, _init_setup_model=True, env=env, **PPO_hyperparams)
                print("DONE INITIALIZING AGENT")

                #Training
                best_mean_reward, n_steps, timesteps = -np.inf, continual_step, int(15e6) - num_envs*continual_step

                print("TRAINING FOR", timesteps, "TIMESTEPS")
                agent.learn(total_timesteps=timesteps, tb_log_name="PPO",callback=TensorboardLogger(agents_dir=agents_dir),progress_bar=True)
                
                print("FINISHED TRAINING AGENT IN", scen.upper())
                save_path = os.path.join(agents_dir, "last_model.zip")
                agent.save(save_path)
                print("SAVE SUCCESSFUL")
    
    #Old mode each scen trained separately and saved 15mill timesteps each easiest to train one at a time scenarios = ["line"]                  
    else:         
        for i, scen in enumerate(scenarios):

            agents_dir = os.path.join(experiment_dir, scen, "agents")
            tensorboard_dir = os.path.join(experiment_dir, scen, "tensorboard")
            scenario_dir = os.path.join(experiment_dir, scen)

            os.makedirs(scenario_dir, exist_ok=True)
            os.makedirs(experiment_dir, exist_ok=True)
            os.makedirs(agents_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)

            with open(os.path.join(scenario_dir, 'lv_vae_config.json'), 'w') as file:
                json.dump(lv_vae_config, file)
            with open(os.path.join(scenario_dir, 'ppo_config.json'), 'w') as file:
                json.dump(PPO_hyperparams, file)

            PPO_hyperparams["tensorboard_log"] = tensorboard_dir
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
            
        
        num_envs = args.n_cpu
        assert num_envs > 0, "Number of cores must be greater than 0"
        assert num_envs <= multiprocessing.cpu_count(), "Number of cores must be less than or equal to the number of cores available"

        print("USING", num_envs, "CORES FOR TRAINING") 
        print("INITIALIZING", num_envs, scen.upper(), "ENVIRONMENTS...")
        if num_envs > 1:
            env = SubprocVecEnv(
                [lambda: Monitor(gym.make(args.env, scenario=scen), agents_dir, allow_early_resets=True)
                for i in range(num_envs)]
            )
        else:
            env = DummyVecEnv(
                [lambda: Monitor(gym.make(args.env, scenario=scen), agents_dir,allow_early_resets=True)]
            )
        print("DONE INITIALIZING ENVIRONMENTS")

        print("INITIALIZING AGENT...")
        agents = glob.glob(os.path.join(experiment_dir, scen, "agents", "model_*.zip"))
        if agents == []:
            continual_step = 0
        else:
            continual_step = max([int(*re.findall(r'\d+', os.path.basename(os.path.normpath(file)))) for file in agents])

        if scen == "proficient" and continual_step == 0: #TODO fix this so dont need to manually change scenario when training new agent(?)
            agent = PPO('MultiInputPolicy', env, **PPO_hyperparams, policy_kwargs=policy_kwargs, seed=seed) #Policykwargs To use homemade feature extractor and architecture
        elif continual_step == 0:
            continual_model = os.path.join(experiment_dir, scenarios[i-1], "agents", "last_model.zip")
            agent = PPO.load(continual_model, _init_setup_model=True, env=env, **PPO_hyperparams)
        else:
            continual_model = os.path.join(experiment_dir, scen, "agents", f"model_{continual_step}.zip")
            agent = PPO.load(continual_model, _init_setup_model=True, env=env, **PPO_hyperparams)
        print("DONE INITIALIZING AGENT")

        best_mean_reward, n_steps, timesteps = -np.inf, continual_step, int(15e6) - num_envs*continual_step
        print("TRAINING FOR", timesteps, "TIMESTEPS")
        agent.learn(total_timesteps=timesteps, tb_log_name="PPO",callback=TensorboardLogger(agents_dir=agents_dir),progress_bar=True)
        print("FINISHED TRAINING AGENT IN", scen.upper())
        save_path = os.path.join(agents_dir, "last_model.zip")
        agent.save(save_path)
        print("SAVE SUCCESSFUL")