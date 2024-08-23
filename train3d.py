import os
import json
import gymnasium as gym
import gym_quad
import numpy as np
import multiprocessing
import glob
import re
import time

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
from gym_quad import register_lv_vae_envs
from logger import TensorboardLogger

from PPO_feature_extractor import *
from train_run_utils import parse_experiment_info

import warnings
# Filter out the specific warning
#NB this is a temporary fix to avoid the warning from pytorch3d #Need the mtl file if we want actual images and not just depth.
warnings.filterwarnings("ignore", message="No mtl file provided", category=UserWarning, module="pytorch3d.io.obj_io")


###---###---### IMPORT THE DRL_CONFIG AND MODIFY IF NEEDED ###---###---###
from drl_config import lv_vae_config
train_config = lv_vae_config.copy()
train_config["max_t_steps"] = 6000
train_config["recap_chance"] = 0.0 # 0.1


###---###---### CHOOSE CURRICULUM SETUP HERE ###---###---### 
"""scenarios = {   "line"                 :  1e6,
                "easy"                 :  1e6,
                "easy_random"          :  1e6, #Randomized pos and att of quad in easy scenario 
                "intermediate"         :  1.5e6,
                "proficient"           :  1.5e6,
                "advanced"             :  2e6, 
                "expert"               :  2e6,
                "proficient_perturbed" :  2e6,
                "expert_perturbed"     :  2e6
             }

scenario_success_threshold = {  "line"                 :  0.6, #TODO make the dict above a list of tuples instead #This is a quick fix
                                "easy"                 :  0.6,
                                "easy_random"          :  0.6,
                                "intermediate"         :  0.8,
                                "proficient"           :  0.8,
                                "advanced"             :  0.9,
                                "expert"               :  0.9,
                                "proficient_perturbed" :  0.95,
                                "expert_perturbed"     :  0.95
                            }"""

"""scenarios = {   "advanced"          :  2e6, 
                "house_easy"        :  2e6,
                "house_easy_obstacles"    :  2e6,
                "house_hard"        :  2e6,
                "house_hard_obstacles"    :  2e6,
             }

scenario_success_threshold = {  "advanced"          :  0.9,
                                "house_easy"        :  0.9,
                                "house_easy_obstacles"    :  0.95,
                                "house_hard"        :  0.95,
                                "house_hard_obstacles"    :  0.95,
                            }"""
scenarios = {   "expert"    :  10e6,
                "house"     :  10e6
                # "expert_perturbed"    :  10e6 
             }

scenario_success_threshold = {  "expert"    :  0.95,
                                "house"     :  1.0 }

k = 100  # 5   # Number of consecutive episode successes that must be above the threshold to move to the next scenario
        # (This is later multiplied by the number of environments to get the total number of successes needed to move on)
        
###---###---### SELECT PPO HYPERPARAMETERS HERE ###---###---###
PPO_hyperparams = {
    'n_steps': 2048, 
    'batch_size': 128,
    'gae_lambda': 0.95,
    'gamma': 0.99, #old:0.99,
    'n_epochs': 10,
    'ent_coef': 0.001,
    'verbose': 2,
    'device':'cuda', #Will be used for both feature extractor and PPO
}

###---###---### SELECT POLICYKWARGS HERE - FEATUREEXTRACTOR AND PPO NETWORK ACRHITECTURE ###---###---###
#VAE
encoder_path = None #f"{os.getcwd()}/VAE_encoders/encoder_conv1_experiment_7_seed1.json"
lock_params = False #True if you want to lock the encoder parameters. False to let them be trained
lock_params_conv = False #True if you want to lock the convolutional layers of the encoder. False to let them be trained

#PPO
ppo_pi_vf_arch = dict(pi = [128,64,32], vf = [128,64,32])

policy_kwargs = dict(
    features_extractor_class = PerceptionIMUDomainExtractor,
    features_extractor_kwargs = dict(img_size=train_config["compressed_depth_map_size"],
                                     features_dim=train_config["latent_dim"],
                                     device = PPO_hyperparams['device'],
                                     lock_params=lock_params,
                                     lock_params_conv = lock_params_conv,
                                     pretrained_encoder_path = encoder_path),
    net_arch = ppo_pi_vf_arch
)

"""
To train the agent, run the following command in terminal select: 
the number of cores to use for training with the --n_cpu flag
the expirment id with the --exp_id flag

python train3d.py --exp_id x --n_cpu x
"""

#To register multiple envs when using SubprocVecEnv
def make_env(env_id, scenario, rank, seed=0):
    def _init():
        register_lv_vae_envs(train_config)  # Register the env in the subprocess
        env = gym.make(env_id, scenario=scenario)
        env.reset(seed=seed + rank)  # Set the seed here via reset
        return Monitor(env)
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    time_trained = 0.0
    print('\nTOTAL CPU CORE COUNT:', multiprocessing.cpu_count())
    experiment_dir, _, args = parse_experiment_info()
    scenario_list = list(scenarios.keys())
    done_training = False
   
    for i, scen in enumerate(scenario_list):
        _s = time.time() #For tracking training time

        print("\nATTEMPT TRAINING IN SCENARIO", scen.upper())
        while os.path.exists(os.path.join(experiment_dir, scen, "agents", "last_model.zip")):
            print(experiment_dir, "ALREADY FINISHED TRAINING IN,", scen.upper(), "MOVING TO THE NEXT STAGE")
            i += 1
            try:
                scen = scenario_list[i]
            except IndexError:
                print("ALL SCENARIOS TRAINED")
                done_training = True
                break
        if done_training:
            break
        
        #Saving configs
        agents_dir = os.path.join(experiment_dir, scen, "agents")
        tensorboard_dir = os.path.join(experiment_dir, scen, "tensorboard")
        config_dir = os.path.join(experiment_dir, scen,"configs")

        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(agents_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)

        with open(os.path.join(config_dir, 'train_config.json'), 'w') as file:
            json.dump(train_config, file)
        with open(os.path.join(config_dir, 'ppo_config.json'), 'w') as file:
            json.dump(PPO_hyperparams, file)
        with open(os.path.join(config_dir, 'curriculum_config.json'), 'w') as file:
            json.dump(scenarios, file)
        with open(os.path.join(config_dir, 'drl_net_arch.json'), 'w') as file: 
            json.dump(ppo_pi_vf_arch, file)    
        with open(os.path.join(config_dir, 'feature_extractor_kwargs.json'), 'w') as file:
            json.dump(policy_kwargs['features_extractor_kwargs'], file)


        PPO_hyperparams["tensorboard_log"] = tensorboard_dir

        seed=np.random.randint(0,10000)
        try:
            with open(f'{experiment_dir}/{scen}/seed.txt', 'r') as file:
                seed = int(file.read())
        except FileNotFoundError: 
            with open(f'{experiment_dir}/{scen}/seed.txt', 'w') as file:
                file.write(str(seed))

        
        num_envs = args.n_cpu
        assert num_envs > 0, "Number of cores must be greater than 0"
        assert num_envs <= multiprocessing.cpu_count(), "Number of cores must be less than or equal to the number of cores available"

        print("\nUSING", num_envs, "CORES FOR TRAINING") 
        print("\nINITIALIZING", num_envs, scen.upper(), "ENVIRONMENTS...",end="")
        if num_envs > 1:
            env = SubprocVecEnv(
                [make_env(args.env, scen, i, seed) for i in range(num_envs)]
            )
        else:
            register_lv_vae_envs(train_config) 
            env = DummyVecEnv(
                [lambda: Monitor(gym.make(args.env, scenario=scen), agents_dir,allow_early_resets=True)]
            )
        print("DONE INITIALIZING ENVIRONMENTS")


        print("\nINITIALIZING AGENT...")
        agents = glob.glob(os.path.join(experiment_dir, scen, "agents", "model_*.zip"))
        if agents == []:
            continual_step = 0
        else:
            continual_step = max([int(*re.findall(r'\d+', os.path.basename(os.path.normpath(file)))) for file in agents])

        if i == 0 and continual_step == 0: #First scenario
            agent = PPO('MultiInputPolicy', env, **PPO_hyperparams, policy_kwargs=policy_kwargs, seed=seed) #Policykwargs To use homemade feature extractor and architecture
        
        if i == 0 and continual_step != 0: #First scenario but not first training
            print("CONTINUING TRAINING FROM", continual_step*num_envs, "TIMESTEPS")
            continual_model = os.path.join(experiment_dir, scen, "agents", f"model_{continual_step}.zip")
            agent = PPO.load(continual_model, _init_setup_model=True, env=env, **PPO_hyperparams)

        if i > 0 and continual_step == 0: #Switching to new scenario using the last model from previous scenario
            print("MOVING FROM SCENARIO", scenario_list[i-1].upper(), "TO", scen.upper())
            print("CONTINUING TRAINING FOR ANOTHER", scenarios[scen], "TIMESTEPS")
            continual_model = os.path.join(experiment_dir, scenario_list[i-1], "agents", "last_model.zip")
            agent = PPO.load(continual_model, _init_setup_model=True, env=env, **PPO_hyperparams)
        
        if i > 0 and continual_step != 0: #Continuing training in most recent scenario
            print("CONTINUING TRAINING FROM", continual_step*num_envs, "TIMESTEPS")
            continual_model = os.path.join(experiment_dir, scen, "agents", f"model_{continual_step}.zip")
            agent = PPO.load(continual_model, _init_setup_model=True, env=env, **PPO_hyperparams)
        print("DONE INITIALIZING AGENT")


        best_mean_reward = -np.inf
        n_steps =  continual_step 
        timesteps = scenarios[scen] - num_envs*continual_step
        print("\nTRAINING FOR", timesteps, "TIMESTEPS", "IN", scen.upper())

        success_threshold = scenario_success_threshold[scen] # Success rate threshold to move to the next scenario 
        callback = TensorboardLogger(agents_dir=agents_dir, 
                                     log_freq=PPO_hyperparams["n_steps"], 
                                     save_freq=10000, 
                                     success_buffer_size=k,
                                     n_cpu=args.n_cpu,
                                     success_threshold=success_threshold,
                                     use_success_as_stopping_criterion=True)
        
        agent.learn(total_timesteps=int(timesteps), reset_num_timesteps=False, tb_log_name="PPO", callback=callback, progress_bar=True)

        print("FINISHED TRAINING AGENT IN", scen.upper())
        save_path = os.path.join(agents_dir, "last_model.zip")
        agent.save(save_path)
        print("SAVE SUCCESSFUL")
        env.close()
        del env
        del agent
        print("ENVIRONMENT CLOSED\n")

        print(f"TRAINIG TIME IN SCENARIO {scen} TOOK {time.strftime('%H:%M:%S', time.gmtime(time.time() - _s))}")
        
        #Write total training time to file:
        try:
            with open(f'{experiment_dir}/training_time_raw.txt', 'r') as file:
                time_trained = float(file.read()) 
        except FileNotFoundError:
            with open(f'{experiment_dir}/training_time_raw.txt', 'w') as file:
                file.write(str(time_trained))
        time_trained += time.time() - _s
        with open(f'{experiment_dir}/training_time_raw.txt', 'w') as file:
            file.write(str(time_trained))

    # Convert the raw training time to hours, minutes and seconds
    with open(f'{experiment_dir}/training_time_raw.txt', 'r') as file:
        time_trained = float(file.read())
    with open(f'{experiment_dir}/training_time.txt', 'w') as file:
        file.write(f"WHOLE TRAINING TOOK {time.strftime('%H:%M:%S', time.gmtime(time_trained))}")
    print(f"WHOLE TRAINING TOOK {time.strftime('%H:%M:%S', time.gmtime(time_trained))}")
    print("TRAINING COMPLETE")