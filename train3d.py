import os
import json
import gymnasium as gym
import gym_quad
import numpy as np
import multiprocessing
from  multiprocessing.pool import Pool as pool
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
#NB this is a temporary fix to avoid the warning from pytorch3d #Need the mtl file if we want actual images.
warnings.filterwarnings("ignore", message="No mtl file provided", category=UserWarning, module="pytorch3d.io.obj_io")


###---###---### IMPORT THE DRL_CONFIG AND MODIFY IF NEEDED ###---###---###
from drl_config import lv_vae_config
train_config = lv_vae_config.copy()
train_config["max_t_steps"] = 8000
train_config["recap_chance"] = 0.1


###---###---### CHOOSE CURRICULUM SETUP HERE ###---###---###
# total_timesteps = 10e6 #15e6
# scenarios = {"line"         :   2.5e5, #Experimental result see Exp 4 on "Jørgen PC"
#              "3d_new"       :   2.5e5,
#              "intermediate" :   total_timesteps*0.2,
#              "proficient"   :   total_timesteps*0.3,
#              "expert"       :   total_timesteps*0.3}

# scenarios = {"3d_new"         :   2048,
#              "proficient"     :   2048,
#              "expert"     :   2048}
             

# #THIS CONFIG GAVE AN AGENT THAT MANAGED EVERYTHING EXCEPT DEADEND (4.27.24) BEFORE INTRODUCTION OF BOX 
#See exp dir 2 on jørgen pc or exp dir 2 in the good experiments folder. 
scenarios = {   "line"          :   1e5,
                "easy"          :   1e6,
                "proficient"    :   1e6,
                "intermediate"  :   1e6,
                "expert"        :   1e6
             }


#This config which is similar to the one above (see exp dir 3 on jørgen pc)
#except that intermediate and proficient is flipped (which according to their names would make more sense)
#Does NOT work. I think this comes from the intermediate scenario only having 1 obstacle resulting in less obsatcle information per run
#Which might lead the agent to focus on path following instead as this yields most rewards in this scenario.
# scenarios = {   "line"          :   2e5,
#                 "easy"          :   1e6,
#                 "intermediate"  :   1e6,
#                 "proficient"    :   1e6,
#                 "expert"        :   1e6
#              }

#I think that doing line-easy-proficient-expert with more time in easy proficient and expert is the way to go. 
#DID NOT WORK ACCORDING TO ONE TEST RUN See "Magnus PC" exp dir 6
# scenarios = {   "line"          :   2e5,
#                 "easy"          :   1.5e6,
#                 "proficient"    :   2e6,
#                 "expert"        :   2.5e6
#             }

#Newest config using the sameish setup as the one succesful one,
#but added easy_random which randomizes the position and attitude of the quad
# scenarios = {   "line"          :   1e5,
#                 "easy"          :   1e6,
#                 "easy_random"   :   1e6, #Randomized pos and att of quad in easy scenario
#                 "proficient"    :   1e6,
#                 "intermediate"  :   1e6,
#                 "expert"        :   1e6
#              }

#This was used and all noise was active the whole run for expid 8 on "Jøreng PC"
scenarios = {   "line"                 :  0.1e6,
                "easy"                 :  0.33e6,
                "easy_random"          :  0.33e6, #Randomized pos and att of quad in easy scenario 
                "intermediate"         :  1e6,
                "proficient"           :  1e6,
                "expert"               :  3e6,
                "proficient_perturbed" :  1e6,
                "expert_perturbed"     :  2e6
             }

#scenarios = {'house': 5e6} # Wont work....?


###---###---### SELECT PPO HYPERPARAMETERS HERE ###---###---###
'''From kulkarni paper:
The neural network is trained with an adaptive learning rate initialized at lr = 10−4. 
The discount factor is set to γ = 0.98. 
The neural network is trained with 1024 environments simulated in parallel with an average time step of 0.1s 
and rollout buffer size set to 32. 
We train this policy for approximately 26 × 10^6 environment steps aggregated over all agents.
'''
#TODO implement the above hyperparameters????
PPO_hyperparams = {
    'n_steps': 2048, 
    'batch_size': 128,
    'gae_lambda': 0.95,
    'gamma': 0.98, #old:0.99,
    'n_epochs': 10,
    'ent_coef': 0.001, 
    'verbose': 2,
    'device':'cuda', #Will be used for both feature extractor and PPO
    #'clip_range': 0.2,
    #'learning_rate': 2.5e-4, #10e-4, #2.5e-4,old # Try default (3e-4)
    #"optimizer_class":torch.optim.Adam, #Throws error (not hos Eirik :)) Now it does idk why sorry man
    #"optimizer_kwargs":{"lr": 10e-4}
}


###---###---### SELECT POLICYKWARGS HERE - FEATUREEXTRACTOR AND PPO NETWORK ACRHITECTURE ###---###---###

#VAE
# encoder_path = None #If you want to train the encoder from scratch
# encoder_path = f"{os.getcwd()}/VAE_encoders/encoder_conv1_experiment_3000_seed1.json"
# encoder_path = None
encoder_path = f"{os.getcwd()}/VAE_encoders/encoder_conv1_experiment_7_seed1.json"
lock_params = False #True if you want to lock the encoder parameters. False to let them be trained

#PPO
#From Ørjan:    net_arch = dict(pi=[128, 64, 32], vf=[128, 64, 32])
#SB3 default:   net_arch = dict(pi=[64, 64], vf=[64, 64])
#From Kulkarni: net_arch = dict(pi=[512, 256, 64], vf=[512, 256, 64]) #NB: GRU is not included in this probs overkill though
ppo_pi_vf_arch = dict(pi = [128,64,32], vf = [128,64,32]) #The PPO network architecture policy and value function

policy_kwargs = dict(
    features_extractor_class = PerceptionIMUDomainExtractor,
    features_extractor_kwargs = dict(img_size=train_config["compressed_depth_map_size"],
                                     features_dim=train_config["latent_dim"],
                                     device = PPO_hyperparams['device'],
                                     lock_params=lock_params,
                                     lock_params_conv = True,
                                     pretrained_encoder_path = encoder_path),
    net_arch = ppo_pi_vf_arch
)


"""
To train the agent, run the following command in terminal exchange x for the experiment id you want to train:
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
    _s = time.time() #For tracking training time

    print('\nTOTAL CPU CORE COUNT:', multiprocessing.cpu_count())
    experiment_dir, _, args = parse_experiment_info()
    scenario_list = list(scenarios.keys())
    done_training = False
    for i, scen in enumerate(scenario_list):

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
            continual_model = os.path.join(experiment_dir, scenario_list[i-1], "agents", f"model_{continual_step}.zip")
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

        agent.learn(total_timesteps=timesteps, 
                    tb_log_name="PPO",
                    callback=TensorboardLogger(agents_dir=agents_dir, log_freq=PPO_hyperparams["n_steps"], save_freq=10000),
                    progress_bar=True)
        
        print("FINISHED TRAINING AGENT IN", scen.upper())
        save_path = os.path.join(agents_dir, "last_model.zip")
        agent.save(save_path)
        print("SAVE SUCCESSFUL")
        env.close()
        del env
        del agent
        print("ENVIRONMENT CLOSED\n")        
    print(f"WHOLE TRAINING TOOK {time.strftime('%H:%M:%S', time.gmtime(time.time() - _s))}")
    #Saving of total training time.
    with open(f'{experiment_dir}/training_time.txt', 'w') as file:
        file.write(f"WHOLE TRAINING TOOK {time.strftime('%H:%M:%S', time.gmtime(time.time() - _s))}")
        file.write("\nSCENARIOS TRAINED IN:")
        for scen, steps in scenarios.items():
            file.write(f"\n{scen}: {steps} timesteps")