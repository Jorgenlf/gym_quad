# from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
# import numpy as np
# import os


#----#-----# PER NOW THE TENSORBOARD LOGGER IS USED DIRECTLY IN THE TRAIN3D.PY FILE #-----#-----#
#TODO WHEN THE TRAIN3D.PY FILE IS CLEANED UP, THE TENSORBOARD LOGGER SHOULD BE MOVED TO THIS FILE
#REMOVE THE CLASSES BELOW AS THEYRE OUTDATED AS THE TENSORBOARD LOGGER IS USED DIRECTLY IN THE TRAIN3D.PY FILE
#----#-----# PER NOW THE TENSORBOARD LOGGER IS USED DIRECTLY IN THE TRAIN3D.PY FILE #-----#-----#


# #LOGGER NUMBER 1 FROM SPECIALIZATION PROJECT BASED ON THOMAS CODE
# class TensorboardLogger(BaseCallback):
#     """
#      A custom callback for tensorboard logging.

#     :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
#     """

#     def __init__(self, agents_dir=None, verbose=0,):
#         super().__init__(verbose)
#         #From tensorboard logger
#         self.n_episodes = 0
#         self.ep_reward = 0
#         self.ep_length = 0
#         #from stats callback
#         self.agents_dir = agents_dir
#         self.n_steps = 0
#         self.n_calls=0
#         self.prev_stats=None
#         self.ob_names=["u","v","w","roll","pitch","yaw","p","q","r","nu_c0","nu_c1","nu_c2","chi_err","upsilon_err","chi_err_1","upsilon_err_1","chi_err_5","upsilon_err_5"]
#         self.state_names=["x","y","z","roll","pitch","yaw","u","v","w","p","q","r"]
#         self.error_names=["e", "h"]

#     # Those variables will be accessible in the callback
#     # (they are defined in the base class)
#     # The RL model
#     # self.model = None  # type: BaseAlgorithm
#     # An alias for self.model.get_env(), the environment used for training
#     # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
#     # Number of time the callback was called
#     # self.n_calls = 0  # type: int
#     # self.num_timesteps = 0  # type: int
#     # local and global variables
#     # self.locals = None  # type: Dict[str, Any]
#     # self.globals = None  # type: Dict[str, Any]
#     # The logger object, used to report things in the terminal
#     # self.logger = None  # stable_baselines3.common.logger
#     # # Sometimes, for event callback, it is useful
#     # # to have access to the parent object
#     # self.parent = None  # type: Optional[BaseCallback]

#     def _on_training_start(self) -> None:
#         """
#         This method is called before the first rollout starts.
#         """
#         pass

#     def _on_rollout_start(self) -> None:
#         """
#         A rollout is the collection of environment interaction
#         using the current policy.
#         This event is triggered before collecting new samples.
#         """
#         pass

#     def _on_step(self) -> bool:
#         """
#         This method will be called by the model after each call to `env.step()`.

#         For child callback (of an `EventCallback`), this will be called
#         when the event is triggered.

#         :return: (bool) If the callback returns False, training is aborted early.
#         """
#         ###From tensorboard logger###
#         # Logging data at the end of an episode - must check if the environment is done
#         done_array = self.locals["dones"]
#         n_done = np.sum(done_array).item()

#         # Only log if any workers are actually at the end of an episode
#         ###From tensorboard logger end###

#         ###From stats callbacks###
#         stats  = {"reward_path_following": [],      #self.reward_path_following_sum, 
#                 "reward_collision_avoidance": [],   #self.reward_collision_avoidance_sum,
#                 "reward_collision": [],             #self.reward_collision,
#                 "obs":[],                           #self.past_obs,
#                 "states":[],                        #self.past_states,
#                 "errors":[]}                        #self.past_errors
#         ###From stats callback end###

#         if n_done > 0:
#             # Record the cumulative number of finished episodes
#             self.n_episodes += n_done
#             self.logger.record('time/episodes', self.n_episodes)

#             # Fetch data from the info dictionary in the environment (convert tuple->np.ndarray for easy indexing)
#             infos = np.array(self.locals["infos"])[done_array]

#             avg_reward = 0
#             avg_length = 0
#             avg_collision_reward = 0
#             avg_collision_avoidance_reward = 0
#             avg_path_adherence = 0
#             avg_path_progression = 0
#             avg_reach_end_reward = 0
#             avg_agressive_alpha_reward = 0
#             for info in infos:
#                 avg_reward += info["reward"]
#                 avg_length += info["env_steps"]
#                 avg_collision_reward += info["collision_reward"]
#                 avg_collision_avoidance_reward += info["collision_avoidance_reward"]
#                 avg_path_adherence += info["path_adherence"]
#                 avg_path_progression += info["path_progression"]
#                 avg_reach_end_reward += info['reach_end_reward'] 

#                 stats["reward_path_following"].append(info["reward_path_following"])
#                 stats["reward_collision_avoidance"].append(info["reward_collision_avoidance"])
#                 stats["reward_collision"].append(info["reward_collision"])
#                 stats["obs"].append(info["obs"])
#                 stats["states"].append(info["states"])
#                 stats["errors"].append(info["errors"])


#             avg_reward /= n_done
#             avg_length /= n_done
#             avg_collision_reward /= n_done
#             avg_collision_avoidance_reward /= n_done
#             avg_path_adherence /= n_done
#             avg_path_progression /= n_done
#             avg_reach_end_reward /= n_done

#             # Write to the tensorboard logger
#             self.logger.record("episodes/avg_reward", avg_reward)
#             self.logger.record("episodes/avg_length", avg_length)
#             self.logger.record("episodes/avg_collision_reward", avg_collision_reward)
#             self.logger.record("episodes/avg_collision_avoidance_reward", avg_collision_avoidance_reward)
#             self.logger.record("episodes/avg_path_adherence_reward", avg_path_adherence)
#             self.logger.record("episodes/avg_path_progression_reward", avg_path_progression)
#             self.logger.record("episodes/avg_reach_end_reward", avg_reach_end_reward)
#             self.logger.record("episodes/avg_agressive_alpha_reward", avg_agressive_alpha_reward)

#             #From stats callback
#             for i in range(len(done_array)): 
#                 if done_array[i]:
#                     if self.prev_stats is not None:
#                         for stat in self.prev_stats[i].keys():
#                             self.logger.record('stats/' + stat, self.prev_stats[i][stat])
#                     # for stat in stats[i].keys():
#                     #     self.logger.record('stats/' + stat, stats[i][stat])
        
#         #From stats callback
#         self.prev_stats = stats

#         if (n_steps + 1) % 10000 == 0:
#             _self = self.locals.get("self")
#             _self.save(os.path.join(self.agents_dir, "model_" + str(n_steps+1) + ".zip"))
#         n_steps += 1
#         return True

#     def _on_rollout_end(self) -> None:
#         """
#         This event is triggered before updating the policy.
#         """

#     def _on_training_end(self) -> None:
#         """
#         This event is triggered before exiting the `learn()` method.
#         """
#         pass

# #OLD LOGGER NUBMER 2 FROM TRAIN3D.PY HERE
# # class StatsCallback(BaseCallback):
# #     def __init__(self):
# #         self.n_steps = 0
# #         self.n_calls=0
# #         self.prev_stats=None
# #         self.ob_names=["u","v","w","roll","pitch","yaw","p","q","r","nu_c0","nu_c1","nu_c2","chi_err","upsilon_err","chi_err_1","upsilon_err_1","chi_err_5","upsilon_err_5"]
# #         self.state_names=["x","y","z","roll","pitch","yaw","u","v","w","p","q","r"]
# #         self.error_names=["e", "h"]

# #     def _on_step(self):
# #         done_array = np.array(self.locals.get("dones") if self.locals.get("dones") is not None else self.locals.get("dones"))
# #         stats = self.locals.get("self").get_env().env_method("get_stats") 
# #         global n_steps
        
# #         for i in range(len(done_array)):
# #             if done_array[i]:
# #                 if self.prev_stats is not None:
# #                     for stat in self.prev_stats[i].keys():
# #                         self.logger.record('stats/' + stat, self.prev_stats[i][stat])
# #                 # for stat in stats[i].keys():
# #                 #     self.logger.record('stats/' + stat, stats[i][stat])
# #         self.prev_stats = stats

# #         # print("\nstats:", stats)
# #         # print("prev_stats:", self.prev_stats)

# #         if (n_steps + 1) % 10000 == 0:
# #             _self = self.locals.get("self")
# #             _self.save(os.path.join(agents_dir, "model_" + str(n_steps+1) + ".zip"))
# #         n_steps += 1
# #         return True
