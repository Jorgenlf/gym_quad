from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
import numpy as np
import os



class TensorboardLogger(BaseCallback):
    '''
    A custom callback for tensorboard logging.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    
    To open tensorboard after/during training, run the following command in terminal:
    tensorboard --logdir 'log/LV_VAE-v0/Experiment x'
    '''

    def __init__(self, agents_dir=None, verbose=0, log_freq=1024, save_freq=10000):
        super().__init__(verbose)
        self.agents_dir = agents_dir
        self.n_episodes = 0
        self.n_steps = 0
        self.n_calls = 0
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.state_names = ["x", "y", "z", "roll", "pitch", "yaw", "u", "v", "w", "p", "q", "r"]
        self.error_names = ["e", "h"]


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        done_array = self.locals["dones"]
        n_done = np.sum(done_array).item()

        if n_done > 0:
            self.n_episodes += n_done
            self.logger.record('time/episodes', self.n_episodes)
            infos = np.array(self.locals["infos"])[done_array]

            avg_reward = np.mean([info["reward"] for info in infos])
            avg_length = np.mean([info["env_steps"] for info in infos])
            avg_collision_reward = np.mean([info["collision_reward"] for info in infos])
            avg_collision_avoidance_reward = np.mean([info["collision_avoidance_reward"] for info in infos])
            avg_path_adherence = np.mean([info["path_adherence"] for info in infos])
            avg_path_progression = np.mean([info["path_progression"] for info in infos])
            avg_reach_end_reward = np.mean([info['reach_end_reward'] for info in infos])
            avg_existence_reward = np.mean([info['existence_reward'] for info in infos])

            self.logger.record("episodes/avg_ep_reward", avg_reward)
            self.logger.record("episodes/avg_ep_length", avg_length)
            self.logger.record("episodes/avg_ep_collision_reward", avg_collision_reward)
            self.logger.record("episodes/avg_ep_collision_avoidance_reward", avg_collision_avoidance_reward)
            self.logger.record("episodes/avg_ep_path_adherence_reward", avg_path_adherence)
            self.logger.record("episodes/avg_ep_path_progression_reward", avg_path_progression)
            self.logger.record("episodes/avg_ep_reach_end_reward", avg_reach_end_reward)
            self.logger.record("episodes/avg_ep_existence_reward", avg_existence_reward)
        
        
        if self.n_steps % self.log_freq == 0:
            infos = self.locals["infos"]
            reward = np.mean([info["reward"] for info in infos])
            length = np.mean([info["env_steps"] for info in infos])
            collision_reward = np.mean([info["collision_reward"] for info in infos])
            collision_avoidance_reward = np.mean([info["collision_avoidance_reward"] for info in infos])
            path_adherence = np.mean([info["path_adherence"] for info in infos])
            path_progression = np.mean([info["path_progression"] for info in infos])
            reach_end_reward = np.mean([info["reach_end_reward"] for info in infos])
            existence_reward = np.mean([info["existence_reward"] for info in infos])

            self.logger.record("iter/reward", reward)
            self.logger.record("iter/length", length)
            self.logger.record("iter/collision_reward", collision_reward)
            self.logger.record("iter/collision_avoidance_reward", collision_avoidance_reward)
            self.logger.record("iter/path_adherence", path_adherence)
            self.logger.record("iter/path_progression", path_progression)
            self.logger.record("iter/reach_end_reward", reach_end_reward)
            self.logger.record("iter/existence_reward", existence_reward)
    

        # Check for model saving frequency
        if self.n_calls % self.save_freq == 0:
            self.model.save(os.path.join(self.agents_dir, "model_" + str(self.n_calls) + ".zip"))

        return True
    





# #-----#------#-----#Temp fix to make the global n_steps variable work pasting the tensorboardlogger class here#-----#------#-----#
#  class TensorboardLogger(BaseCallback):
#     '''
#      A custom callback for tensorboard logging.

#     :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    
#     To open tensorboard after/during training, run the following command in terminal:
#     tensorboard --logdir 'log/LV_VAE-v0/Experiment x'
#     '''

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
#         self.state_names=["x","y","z","roll","pitch","yaw","u","v","w","p","q","r"]
#         self.error_names=["e", "h"]

#     ''' info about the callback class
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
#     '''

#     def _on_training_start(self) -> None:
#         """
#         This method is called before the first rollout starts.
#         """
#         pass

#     # def _on_training_start(self):
#     #     self._log_freq = 1000  # log every 1000 calls

#     #     output_formats = self.logger.output_formats
#     #     # Save reference to tensorboard formatter object
#     #     # note: the failure case (not formatter found) is not handled here, should be done with try/except.
#     #     self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

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
#         # Logging data at the end of an episode - must check if the environment is done
#         done_array = self.locals["dones"]
#         n_done = np.sum(done_array).item()
#         # Only log if any workers are actually at the end of an episode
    

#         global n_steps
#         ###From stats callback end###

#         if n_done > 0:
#             # Record the cumulative number of finished episodes
#             self.n_episodes += n_done
#             self.logger.record('time/episodes', self.n_episodes)

#             # Fetch data from the info dictionary of the environments that have reached a done condition (convert tuple->np.ndarray for easy indexing)
#             infos = np.array(self.locals["infos"])[done_array]

#             avg_reward = 0
#             avg_length = 0
#             avg_collision_reward = 0
#             avg_collision_avoidance_reward = 0
#             avg_path_adherence = 0
#             avg_path_progression = 0
#             avg_reach_end_reward = 0
#             avg_existence_reward = 0
#             for info in infos:
#                 avg_reward += info["reward"]
#                 avg_length += info["env_steps"]
#                 avg_collision_reward += info["collision_reward"]
#                 avg_collision_avoidance_reward += info["collision_avoidance_reward"]
#                 avg_path_adherence += info["path_adherence"]
#                 avg_path_progression += info["path_progression"]
#                 avg_reach_end_reward += info['reach_end_reward'] 
#                 avg_existence_reward += info['existence_reward']

#             avg_reward /= n_done
#             avg_length /= n_done
#             avg_collision_reward /= n_done
#             avg_collision_avoidance_reward /= n_done
#             avg_path_adherence /= n_done
#             avg_path_progression /= n_done
#             avg_reach_end_reward /= n_done
#             avg_existence_reward /= n_done

#             # Write to the tensorboard logger
#             self.logger.record("episodes/avg_reward", avg_reward)
#             self.logger.record("episodes/avg_length", avg_length)
#             self.logger.record("episodes/avg_collision_reward", avg_collision_reward)
#             self.logger.record("episodes/avg_collision_avoidance_reward", avg_collision_avoidance_reward)
#             self.logger.record("episodes/avg_path_adherence_reward", avg_path_adherence)
#             self.logger.record("episodes/avg_path_progression_reward", avg_path_progression)
#             self.logger.record("episodes/avg_reach_end_reward", avg_reach_end_reward)
#             self.logger.record("episodes/avg_existence_reward", avg_existence_reward)

#             #Can log error and state here if wanted

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
# #-----#------#-----#Temp fix to make the global n_steps variable work pasting the tensorboardlogger class above#-----#------#-----#
# #TODO make it work without a global variable please