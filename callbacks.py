'''
Callback functions, e.g., for SB3 DRL training
'''
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class MyCallback(BaseCallback):
    """
     A custom callback for tensorboard logging.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        #self.ep_reward = 0
        self.ep_length = 0
        #self.ep_distance = 0
        #self.reached_goal = False
        self.n_episodes = 0
        self.colav_reward = 0
        self.pathadherence_reward = 0
        self.pathprogression_reward = 0
        self.collision_reward = 0
        self.reachend_reward = 0
        self.existence_reward = 0
        self.total_reward = 0


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

        info = self.locals["infos"][0]


        #self.ep_distance = self.locals["infos"][0]["d_g"]  # Distance to goal
        #self.reached_goal = self.locals["infos"][0]["on_goal"]
        self.colav_reward += info["collision_avoidance_reward"]
        self.pathadherence_reward += info["path_adherence"]
        self.pathprogression_reward += info["path_progression"]
        self.collision_reward += info["collision_reward"]
        self.reachend_reward += info["reach_end_reward"]
        self.existence_reward += info["existence_reward"]
        self.total_reward += info["reward"]
    
        #self.ep_reward += info["reward"]
        self.ep_length += 1

        # TODO: Check done-arrays to log episode data only at the end of the episode
        # Logging data at the end of an episode - must check if the environment is done
        done_array = self.locals["dones"]
        n_done = np.sum(done_array).item()

        # Only log if any workers are actually at the end of an episode
        if n_done > 0:
            # Record the cumulative number of finished episodes
            self.n_episodes += n_done
            self.logger.record('time/episodes', self.n_episodes)

            # Fetch data from the info dictionary in the environment (convert tuple->np.ndarray for easy indexing)
            infos = np.array(self.locals["infos"])[done_array]

            # Determine the keys to log to tensorboard
            keys = infos[0].keys()
            n_keys = len(keys)# - 3  # Ignore two last entries of infos (TimeLimit.truncated, terminal_observation)

            #"""# Calculate averages
            n_envs = np.sum(done_array)  #
            avgs = np.zeros(shape=(n_keys,))  # numpy version
            for _dict in infos:
                values = [v for v in _dict.values() if type(v) != dict and type(v) != np.ndarray]

                # ppo_items = -2, sac_items = -3
                avgs += np.array(values) / n_envs
                
            # Write to the tensorboard logger
            for key, avg in zip(keys, avgs):
                self.logger.record(f"infos/{key}", avg)
                #"""
            """# Just using the first env:
            # Write to the tensorboard logger
            for key, value in infos[0].items():
                if type(value) != dict and type(value) != np.ndarray:
                    self.logger.record(f"infos/{key}", value)

                elif type(value) == dict:
                    for _k, _v in value.items():
                        self.logger.record(f"{key}/{_k}", _v)"""

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        #self.logger.record("rollout/ep_reward", self.ep_reward)
        self.logger.record("rollout/ep_length", self.ep_length)
        #self.logger.record("rollout/ep_distance", self.ep_distance)
        #self.logger.record("rollout/reached_goal", self.reached_goal)
        self.logger.record("rollout/colav_reward", self.colav_reward)
        self.logger.record("rollout/pathadherence_reward", self.pathadherence_reward)
        self.logger.record("rollout/pathprogression_reward", self.pathprogression_reward)
        self.logger.record("rollout/collision_reward", self.collision_reward)
        self.logger.record("rollout/reachend_reward", self.reachend_reward)
        self.logger.record("rollout/existence_reward", self.existence_reward)
        self.logger.record("rollout/total_reward", self.total_reward)


        #self.ep_reward = 0
        self.ep_length = 0
        #self.ep_distance = 0
        #self.reached_goal = False
        self.colav_reward = 0
        self.pathadherence_reward = 0
        self.pathprogression_reward = 0
        self.collision_reward = 0
        self.reachend_reward = 0
        self.existence_reward = 0
        self.total_reward = 0


    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass