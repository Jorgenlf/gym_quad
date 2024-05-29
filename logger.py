from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
import numpy as np
import os


class TensorboardLogger(BaseCallback):
    '''
    A custom callback for tensorboard logging.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    
    To open tensorboard after/during training, run the following command in terminal:
    nonmesh:
    tensorboard --logdir 'log/LV_VAE-v0/Experiment x'
    mesh:
    tensorboard --logdir 'log/LV_VAE_MESH-v0/Experiment x'
    '''

    def __init__(self, agents_dir=None, verbose=0, log_freq=1024, save_freq=10000, success_buffer_size=5, n_cpu = 1 , success_threshold=0.8, use_success_as_stopping_criterion=False):
        super().__init__(verbose)
        self.agents_dir = agents_dir
        self.n_episodes = 0
        self.n_steps = 0
        self.n_calls = 0
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.success_buffer_size = success_buffer_size * n_cpu
        self.prev_successes = [0]*self.success_buffer_size
        self.success_threshold = success_threshold
        self.use_success_as_stopping_criterion = use_success_as_stopping_criterion
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

            #The np.mean covers the rare case of several agents being done at the same time
            
            #Average over all agents last time step (done by record_mean function)
            # Reward metrics
            # avg_agent_last_t_reward = np.mean([info["reward"] for info in infos])
            # avg_agent_last_t_length = np.mean([info["env_steps"] for info in infos])
            # avg_agent_last_t_collision_reward = np.mean([info["collision_reward"] for info in infos])
            # avg_agent_last_t_collision_avoidance_reward = np.mean([info["collision_avoidance_reward"] for info in infos])
            # avg_agent_last_t_path_adherence = np.mean([info["path_adherence"] for info in infos])
            # avg_agent_last_t_path_progression = np.mean([info["path_progression"] for info in infos])
            # avg_agent_last_t_reach_end_reward = np.mean([info['reach_end_reward'] for info in infos])
            # avg_agent_last_t_existence_reward = np.mean([info['existence_reward'] for info in infos])
            # avg_agent_last_t_approach_end_reward = np.mean([info['approach_end_reward'] for info in infos])
            # self.logger.record_mean("episodes/avg_ep_reward", avg_agent_last_t_reward)
            # self.logger.record_mean("episodes/avg_ep_length", avg_agent_last_t_length)
            # self.logger.record_mean("episodes/avg_ep_collision_reward", avg_agent_last_t_collision_reward)
            # self.logger.record_mean("episodes/avg_ep_collision_avoidance_reward", avg_agent_last_t_collision_avoidance_reward)
            # self.logger.record_mean("episodes/avg_ep_path_adherence_reward", avg_agent_last_t_path_adherence)
            # self.logger.record_mean("episodes/avg_ep_path_progression_reward", avg_agent_last_t_path_progression)
            # self.logger.record_mean("episodes/avg_ep_reach_end_reward", avg_agent_last_t_reach_end_reward)
            # self.logger.record_mean("episodes/avg_ep_existence_reward", avg_agent_last_t_existence_reward)
            # self.logger.record_mean("episodes/avg_ep_approach_end_reward", avg_agent_last_t_approach_end_reward)


            #Average over all agents last time step
            # Metrics for report plotting 
            # avg_agent_last_t_path_prog = np.mean([info["progression"] for info in infos])
            # avg_agent_last_t_time = np.mean([info["time"] for info in infos])
            # avg_agent_last_t_collision_rate = np.mean([info["collision_rate"] for info in infos])
            # avg_agent_last_t_total_path_deviance = np.mean([info["total_path_deviance"] for info in infos])
            # avg_agent_last_t_error_e = np.mean([info["errors"][0] for info in infos])
            # avg_agent_last_t_error_h = np.mean([info["errors"][1] for info in infos])

            # self.logger.record_mean("metrics_agent_avg/path_progression", avg_agent_last_t_path_prog)
            # self.logger.record_mean("metrics_agent_avg/time", avg_agent_last_t_time)
            # self.logger.record_mean("metrics_agent_avg/collision_rate", avg_agent_last_t_collision_rate)
            # self.logger.record_mean("metrics_agent_avg/total_path_deviance", avg_agent_last_t_total_path_deviance)
            # self.logger.record_mean("metrics_agent_avg/error_e", avg_agent_last_t_error_e)
            # self.logger.record_mean("metrics_agent_avg/error_h", avg_agent_last_t_error_h)

            #Average over episode and agents
            #Cumulative errors divided by the number of steps to get averages per episode
            #Reward metrics
            avg_episode_avg_agent_reward = np.mean([info["cumulative_reward"]/info["env_steps"] for info in infos])
            avg_agent_cum_reward = np.mean([info["cumulative_reward"] for info in infos])
            avg_episode_avg_agent_CA_reward = np.mean([info["cum_CA_rew"]/info["env_steps"] for info in infos])
            avg_episode_avg_agent_path_adherence = np.mean([info["cum_path_adherence_rew"]/info["env_steps"] for info in infos])
            avg_episode_avg_agent_path_progression = np.mean([info["cum_path_progression_rew"]/info["env_steps"] for info in infos])
            avg_episode_avg_agent_existence_reward = np.mean([info["cum_existence_rew"]/info["env_steps"] for info in infos])
            avg_episode_avg_agent_collision_reward = np.mean([info["cum_collision_rew"]/info["env_steps"] for info in infos])
            avg_episode_avg_agent_reach_end_reward = np.mean([info["cum_reach_end_rew"]/info["env_steps"] for info in infos])
            avg_episode_avg_agent_lambda_PA = np.mean([info["cum_lambda_PA"]/info["env_steps"] for info in infos]) 
            avg_episode_avg_agent_lambda_CA = np.mean([info["cum_lambda_CA"]/info["env_steps"] for info in infos])
            avg_episode_avg_agent_pass_wp_reward = np.mean([info["cum_pass_wp_rew"]/info["env_steps"] for info in infos])

            self.logger.record_mean("1_reward/a_avg_cum_reward", avg_agent_cum_reward)
            self.logger.record_mean("1_reward/ep_&_a_avg_reward", avg_episode_avg_agent_reward)
            self.logger.record_mean("1_reward/ep_&_a_avg_CA_reward", avg_episode_avg_agent_CA_reward)
            self.logger.record_mean("1_reward/ep_&_a_avg_path_adherence", avg_episode_avg_agent_path_adherence)
            self.logger.record_mean("1_reward/ep_&_a_avg_path_progression", avg_episode_avg_agent_path_progression)
            self.logger.record_mean("1_reward/ep_&_a_avg_existence_reward", avg_episode_avg_agent_existence_reward)
            self.logger.record_mean("1_reward/ep_&_a_avg_collision_reward", avg_episode_avg_agent_collision_reward)
            self.logger.record_mean("1_reward/ep_&_a_avg_reach_end_reward", avg_episode_avg_agent_reach_end_reward)
            self.logger.record_mean("1_reward/ep_&_a_avg_lambda_PA", avg_episode_avg_agent_lambda_PA)
            self.logger.record_mean("1_reward/ep_&_a_avg_lambda_CA", avg_episode_avg_agent_lambda_CA)
            self.logger.record_mean("1_reward/ep_&_a_avg_pass_wp_reward", avg_episode_avg_agent_pass_wp_reward)

            #Terminal metrics
            avg_agent_collision = np.mean([info["collision_rate"] for info in infos])
            avg_agent_timeout = np.mean([info["timeout"] for info in infos])
            avg_agent_min_rew_reached = np.mean([info["min_rew_reached"] for info in infos])
            avg_agent_success = np.mean([info["success"] for info in infos])
            self.logger.record_mean("2_terminal_metrics/collision_rate", avg_agent_collision)
            self.logger.record_mean("2_terminal_metrics/timeout_rate", avg_agent_timeout)
            self.logger.record_mean("2_terminal_metrics/min_rew_reached_rate", avg_agent_min_rew_reached)
            self.logger.record_mean("2_terminal_metrics/success_rate", avg_agent_success)

            #Using success rate as stopping criterion
            if self.use_success_as_stopping_criterion:
                if len(self.prev_successes) > self.success_buffer_size:
                    self.prev_successes.pop(0)
                self.prev_successes.append(avg_agent_success)
                print("Avg successes across last n_cpu agents times k: ", np.mean(self.prev_successes))
                if np.mean(self.prev_successes) > self.success_threshold:
                    return False #Stop training and let train3d.py save the model

            #Metrics for report plotting
            avg_episode_avg_agent_path_prog = np.mean([info["progression"] for info in infos])
            avg_episode_avg_agent_time = np.mean([info["time"] for info in infos])
            avg_episode_avg_agent_total_path_deviance = np.mean([info["cum_total_path_deviance"]/info["env_steps"] for info in infos])
            avg_episode_avg_agent_error_e = np.mean([info["cum_e_error"]/info["env_steps"] for info in infos])
            avg_episode_avg_agent_error_h = np.mean([info["cum_h_error"]/info["env_steps"] for info in infos])

            self.logger.record_mean("3_metrics/a_avg_path_progression", avg_episode_avg_agent_path_prog)
            self.logger.record_mean("3_metrics/a_avg_time", avg_episode_avg_agent_time)
            self.logger.record_mean("3_metrics/ep_&_a_avg_total_path_deviance", avg_episode_avg_agent_total_path_deviance)
            self.logger.record_mean("3_metrics/ep_&_a_avg_error_e", avg_episode_avg_agent_error_e)
            self.logger.record_mean("3_metrics/ep_&_a_avg_error_h", avg_episode_avg_agent_error_h)

            #Quadcopter state
            avg_ep_a_speed = np.mean([info["cum_speed"]/info["env_steps"] for info in infos])

            self.logger.record_mean("4_quadcopter_state/ep_&_a_avg_speed", avg_ep_a_speed)


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
    