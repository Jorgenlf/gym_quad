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

            # Standard tensorboard plotting
            avg_reward = np.mean([info["reward"] for info in infos])
            avg_length = np.mean([info["env_steps"] for info in infos])
            avg_collision_reward = np.mean([info["collision_reward"] for info in infos])
            avg_collision_avoidance_reward = np.mean([info["collision_avoidance_reward"] for info in infos])
            avg_path_adherence = np.mean([info["path_adherence"] for info in infos])
            avg_path_progression = np.mean([info["path_progression"] for info in infos])
            avg_reach_end_reward = np.mean([info['reach_end_reward'] for info in infos])
            avg_existence_reward = np.mean([info['existence_reward'] for info in infos])
            avg_approach_end_reward = np.mean([info['approach_end_reward'] for info in infos])

            # Metrics for report plotting
            avg_path_prog = np.mean([info["progression"] for info in infos])
            avg_time = np.mean([info["time"] for info in infos])
            avg_collision_rate = np.mean([info["collision_rate"] for info in infos])
            avg_total_path_deviance = np.mean([info["total_path_deviance"] for info in infos])
            avg_error_e = np.mean([info["errors"][0] for info in infos])
            avg_error_h = np.mean([info["errors"][1] for info in infos])
            
            # Log into two different folders
            self.logger.record("episodes/avg_ep_reward", avg_reward)
            self.logger.record("episodes/avg_ep_length", avg_length)
            self.logger.record("episodes/avg_ep_collision_reward", avg_collision_reward)
            self.logger.record("episodes/avg_ep_collision_avoidance_reward", avg_collision_avoidance_reward)
            self.logger.record("episodes/avg_ep_path_adherence_reward", avg_path_adherence)
            self.logger.record("episodes/avg_ep_path_progression_reward", avg_path_progression)
            self.logger.record("episodes/avg_ep_reach_end_reward", avg_reach_end_reward)
            self.logger.record("episodes/avg_ep_existence_reward", avg_existence_reward)

            self.logger.record("metrics/avg_path_progression", avg_path_prog)
            self.logger.record("metrics/avg_time", avg_time)
            self.logger.record("metrics/avg_collision_rate", avg_collision_rate)
            self.logger.record("metrics/avg_total_path_deviance", avg_total_path_deviance)
            self.logger.record("metrics/avg_error_e", avg_error_e)
            self.logger.record("metrics/avg_error_h", avg_error_h)


            self.logger.record("episodes/avg_ep_approach_end_reward", avg_approach_end_reward)

            # test this setup
            # This setup does for some reason only send to tensorbord every n_spes*n_cpus steps, we try avergaing but that just gives the last value (i.e. just for one agent(????))
            # since .record only uses last value when called multiple times (as in mult cpu training), we need to use .record_mean to get actual
            self.logger.record_mean("metrics_2/avg_path_progression", avg_path_prog)
            self.logger.record_mean("metrics_2/avg_time", avg_time)
            self.logger.record_mean("metrics_2/avg_collision_rate", avg_collision_rate)
            self.logger.record_mean("metrics_2/avg_total_path_deviance", avg_total_path_deviance)
            self.logger.record_mean("metrics_2/avg_error_e", avg_error_e)
            self.logger.record_mean("metrics_2/avg_error_h", avg_error_h)
        
        
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
            approach_end_reward = np.mean([info["approach_end_reward"] for info in infos])

            self.logger.record("iter/reward", reward)
            self.logger.record("iter/length", length)
            self.logger.record("iter/collision_reward", collision_reward)
            self.logger.record("iter/collision_avoidance_reward", collision_avoidance_reward)
            self.logger.record("iter/path_adherence", path_adherence)
            self.logger.record("iter/path_progression", path_progression)
            self.logger.record("iter/reach_end_reward", reach_end_reward)
            self.logger.record("iter/existence_reward", existence_reward)
            self.logger.record("iter/approach_end_reward", approach_end_reward)
    

        # Check for model saving frequency
        if self.n_calls % self.save_freq == 0:
            self.model.save(os.path.join(self.agents_dir, "model_" + str(self.n_calls) + ".zip"))

        return True
    