import argparse
import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import pandas as pd
from cycler import cycler
from stable_baselines3 import PPO


def parse_experiment_info():
    """Parser for the flags that can be passed with the run/train/test scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="LV_VAE_MESH-v0", type=str, help="Which environment to run/train/test")
    parser.add_argument("--n_cpu", default=2, type=int, help="Number of CPUs to use")
    parser.add_argument("--exp_id", default = 1, type=int, help="Which experiment number to run/train/test")
    parser.add_argument("--run_scenario", default="line", type=str, help="Which scenario to run")
    parser.add_argument("--trained_scenario", default=None, type=str, help="Which scenario the agent was trained in")
    parser.add_argument("--agent", default=None, type=int, help="Which agent/model to load as main controller. Requires only integer. If None, the last model will be loaded.")
    parser.add_argument("--episodes", default=1, type=int, help="How many episodes to run when testing the quadcopter")
    parser.add_argument("--manual_control", default=False, type=bool, help="Whether to use manual control or not")
    parser.add_argument("--RT_vis", default=False, type=bool, help="Whether to visualize in realtime training or not")
    parser.add_argument("--save_depth_maps", default=False, type=bool, help="Whether to save depth maps or not")
    parser.add_argument("--test_list", nargs='+', default=["horizontal", "vertical"], help="List of test scenarios to run for generating results. Default is horizontal and vertical.")
    parser.add_argument("--trained_list", nargs='+', default=["expert"], help="List of trained scenarios to run for generating results. Default is expert.")
    parser.add_argument("--test_all_agents", default=False, type=bool, help="Whether to test all agents or not")
    args = parser.parse_args()


    experiment_dir = os.path.join(r"./log", r"{}".format(args.env), r"Experiment {}".format(args.exp_id))

    if args.trained_scenario is not None:
        agent_path = os.path.join(experiment_dir, args.trained_scenario, "agents")
    else:
        agent_path = os.path.join(experiment_dir, args.run_scenario, "agents")
    
    if args.agent is not None:
        agent_path = os.path.join(agent_path, "model_" + str(args.agent) + ".zip")
    else:
        agent_path = os.path.join(agent_path,"last_model.zip")
    
    return experiment_dir, agent_path, args


def calculate_IAE(sim_df):
    """
    Calculates and prints the integral absolute error provided an environment id and simulation data
    """
    IAE_cross = sim_df[r"$e$"].abs().sum()
    IAE_vertical = sim_df[r"$h$"].abs().sum()
    return IAE_cross, IAE_vertical


def simulate_environment(episode, env, agent: PPO, test_dir, sdm=False):
    """
    Input 
        episode:    episode number to run
        env:        environment to run
        agent:      agent to run
        test_dir:   directory to save the simulation data
        sdm:        save depth maps flag

    Output
        df:         pandas DataFrame with simulation data
        env:        environment after simulation

    Simulates an environment for with the provided agent and returns the simulation data as a pandas DataFrame
    """

    state_labels = [r"$X$", r"$Y$", r"$Z$", r"$\phi$", r"$\theta$", r"$\psi$", r"$u$", r"$v$", r"$w$", r"$p$", r"$q$", r"$r$"]
    action_labels = [r"$\v_{cmd}$", r"$\incline_{cmd}$",r"$\r_{cmd}$"]
    error_labels = [r"$e$",r"$h$"]
    observation_labels = [r"$\dot{u}^b$",r"$\dot{v}^b$",r"$\dot{w}^b$",\
                          r"$p_o$",r"$q_o$",r"$r_o$",\
                          r"$\chi_e$", r"$\upsilon_e$",\
                          r"$x_{cpp}^b$", r"$y_{cpp}^b$", r"$z_{cpp}^b$",\
                          r"$\upsilon_{cpp}^b$", r"$\chi_{cpp}^b$",\
                          r"$d_{nwp}$",r"$d_{end}$",\
                          r"$la_{x}$", r"$la_{y}$", r"$la_{z}$",
                          r"$a_0$", r"$a_1$", r"$a_2$",\
                        ]
    
    done = False
    obs = env.reset()[0]
    total_t_steps = 0
    time = []
    progression = []
    reward = []

    past_states = []
    past_actions = []
    errors = []
    pure_observations = []
    normed_domain_observations = []

    while not done: 
        action = agent.predict(obs, deterministic=True)[0]
        obs, _, done, _, info = env.step(action)

        if sdm:
            save_depth_maps(env, test_dir)  #Now this saves depthmaps online per timestep when obstacle is close, might be better to save up all then save all at once
        
        total_t_steps = info['env_steps']
        progression.append(info['progression'])
        time.append(info['time'])
        reward.append(info['reward'])
        past_states.append(info['state'])
        errors.append(info['errors'])
        past_actions.append(action)
        pure_observations.append(info['pure_obs'])
        normed_domain_observations.append(info['domain_obs'])

    episode = np.full(((total_t_steps,1)), episode)
    time = np.array(time).reshape((total_t_steps,1))
    progression = np.array(progression).reshape((total_t_steps,1))
    reward = np.array(reward).reshape((total_t_steps,1))
    past_actions = np.array(past_actions).reshape((total_t_steps,3))
    errors = np.array(errors).reshape((total_t_steps,2))
    pure_observations = np.array(pure_observations)
    normed_domain_observations = np.array(normed_domain_observations)

    labels = np.hstack(["Episode",  "Reward", "Time", "Progression", state_labels, error_labels, action_labels, observation_labels])
    sim_data = np.hstack([episode, reward, time, progression,     past_states,  errors,       past_actions,  pure_observations, normed_domain_observations])
    
    #make labels of obs to be obs0 obs1 depending on how many obs are in normed_domain_observations
    #Consult the env to correlate what obs0 obs1 means
    normed_obs_labels = []
    for i in range(normed_domain_observations.shape[1]):
        normed_obs_labels = np.hstack([normed_obs_labels, f"obs{i}"])
    labels = np.hstack([labels, normed_obs_labels])

    #Check that the length of each label matches the length of the info[] data
    assert len(state_labels) == len(info['state']), f"Length mismatch between state labels{len(state_labels)} and state data{len(info['state'])}"
    assert len(action_labels) == len(info['action']), f"Length mismatch between action labels{len(action_labels)} and action data{len(info['action'])}"
    assert len(error_labels) == len(info['errors']), f"Length mismatch between error labels{len(error_labels)} and error data{len(info['errors'])}"
    assert len(observation_labels) == len(info['pure_obs']), f"Length mismatch between pure observation labels{len(observation_labels)} and pure observation data{len(info['pure_obs'])}"
    assert len(normed_obs_labels) == len(info['domain_obs']), f"Length mismatch between normed observation labels{len(normed_obs_labels)} and normed observation data{len(info['domain_obs'])}"

    df = pd.DataFrame(sim_data, columns=labels)
    return df, env

#saving depth maps
def save_depth_maps(env, test_dir):
    path = os.path.join(test_dir, "depth_maps")
    if not os.path.exists(path):
        os.mkdir(path)

    empty_depth_map = torch.zeros((env.unwrapped.depth_map_size[0], env.unwrapped.depth_map_size[1]), dtype=torch.float32, device=env.unwrapped.device)

    if env.unwrapped.closest_measurement < env.unwrapped.max_depth: #Only save depth maps if there is a nearby obstacle else we get a large amount of empty depth maps
        
        if not torch.equal(env.unwrapped.noisy_depth_map, empty_depth_map):
            env.unwrapped.renderer.save_depth_map(f"{path}/noisy_depth_map_{env.unwrapped.total_t_steps}", env.unwrapped.noisy_depth_map)
        if not torch.equal(env.unwrapped.depth_map, empty_depth_map):
            env.unwrapped.renderer.save_depth_map(f"{path}/depth_map_{env.unwrapped.total_t_steps}", env.unwrapped.depth_map)

    else:
        pass

        #Comment/uncomment if you want to save the empty depth maps from envs without obstacles
        # depth=env.depth_map
        # path = f"{path}/depth_map_{env.total_t_steps}"
        # plt.style.use('ggplot')
        # plt.rc('font', family='serif')
        # plt.rc('xtick', labelsize=12)
        # plt.rc('ytick', labelsize=12)
        # plt.rc('axes', labelsize=12)

        # plt.figure(figsize=(8, 6))
        # plt.imshow(depth.cpu().numpy(), cmap="magma")
        # plt.clim(0.0, env.max_depth)
        # plt.colorbar(label="Depth [m]", aspect=30, orientation="vertical", fraction=0.0235, pad=0.04)
        # plt.axis("off")
        # plt.savefig(path, bbox_inches='tight')
        # plt.close()

#PLOTTING UTILITY FUNCTIONS#

#TODO add another layer of folders based on the episode number (when we start running multiple episodes)
def create_plot_folders(test_dir):
    """Creates the folders to save the plots in"""
    
    #The main folder
    if not os.path.exists(os.path.join(test_dir, "plots")):
        os.makedirs(os.path.join(test_dir, "plots"))
    
    #The observations
    if not os.path.exists(os.path.join(test_dir, "plots/observations")):
        os.makedirs(os.path.join(test_dir, "plots/observations"))
    
    #The normalized observations
    if not os.path.exists(os.path.join(test_dir, "plots/observations/normalized")):
        os.makedirs(os.path.join(test_dir, "plots/observations/normalized"))
    
    #The pure observations
    if not os.path.exists(os.path.join(test_dir, "plots/observations/pure")):
        os.makedirs(os.path.join(test_dir, "plots/observations/pure"))
    
    #The states
    if not os.path.exists(os.path.join(test_dir, "plots/states")):
        os.makedirs(os.path.join(test_dir, "plots/states"))


def set_default_plot_rc():
    """Sets the style for the plots report-ready"""
    colors = (cycler(color= ['#EE6666', '#3388BB', '#88DD89', '#EECC55', '#88BB44', '#FFBBBB']) +
                cycler(linestyle=['-',       '-',      '-',     '--',      ':',       '-.']))
    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('axes', facecolor='#ffffff', edgecolor='black',
        axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('grid', color='gray', linestyle='--')
    plt.rc('xtick', direction='out', color='black', labelsize=14)
    plt.rc('ytick', direction='out', color='black', labelsize=14)
    plt.rc('patch', edgecolor='#ffffff')
    plt.rc('lines', linewidth=4)

#OBSERVATION PLOTTING# 
def plot_all_normed_domain_observations(sim_df,test_dir):
    """Plots all normalized domain observations"""
    set_default_plot_rc()
    #Find largest observation index to use as range
    labels = sim_df.columns
    obs_labels = [label for label in labels if "obs" in label]
    range_obs = len(obs_labels)
    try:
        for i in range(0, range_obs):
            ax = sim_df.plot(x="Time", y=f"obs{i}", kind="line")
            ax.set_xlabel(xlabel="Time [s]", fontsize=14)
            ax.set_ylabel(ylabel="Normalized Observation", fontsize=14)
            ax.legend(loc="lower right", fontsize=14)
            ax.set_ylim([-1.25,1.25])
            plt.savefig(os.path.join(test_dir, "plots/observations/normalized", f"normed_obs{i}_episode{int(sim_df['Episode'].iloc[0])}.pdf"))
    except KeyError:
        print("Keyerror or All obs plotted or no normalized domain observations to plot")

def plot_observation_body_accl(sim_df,test_dir):
    """Plots body frame acceleration from the observation"""
    set_default_plot_rc()
    ax = sim_df.plot(x="Time", y=[r"$\dot{u}^b$",r"$\dot{v}^b$", r"$\dot{w}^b$"], kind="line")
    ax.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax.set_ylabel(ylabel="Acceleration [m/s^2]", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-1.25,1.25])
    plt.savefig(os.path.join(test_dir, "plots/observations/pure", f"body_accl_episode{int(sim_df['Episode'].iloc[0])}.pdf"))

def plot_observation_body_angvel(sim_df,test_dir):
    """Plots body frame angular velocity from the observation"""
    set_default_plot_rc()
    ax = sim_df.plot(x="Time", y=[r"$p_o$",r"$q_o$", r"$r_o$"], kind="line")
    ax.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax.set_ylabel(ylabel="Angular Velocity [rad/s]", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-1,1])
    plt.savefig(os.path.join(test_dir, "plots/observations/pure", f"body_angvel_episode{int(sim_df['Episode'].iloc[0])}.pdf"))

def plot_observation_cpp(sim_df,test_dir):
    """Plots the closest point on path in body frame from the observation"""
    set_default_plot_rc()
    ax = sim_df.plot(x="Time", y=[r"$x_{cpp}^b$",r"$y_{cpp}^b$", r"$z_{cpp}^b$"], kind="line")
    ax.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax.set_ylabel(ylabel="Position [m]", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-1.25,1.25])
    plt.savefig(os.path.join(test_dir, "plots/observations/pure", f"cpp_episode{int(sim_df['Episode'].iloc[0])}.pdf"))

def plot_observation_cpp_azi_ele(sim_df,test_dir):
    """Plots the aziumuth (chi) and elevation (upslion) of the closest point on path in body frame from the observation in degrees"""
    set_default_plot_rc()
    sim_df[r"$\chi_{cpp}^b$"] = np.rad2deg(sim_df[r"$\chi_{cpp}^b$"])
    sim_df[r"$\upsilon_{cpp}^b$"] = np.rad2deg(sim_df[r"$\upsilon_{cpp}^b$"])
    ax = sim_df.plot(x="Time", y=[r"$\chi_{cpp}^b$",r"$\upsilon_{cpp}^b$"], kind="line")
    ax.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax.set_ylabel(ylabel="Direction from body x to CPP [deg]", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-180,180])
    plt.savefig(os.path.join(test_dir, "plots/observations/pure", f"cpp_azi_ele_episode{int(sim_df['Episode'].iloc[0])}.pdf"))

def plot_observation_e_azi_ele(sim_df,test_dir):
    """Plots the aziumuth (chi) and elevation (upslion) error between lookahead vector and velocity vector in world (i think) from the observation"""
    set_default_plot_rc()
    sim_df[r"$\chi_e$"] = np.rad2deg(sim_df[r"$\chi_e$"])
    sim_df[r"$\upsilon_e$"] = np.rad2deg(sim_df[r"$\upsilon_e$"])
    ax = sim_df.plot(x="Time", y=[r"$\chi_e$", r"$\upsilon_e$"], kind="line")
    ax.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax.set_ylabel(ylabel="Error between velocity vec and LA [Deg]", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-180,180])
    plt.savefig(os.path.join(test_dir, "plots/observations/pure", f"e_azi_ele_episode{int(sim_df['Episode'].iloc[0])}.pdf"))

def plot_observation_dists(sim_df,test_dir):
    """Plots the distance to the next waypoint and the distance to the end of the path from the observation"""
    set_default_plot_rc()
    ax = sim_df.plot(x="Time", y=[r"$d_{nwp}$", r"$d_{end}$"], kind="line")
    ax.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax.set_ylabel(ylabel="Distance [m]", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([0,100])
    plt.savefig(os.path.join(test_dir, "plots/observations/pure", f"d_wp_d_end_episode{int(sim_df['Episode'].iloc[0])}.pdf"))

#PER NOW THE BODY VELOCITIES ARE NOT IN THE OBSERVATION
# def plot_observation_body_velocities(sim_df,test_dir): 
#     """Plots the body frame velocities from the observation"""
#     set_default_plot_rc()
#     ax = sim_df.plot(x="Time", y=[r"$u_o$",r"$v_o$", r"$w_o$"], kind="line")
#     ax.set_xlabel(xlabel="Time [s]", fontsize=14)
#     ax.set_ylabel(ylabel="Velocity [m/s]", fontsize=14)
#     ax.legend(loc="lower right", fontsize=14)
#     ax.set_ylim([-1.25,1.25])
#     plt.savefig(os.path.join(test_dir, "plots/observations/pure", f"episode{int(sim_df['Episode'].iloc[0])}.pdf"))

def plot_observation_LA(sim_df,test_dir):
    """Plots the lookahead vector in body frame from the observation"""
    set_default_plot_rc()
    ax = sim_df.plot(x="Time", y=[r"$la_{x}$", r"$la_{y}$", r"$la_{z}$"], kind="line")
    ax.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax.set_ylabel(ylabel="Position [m]", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-10,10])
    plt.savefig(os.path.join(test_dir, "plots/observations/pure", f"LA_episode{int(sim_df['Episode'].iloc[0])}.pdf"))

#STATE PLOTTING#
def plot_attitude(sim_df,test_dir):
    """Plots the state trajectories for the simulation data"""
    set_default_plot_rc()
    ax = sim_df.plot(x="Time", y=[r"$\phi$",r"$\theta$", r"$\psi$"], kind="line")
    ax.set_xlabel(xlabel="Time [s]",fontsize=14)
    ax.set_ylabel(ylabel="Angular position [rad]",fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-np.pi,np.pi])
    plt.savefig(os.path.join(test_dir, "plots/states", f"attitude_episode{int(sim_df['Episode'].iloc[0])}.pdf"))


def plot_velocity(sim_df,test_dir):
    """Plots the velocity trajectories for the simulation data"""
    set_default_plot_rc()
    ax = sim_df.plot(x="Time", y=[r"$u$",r"$v$"], kind="line")
    ax.plot(sim_df["Time"], sim_df[r"$w$"], dashes=[3,3], color="#88DD89", label=r"$w$")
    #Plot the r"$\v_{cmd}$" as the desired velocity
    ax.plot(sim_df["Time"], sim_df[r"$\v_{cmd}$"], dashes=[3,3], color="#EECC55", label=r"$v_{cmd}$")
    ax.plot
    ax.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax.set_ylabel(ylabel="Velocity [m/s]", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-0.25,2.25])
    plt.savefig(os.path.join(test_dir, "plots/states", f"velocity_episode{int(sim_df['Episode'].iloc[0])}.pdf"))


def plot_angular_velocity(sim_df,test_dir):
    """Plots the angular velocity trajectories for the simulation data"""
    set_default_plot_rc()
    ax = sim_df.plot(x="Time", y=[r"$p$",r"$q$", r"$r$"], kind="line")
    ax.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax.set_ylabel(ylabel="Angular Velocity [rad/s]", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-1,1])
    plt.savefig(os.path.join(test_dir, "plots/states", f"ang_vel_episode{int(sim_df['Episode'].iloc[0])}.pdf"))

#TODO add plotting of rewards yes
#REWARD PLOTTING#


#TODO add plotting of control inputs maybe??


#Outdated #TODO update to our case maybe??
# def plot_control_inputs(sim_dfs):
#     """ Plot control inputs from simulation data"""
#     set_default_plot_rc()
#     c = ['#EE6666', '#88BB44', '#EECC55']
#     for i, sim_df in enumerate(sim_dfs):
#         control = np.sqrt(sim_df[r"$\delta_r$"]**2+sim_df[r"$\delta_s$"]**2)
#         plt.plot(sim_df["Time"], sim_df[r"$\delta_s$"], linewidth=4, color=c[i])
#     plt.xlabel(xlabel="Time [s]", fontsize=14)
#     plt.ylabel(ylabel="Normalized Input", fontsize=14)
#     plt.legend(loc="lower right", fontsize=14)
#     plt.legend([r"$\lambda_r=0.9$", r"$\lambda_r=0.5$", r"$\lambda_r=0.1$"], loc="upper right", fontsize=14)
#     plt.ylim([-1.25,1.25])
#     plt.show()


# def plot_control_errors(sim_dfs):
#     """
#     Plot control inputs from simulation data
#     """
#     set_default_plot_rc()
#     c = ['#EE6666', '#88BB44', '#EECC55']
#     for i, sim_df in enumerate(sim_dfs):
#         error = np.sqrt(sim_df[r"e"]**2+sim_df[r"h"]**2)
#         plt.plot(sim_df["Time"], error, linewidth=4, color='r')
#         plt.plot(sim_df["Time"], sim_df[r"e"], linewidth=4, color='g')
#         plt.plot(sim_df["Time"], sim_df[r"h"], linewidth=4, color='b')
#         plt.plot(sim_df["Time"], sim_df[r"$\tilde{\chi}$"], linewidth=4, color='k')
#         plt.plot(sim_df["Time"], sim_df[r"$\tilde{\upsilon}$"], linewidth=4, color='y')
#     plt.xlabel(xlabel="Time [s]", fontsize=12)
#     plt.ylabel(ylabel="Tracking Error [m]", fontsize=12)
#     #plt.ylim([0,15])
#     plt.show()

#TRAJECTORY PLOTTING#
def plot_3d(env, sim_df, test_dir):
    """
    Plots the Quadcopter path in 3D inside the environment provided.
    """
    #plt.rcdefaults()
    #plt.rc('lines', linewidth=3)

    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('axes', facecolor='#ffffff', edgecolor='gray',
        axisbelow=True, grid=True)
    plt.rc('patch', edgecolor='#ffffff')

    # OLD PLOTTING
    # ax = env.plot3D()#(wps_on=False)
    # ax.scatter3D(sim_df[r"$X$"][0], sim_df[r"$Y$"][0], sim_df[r"$Z$"][0], color="#66FF66", label="Initial Position")
    # ax.plot3D(sim_df[r"$X$"], sim_df[r"$Y$"], sim_df[r"$Z$"], color="#EECC55", label="Quadcopter Path")#, linestyle="dashed")
    # ax.set_xlabel(xlabel=r"$x_w$ [m]", fontsize=18)
    # ax.set_ylabel(ylabel=r"$y_w$ [m]", fontsize=18)
    # ax.set_zlabel(zlabel=r"$z_w$ [m]", fontsize=18)
    # ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.legend(loc="upper right", fontsize=14)
    # f = lambda x,y,z: proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
    # ax.legend(loc="lower left", bbox_to_anchor=f(0,-120,100), 
    #       bbox_transform=ax.transData, fontsize=16)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # remove the ticks and labels (quickfix to fix old plotting code)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
  
    ax = env.unwrapped.plot3D(leave_out_first_wp=True)#(wps_on=False) # leave out first waypoint when initial position is same as first waypoint
    ax.scatter3D(sim_df[r"$X$"][0], sim_df[r"$Y$"][0], sim_df[r"$Z$"][0], color="#66FF66", label="Initial Position")
    ax.plot3D(sim_df[r"$X$"], sim_df[r"$Y$"], sim_df[r"$Z$"], color="#EECC55", label="Quadcopter Path")#, linestyle="dotted")
    ax.set_xlabel(xlabel="x [m]")
    ax.set_ylabel(ylabel="y [m]")
    ax.set_zlabel(zlabel="z [m]")

    # Adjust tick settings
    ax.tick_params(axis='x', which='major', labelsize=8)
    ax.tick_params(axis='y', which='major', labelsize=8, labelleft=True)
    ax.tick_params(axis='z', which='major', labelsize=8, labelleft=True)

    # Reduce the number of ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))

    ax.legend(loc="upper right")


    # Set limits of plot based on extremum points of path and obstacles
    
    # Force limits to be equal
    # lim = 100
    # ax.set_xlim([-lim,lim])
    # ax.set_ylim([-lim,lim])
    # ax.set_zlim([-lim,lim])
    plt.savefig(os.path.join(test_dir, "plots", f"episode{int(sim_df['Episode'].iloc[0])}.pdf"), bbox_inches='tight', pad_inches=0.4)


def plot_multiple_3d(env, sim_dfs):
    """
    Plots multiple Quadcopter paths in 3D inside the environment provided.
    """
    plt.rcdefaults()
    c = ['#EE6666', '#88BB44', '#EECC55']
    styles = ["dashed", "dashed", "dashed"]
    plt.rc('lines', linewidth=3)
    ax = env.unwrapped.plot3D()#(wps_on=False)
    for i,sim_df in enumerate(sim_dfs):
        ax.plot3D(sim_df[r"$X$"], sim_df[r"$Y$"], sim_df[r"$Z$"], color=c[i], linestyle=styles[i])
    ax.set_xlabel(xlabel="North [m]", fontsize=14)
    ax.set_ylabel(ylabel="East [m]", fontsize=14)
    ax.set_zlabel(zlabel="Down [m]", fontsize=14)
    ax.legend(["Path",r"$\lambda_r=0.9$", r"$\lambda_r=0.5$",r"$\lambda_r=0.1$"], loc="upper right", fontsize=14)
    plt.show()
    
#CURRENT PLOTTING#
def plot_current_data(sim_df):
    set_default_plot_rc()
    #---------------Plot current intensity------------------------------------
    current_labels = [r"$u_c$", r"$v_c$", r"$w_c$"] #TODO verify
    ax1 = sim_df.plot(x="Time", y=current_labels, linewidth=4, style=["-", "-", "-"] )
    ax1.set_title("Current", fontsize=18)
    ax1.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax1.set_ylabel(ylabel="Velocity [m/s]", fontsize=14)
    ax1.set_ylim([-1.25,1.25])
    ax1.legend(loc="right", fontsize=14)
    #ax1.grid(color='k', linestyle='-', linewidth=0.1)
    plt.show()
    #---------------Plot current direction------------------------------------
    """
    ax2 = ax1.twinx()
    ax2 = sim_df.plot(x="Time", y=[r"$\alpha_c$", r"$\beta_c$"], linewidth=4, style=["-", "--"] )
    ax2.set_title("Current", fontsize=18)
    ax2.set_xlabel(xlabel="Time [s]", fontsize=12)
    ax2.set_ylabel(ylabel="Direction [rad]", fontsize=12)
    ax2.set_ylim([-np.pi, np.pi])
    ax2.legend(loc="right", fontsize=12)
    ax2.grid(color='k', linestyle='-', linewidth=0.1)
    plt.show()
    """

#SPECIAL STUFF#
def plot_collision_reward_function():
    horizontal_angles = np.linspace(-70, 70, 300)
    vertical_angles = np.linspace(-70, 70, 300)
    gamma_x = 25
    epsilon = 0.05
    sensor_readings = 0.4*np.ones((300,300))
    image = np.zeros((len(vertical_angles), len(horizontal_angles)))
    for i, horizontal_angle in enumerate(horizontal_angles):
        horizontal_factor = (1-(abs(horizontal_angle)/horizontal_angles[-1]))
        for j, vertical_angle in enumerate(vertical_angles):
            vertical_factor = (1-(abs(vertical_angle)/vertical_angles[-1]))
            beta = horizontal_factor*vertical_factor + epsilon
            image[j,i] = beta*(1/(gamma_x*(sensor_readings[j,i])**4))
    print(image.round(2))
    ax = plt.axes()
    plt.colorbar(plt.imshow(image),ax=ax)
    ax.imshow(image, extent=[-70,70,-70,70])
    ax.set_ylabel("Vertical quadcopter-relative sensor angle [deg]", fontsize=14)
    ax.set_xlabel("Horizontal quadcopter-relative sensor angle [deg]", fontsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.show()


def write_report(test_dir: str, sim_df: pd.DataFrame, env, episode: int) -> None:
    sim_df.to_csv(os.path.join(test_dir, 'test_sim.csv'), index=False)
    episode_df = sim_df.loc[sim_df['Episode'] == episode]

    timesteps = episode_df.shape[0]
    avg_speed = np.sqrt(episode_df[r"$u$"]**2 + episode_df[r"$v$"]**2 + episode_df[r"$w$"]**2).mean()
    avg_ape = np.sqrt(episode_df[r'$e$']**2 + episode_df[r'$h$']**2).mean()
    iae_cross, iae_vertical = calculate_IAE(episode_df)
    progression = episode_df['Progression'].max()
    success = int(env.unwrapped.success)
    collision = int(env.unwrapped.collided)
    data = {
        'Episode': episode, 
        'Timesteps': timesteps, 
        'Avg Absolute Path Error': avg_ape,
        'Avg Speed': avg_speed,
        'IAE Cross': iae_cross,
        'IAE Vertical': iae_vertical,
        'Progression': progression, 
        'Success': success,
        'Collision': collision
    }
    summary = pd.read_csv(os.path.join(test_dir, 'test_summary.csv'))
    data_df = pd.DataFrame([data])

    # summary = summary.dropna(axis=1, how='all') #drop empty columns might be needed in the future
    # data_df = data_df.dropna(axis=1, how='all')

    summary = pd.concat([summary, data_df ], ignore_index=True)
    summary.to_csv(os.path.join(test_dir, 'test_summary.csv'), index=False)

    with open(os.path.join(test_dir, 'report.txt'), 'w') as f:
        f.write('# PERFORMANCE METRICS (LAST {} EPISODES AVG.)\n'.format(summary.shape[0]))
        f.write('{:<30}{:<30.2f}\n'.format('Avg. Number of Timesteps', summary['Timesteps'].mean()))
        f.write('{:<30}{:<30.2f}\n'.format('Avg. Absolute Path Error [m]', summary['Avg Absolute Path Error'].mean()))
        f.write('{:<30}{:<30.2f}\n'.format('Avg. Speed [m/s]', summary['Avg Speed'].mean()))
        f.write('{:<30}{:<30.2f}\n'.format('Avg. IAE Cross Track [m]', summary['IAE Cross'].mean())) #integral absolute error
        f.write('{:<30}{:<30.2f}\n'.format('Avg. IAE Verical Track [m]', summary['IAE Vertical'].mean()))
        f.write('{:<30}{:<30.2f}\n'.format('Avg. Progression [%]', summary['Progression'].mean()*100))
        f.write('{:<30}{:<30.2f}\n'.format('Success Rate [%]', summary['Success'].mean()*100))
        f.write('{:<30}{:<30.2f}\n'.format('Collision Rate [%]', summary['Collision'].mean()*100))


def plot_lidar():
    """
    Plots the Lidar readings in 3D example
    """
    plt.rcdefaults()
    plt.rc('lines', linewidth=3)

    ax = plt.axes(projection='3d')

    horizontal_angles = np.linspace(-180, 180, 15)
    vertical_angles = np.linspace(-90, 90, 15)
    distance = 25
    for horizontal_angle in horizontal_angles:
        for vertical_angle in vertical_angles:
            x = distance*np.cos(horizontal_angle)*np.cos(vertical_angle)
            y = distance*np.sin(horizontal_angle)*np.cos(vertical_angle)
            z = distance*np.sin(vertical_angle)
            ax.quiver(0, 0, 0, x, y, z, color='r')

    ax.set_xlabel(xlabel=r"$x_b$ [m]", fontsize=18)
    ax.set_ylabel(ylabel=r"$y_b$ [m]", fontsize=18)
    ax.set_zlabel(zlabel=r"$z_b$ [m]", fontsize=18)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc="upper right", fontsize=14)
    # f = lambda x,y,z: proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
    # ax.legend(loc="lower left", bbox_to_anchor=f(0,-120,100), 
    #       bbox_transform=ax.transData, fontsize=16)
    ax.set_xlim([-40,40])
    ax.set_ylim([-40,40])
    ax.set_zlim([-40,40])
    
    plt.show()


def rew_from_dist(danger_range, drone_closest_obs_dist, inv_abs_min_rew):
    r = -(((danger_range + inv_abs_min_rew*danger_range)/(drone_closest_obs_dist + inv_abs_min_rew*danger_range))-1)
    if r>0: r = 0
    return r

def rew_from_angle(danger_angle, angle_diff, inv_abs_min_rew):
    r = -(((danger_angle + inv_abs_min_rew*danger_angle)/(angle_diff + inv_abs_min_rew*danger_angle))-1)
    if r>0: r = 0
    return r


def collision_avoidance_reward_function(obsdists:np.array, 
                                             angle_diffs:np.array,
                                             danger_range:int, 
                                             danger_angle:int,  
                                             inv_abs_min_rew:float):
    reward_collision_avoidance = []
    for i in range(len(obsdists)):
        
        drone_closest_obs_dist = obsdists[i]
        angle_diff = angle_diffs[i]

        if (drone_closest_obs_dist < danger_range) and (angle_diff < danger_angle):
            range_rew = rew_from_dist(danger_range, drone_closest_obs_dist, inv_abs_min_rew)
            angle_rew = rew_from_angle(danger_angle, angle_diff, inv_abs_min_rew)
            
            if angle_rew > 0: angle_rew = 0 #Might not be necessary here but ensures that the reward is always negative 
            if range_rew > 0: range_rew = 0
            reward_collision_avoidance.append(range_rew + angle_rew)

        elif drone_closest_obs_dist < danger_range:
            range_rew = rew_from_dist(danger_range, drone_closest_obs_dist, inv_abs_min_rew)
            angle_rew = rew_from_angle(danger_angle, angle_diff, inv_abs_min_rew)

            if angle_rew > 0: angle_rew = 0 #In this case the angle reward may become positive as anglediff may < danger_angle
            if range_rew > 0: range_rew = 0 #So the capping is necessary

            reward_collision_avoidance.append(range_rew + angle_rew)
            
        else:
            reward_collision_avoidance.append(0)

    return reward_collision_avoidance

def CA_rew_v2(gauss_peak=1.5, gauss_std=50):
    TwoD_gaussian = np.zeros((240,320))
    center_synthetic_depth_map = np.zeros((240,320))
    corner_synthetic_depth_map = np.zeros((240,320))
    for i in range(240):
        for j in range(320):
            TwoD_gaussian[i,j] = gauss_peak*np.exp(-((i-240/2)**2 + (j-320/2)**2)/(2*gauss_std**2))
            #Add values between 0 and 10 to the depth map in the center of the depthmap
            if (i > 80 and i < 160) and (j > 120 and j < 200):
                center_synthetic_depth_map[i,j] = np.random.uniform(0,5)
            else:
                center_synthetic_depth_map[i,j] = 10
            #Add values to one corner of the depth map
            if (i > 160) and (j > 200):
                corner_synthetic_depth_map[i,j] = np.random.uniform(0,5)
            else:
                corner_synthetic_depth_map[i,j] = 10
    return TwoD_gaussian, center_synthetic_depth_map, corner_synthetic_depth_map

if __name__ == "__main__":

    # plot_lidar()

    ###Plotting of the collision avoidance reward### #OLD with angles
    # datapoints = 50
    # obsdists = np.linspace(0, 50, datapoints)
    # angle_diffs = np.linspace(0, 50, datapoints)
    # danger_range = 10 #m
    # danger_angle = 20 #deg
    # inv_abs_min_rew = 1/8
    # dist_rew = []
    # angle_rew = []
    # for i in range(len(obsdists)):
    #     dist_rew.append(rew_from_dist(danger_range, obsdists[i], inv_abs_min_rew))
    #     angle_rew.append(rew_from_angle(danger_angle, angle_diffs[i], inv_abs_min_rew))
    
    # capped_rew = collision_avoidance_reward_function(obsdists, angle_diffs, danger_range, danger_angle, inv_abs_min_rew)

    # set_default_plot_rc()
    # plt.plot(dist_rew)
    # plt.plot(angle_rew)
    # plt.plot(capped_rew)
    # plt.plot(np.array(dist_rew) + np.array(angle_rew))
    # plt.axhline(y=0, color='gray', linewidth=1)
    # plt.plot(np.argwhere(np.diff(np.sign(dist_rew)))[0], 0, 'ro')
    # plt.text(np.argwhere(np.diff(np.sign(dist_rew)))[0], 0, f"({np.argwhere(np.diff(np.sign(dist_rew)))[0][0]},{0})", fontsize=12)
    # plt.plot(np.argwhere(np.diff(np.sign(angle_rew)))[0], 0, 'ro')
    # plt.text(np.argwhere(np.diff(np.sign(angle_rew)))[0], 0, f"({np.argwhere(np.diff(np.sign(angle_rew)))[0][0]},{0})", fontsize=12)
    # plt.text(datapoints/2, -1/inv_abs_min_rew, f"danger range: {danger_range}m\n danger angle: {danger_angle}deg", fontsize=12)
    # plt.xlabel("Distance or angle [m or deg]")
    # plt.ylabel("Reward")
    # plt.legend(["Distance reward", "Angle reward", "Capped total reward", "Total reward"])
    # plt.title("Collision Avoidance Reward Parts")
    # plt.show()
    
    ###Plotting of the collision avoidance reward### #NEW only using distance
    # datapoints = 50
    # obsdists = np.linspace(0, 50, datapoints)
    # danger_range = 10 #m
    # inv_abs_min_rew = 1/16
    # dist_rew = []
    # for i in range(len(obsdists)):
    #     dist_rew.append(rew_from_dist(danger_range, obsdists[i], inv_abs_min_rew))
    
    # set_default_plot_rc()
    # plt.plot(dist_rew,label="Collision avoidance reward")
    # plt.axhline(y=0, color='gray', linewidth=1)

    # plt.plot(10, 0, 'co',label="Sensor range")
    # plt.text(10, 0, 10, fontsize=12)

    # plt.plot(0, -1/inv_abs_min_rew, 'mo',label="s3")
    # plt.text(0, -1/inv_abs_min_rew, f"{int(-1/inv_abs_min_rew)}", fontsize=12)

    # plt.xlabel("Distance [m]")
    # plt.ylabel("Reward")
    # plt.legend()
    # plt.show()

    #Plotting the v2 CA rew:
    #Setting the style to what we use:
    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('axes', facecolor='#ffffff', edgecolor='gray',
        axisbelow=True, grid=True)
    plt.rc('patch', edgecolor='#ffffff')

    dgaus,ce_synthdm,co_synthdm = CA_rew_v2(gauss_peak=1.5, gauss_std=30)
    #plot the 2d gaussian in 3d
    fig = plt.figure(1)
    # fig.suptitle("2D gauss to multiply with depthmaps", fontsize=16)
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(0, 320, 320)
    y = np.linspace(0, 240, 240)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, dgaus, cmap='magma')

    # #plot the center synthetic depth map in 3d
    # fig = plt.figure(2)
    # fig.suptitle("Synthetic depthmap object in center", fontsize=16)
    # ax = fig.add_subplot(111, projection='3d')
    # x = np.linspace(0, 320, 320)
    # y = np.linspace(0, 240, 240)
    # X, Y = np.meshgrid(x, y)
    # ax.plot_surface(X, Y, ce_synthdm, cmap='magma')

    # #plot the corner synthetic depth map in 3d
    # fig = plt.figure(3)
    # fig.suptitle("Synthetic depthmap object in corner", fontsize=16)
    # ax = fig.add_subplot(111, projection='3d')
    # x = np.linspace(0, 320, 320)
    # y = np.linspace(0, 240, 240)
    # X, Y = np.meshgrid(x, y)
    # ax.plot_surface(X, Y, co_synthdm, cmap='magma')

    #import depthmap_06 from npy file
    actual_depthmap = np.load("tests/test_img/depth_maps/depth_map0.npy")
    
    #plot the actual depth map in 3d
    fig = plt.figure(4)
    # fig.suptitle("Depth map", fontsize=16)
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(0, 320, 320)
    y = np.linspace(0, 240, 240)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, actual_depthmap, cmap='magma')

    #Do 1/depthmap
    inv_depthmap = 1/(actual_depthmap+0.0001)
    #plot the inv depth map in 3d
    fig = plt.figure(5)
    # fig.suptitle("1/Depth map", fontsize=16)
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(0, 320, 320)
    y = np.linspace(0, 240, 240)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, inv_depthmap, cmap='magma')

    #Multiply the 1/depthmap with the 2d gaussian
    scaled_actualDM_product = (dgaus*inv_depthmap)/1000
    #plot the product in 3d
    fig = plt.figure(6)
    # fig.suptitle("Scaled product of 2d gauss and 1/depth map", fontsize=16)
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(0, 320, 320)
    y = np.linspace(0, 240, 240)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, scaled_actualDM_product, cmap='magma')


    #Calculate the reward from the synthetic depth maps
    # non_s_ce_dm = ce_synthdm + 0.0001
    # divby1_non_s_ce_dm = 1/non_s_ce_dm
    # scaled_center_product = (dgaus*divby1_non_s_ce_dm)/1000
    # rew = -np.sum(scaled_center_product)
    # print("rew when in center", rew)

    # non_s_co_dm = co_synthdm + 0.0001
    # divby1_non_s_co_dm = 1/non_s_co_dm
    # scaled_corner_product = (dgaus*divby1_non_s_co_dm)/1000
    # rew = -np.sum(scaled_corner_product)
    # print("rew when in corner", rew)

    # #Visualizing the reward before it gets summed
    # fig = plt.figure(4)
    # fig.suptitle("Rew for center obj", fontsize=16)
    # ax = fig.add_subplot(111, projection='3d')
    # x = np.linspace(0, 320, 320)
    # y = np.linspace(0, 240, 240)
    # X, Y = np.meshgrid(x, y)
    # ax.plot_surface(X, Y, scaled_center_product, cmap='magma')

    # fig = plt.figure(5)
    # fig.suptitle("Rew for corner obj", fontsize=16)
    # ax = fig.add_subplot(111, projection='3d')
    # x = np.linspace(0, 320, 320)
    # y = np.linspace(0, 240, 240)
    # X, Y = np.meshgrid(x, y)
    # ax.plot_surface(X, Y, scaled_corner_product, cmap='magma')
    
    plt.show()