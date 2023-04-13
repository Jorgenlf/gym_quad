import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler


def parse_experiment_info():
    """Parser for the flags that can be passed with the run/train/test scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="WaypointPlanner-v0", type=str, help="Which environment to run/train/test")
    parser.add_argument("--exp_id", type=int, help="Which experiment number to run/train/test")
    parser.add_argument("--scenario", default="line", type=str, help="Which scenario to run")
    parser.add_argument("--controller_scenario", default=None, type=str, help="Which scenario the agent was trained in")
    parser.add_argument("--controller", default=None, type=int, help="Which model to load as main controller. Requires only integer")
    parser.add_argument("--episodes", default=1, type=int, help="How many episodes to run when testing the quadcopter")
    args = parser.parse_args()
    
    experiment_dir = os.path.join(r"./log", r"{}".format(args.env), r"Experiment {}".format(args.exp_id))

    if args.controller_scenario is not None:
        agent_path = os.path.join(experiment_dir, args.controller_scenario, "agents")
    else:
        agent_path = os.path.join(experiment_dir, args.scenario, "agents")
    if args.controller is not None:
        agent_path = os.path.join(agent_path, "model_" + str(args.controller) + ".pkl")
    else:
        agent_path = os.path.join(agent_path,"last_model.pkl")
    return experiment_dir, agent_path, args


def calculate_IAE(sim_df):
    """
    Calculates and prints the integral absolute error provided an environment id and simulation data
    """
    IAE_cross = sim_df[r"e"].abs().sum()
    IAE_vertical = sim_df[r"h"].abs().sum()
    return IAE_cross, IAE_vertical


def simulate_environment(episode, env, agent):
    global error_labels, current_labels, input_labels, state_labels
    state_labels = [r"$N$", r"$E$", r"$D$", r"$\phi$", r"$\theta$", r"$\psi$", r"$u$", r"$v$", r"$w$", r"$p$", r"$q$", r"$r$"]
    input_labels = [r"$\eta$", r"$\delta_r$", r"$\delta_s$",r"$\F_4$"]
    error_labels = [r"$\tilde{u}$", r"$\tilde{\chi}$", r"e", r"$\tilde{\upsilon}$", r"h"]
    # error_labels = [r"e", r"h"]
    labels = np.hstack(["Episode", "Time", "Progression", state_labels, input_labels, error_labels])
    
    done = False
    env.reset()
    while not done:
        action = agent.predict(env.observation, deterministic=True)[0]
        _, _, done, _ = env.step(action)
    errors = np.array(env.past_errors)
    time = np.array(env.time).reshape((env.total_t_steps,1))
    episode = np.full(((env.total_t_steps,1)), episode)
    progression = np.array(env.progression).reshape((env.total_t_steps,1))
    sim_data = np.hstack([episode, time, progression, env.past_states, env.past_actions, errors])
    df = pd.DataFrame(sim_data, columns=labels)
    return df, env


def set_default_plot_rc():
    """Sets the style for the plots report-ready"""
    colors = (cycler(color= ['#EE6666', '#3388BB', '#88DD89', '#EECC55', '#88BB44', '#FFBBBB']) +
                cycler(linestyle=['-',       '-',      '-',     '--',      ':',       '-.']))
    plt.rc('axes', facecolor='#ffffff', edgecolor='black',
        axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('grid', color='gray', linestyle='--')
    plt.rc('xtick', direction='out', color='black', labelsize=14)
    plt.rc('ytick', direction='out', color='black', labelsize=14)
    plt.rc('patch', edgecolor='#ffffff')
    plt.rc('lines', linewidth=4)


def plot_attitude(sim_df):
    """Plots the state trajectories for the simulation data"""
    set_default_plot_rc()
    ax = sim_df.plot(x="Time", y=[r"$\phi$",r"$\theta$", r"$\psi$"], kind="line")
    ax.set_xlabel(xlabel="Time [s]",fontsize=14)
    ax.set_ylabel(ylabel="Angular position [rad]",fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-np.pi,np.pi])
    plt.show()


def plot_velocity(sim_df):
    """Plots the velocity trajectories for the simulation data"""
    set_default_plot_rc()
    ax = sim_df.plot(x="Time", y=[r"$u$",r"$v$"], kind="line")
    ax.plot(sim_df["Time"], sim_df[r"$w$"], dashes=[3,3], color="#88DD89", label=r"$w$")
    ax.plot([0,sim_df["Time"].iloc[-1]], [1.5,1.5], label=r"$u_d$")
    ax.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax.set_ylabel(ylabel="Velocity [m/s]", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-0.25,2.25])
    plt.show()


def plot_angular_velocity(sim_df):
    """Plots the angular velocity trajectories for the simulation data"""
    set_default_plot_rc()
    ax = sim_df.plot(x="Time", y=[r"$p$",r"$q$", r"$r$"], kind="line")
    ax.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax.set_ylabel(ylabel="Angular Velocity [rad/s]", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-1,1])
    plt.show()


def plot_control_inputs(sim_dfs):
    """ Plot control inputs from simulation data"""
    set_default_plot_rc()
    c = ['#EE6666', '#88BB44', '#EECC55']
    for i, sim_df in enumerate(sim_dfs):
        control = np.sqrt(sim_df[r"$\delta_r$"]**2+sim_df[r"$\delta_s$"]**2)
        plt.plot(sim_df["Time"], sim_df[r"$\delta_s$"], linewidth=4, color=c[i])
    plt.xlabel(xlabel="Time [s]", fontsize=14)
    plt.ylabel(ylabel="Normalized Input", fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.legend([r"$\lambda_r=0.9$", r"$\lambda_r=0.5$", r"$\lambda_r=0.1$"], loc="upper right", fontsize=14)
    plt.ylim([-1.25,1.25])
    plt.show()


def plot_control_errors(sim_dfs):
    """
    Plot control inputs from simulation data
    """
    set_default_plot_rc()
    c = ['#EE6666', '#88BB44', '#EECC55']
    for i, sim_df in enumerate(sim_dfs):
        error = np.sqrt(sim_df[r"e"]**2+sim_df[r"h"]**2)
        plt.plot(sim_df["Time"], error, linewidth=4, color='r')
        plt.plot(sim_df["Time"], sim_df[r"e"], linewidth=4, color='g')
        plt.plot(sim_df["Time"], sim_df[r"h"], linewidth=4, color='b')
        plt.plot(sim_df["Time"], sim_df[r"$\tilde{\chi}$"], linewidth=4, color='k')
        plt.plot(sim_df["Time"], sim_df[r"$\tilde{\upsilon}$"], linewidth=4, color='y')
    plt.xlabel(xlabel="Time [s]", fontsize=12)
    plt.ylabel(ylabel="Tracking Error [m]", fontsize=12)
    #plt.ylim([0,15])
    plt.show()


def plot_3d(env, sim_df, test_dir):
    """
    Plots the Quadcopter path in 3D inside the environment provided.
    """
    plt.rcdefaults()
    plt.rc('lines', linewidth=3)

    ax = env.plot3D()#(wps_on=False)
    ax.plot3D(sim_df[r"$N$"], sim_df[r"$E$"], sim_df[r"$D$"], color="#EECC55", label="Quadcopter Path")#, linestyle="dashed")
    ax.set_xlabel(xlabel=r"$x_w$ [m]", fontsize=14)
    ax.set_ylabel(ylabel=r"$y_w$ [m]", fontsize=14)
    ax.set_zlabel(zlabel=r"$z_w$ [m]", fontsize=14)
    ax.legend(loc="upper right", fontsize=14)
    # ax.set_xlim([-10,40])
    # ax.set_ylim([-10,40])
    # ax.set_zlim([-10,10])
    ax.set_xlim([-200,200])
    ax.set_ylim([-200,200])
    ax.set_zlim([-200,200])
    
    plt.savefig(os.path.join(test_dir, "plots", f"episode{int(sim_df['Episode'].iloc[0])}.pdf"))
    # plt.show()


def plot_multiple_3d(env, sim_dfs):
    """
    Plots multiple Quadcopter paths in 3D inside the environment provided.
    """
    plt.rcdefaults()
    c = ['#EE6666', '#88BB44', '#EECC55']
    styles = ["dashed", "dashed", "dashed"]
    plt.rc('lines', linewidth=3)
    ax = env.plot3D()#(wps_on=False)
    for i,sim_df in enumerate(sim_dfs):
        ax.plot3D(sim_df[r"$N$"], sim_df[r"$E$"], sim_df[r"$D$"], color=c[i], linestyle=styles[i])
    ax.set_xlabel(xlabel="North [m]", fontsize=14)
    ax.set_ylabel(ylabel="East [m]", fontsize=14)
    ax.set_zlabel(zlabel="Down [m]", fontsize=14)
    ax.legend(["Path",r"$\lambda_r=0.9$", r"$\lambda_r=0.5$",r"$\lambda_r=0.1$"], loc="upper right", fontsize=14)
    plt.show()
    

def plot_current_data(sim_df):
    set_default_plot_rc()
    #---------------Plot current intensity------------------------------------
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
    avg_ape = np.sqrt(episode_df['e']**2 + episode_df['h']**2).mean()
    iae_cross, iae_vertical = calculate_IAE(episode_df)
    progression = episode_df['Progression'].max()
    success = int(env.success)
    collision = int(env.collided)
    data = {
        'Episode': episode, 
        'Timesteps': timesteps, 
        'Avg Absolute Path Error': avg_ape,
        'IAE Cross': iae_cross,
        'IAE Vertical': iae_vertical,
        'Progression': progression, 
        'Success': success,
        'Collision': collision
    }

    summary = pd.read_csv(os.path.join(test_dir, 'test_summary.csv'))
    summary = summary.append(data, ignore_index=True)
    summary.to_csv(os.path.join(test_dir, 'test_summary.csv'), index=False)

    with open(os.path.join(test_dir, 'report.txt'), 'w') as f:
        f.write('# PERFORMANCE METRICS (LAST {} EPISODES AVG.)\n'.format(summary.shape[0]))
        f.write('{:<30}{:<30.2f}\n'.format('Avg. Number of Timesteps', summary['Timesteps'].mean()))
        f.write('{:<30}{:<30.2f}\n'.format('Avg. Absolute Path Error [m]', summary['Avg Absolute Path Error'].mean()))
        f.write('{:<30}{:<30.2f}\n'.format('Avg. IAE Cross Track [m]', summary['IAE Cross'].mean()))
        f.write('{:<30}{:<30.2f}\n'.format('Avg. IAE Verical Track [m]', summary['IAE Vertical'].mean()))
        f.write('{:<30}{:<30.2f}\n'.format('Avg. Progression [%]', summary['Progression'].mean()*100))
        f.write('{:<30}{:<30.2f}\n'.format('Success Rate [%]', summary['Success'].mean()*100))
        f.write('{:<30}{:<30.2f}\n'.format('Collision Rate [%]', summary['Collision'].mean()*100))


if __name__ == "__main__":
    plot_collision_reward_function()
