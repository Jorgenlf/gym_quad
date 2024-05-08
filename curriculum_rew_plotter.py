import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from train3d import scenarios

# CONSIDER DOING VARIANCE BANDS OR CONFIDENCE INTERVALS INSTEAD OF TRUE DATA BEHIND SMOOTHED DATA FOR LESS CLUTTERED PLOTS


# Constants
AGENT_PATH_NAMES = ["agent1", "agent2"]
AGENT_PLOT_NAMES = ["Agent 1", "Agent 2"]
VALUE_PATH_NAME = "reward"
VALUE_TO_PLOT = "Reward"
SUFFIX = "_tensorboard_PPO_1"

PATH = "./tensorboard_logs_to_plot_EXAMPLE/"
SAVE_NAME = f"{PATH}/plots/{VALUE_PATH_NAME}/{VALUE_PATH_NAME}_{'_'.join(AGENT_PATH_NAMES)}.pdf"
os.makedirs(f"{PATH}/plots/{VALUE_PATH_NAME}/", exist_ok=True)

SCENARIOS = {   "line"                 :  1e5,
                "easy"                 :  1e6,
                "proficient"           :  1e6,
                "intermediate"         :  1e6,
                #"expert"               :  1e6, #FILL IN 
                "easy_perturbed"       :  1e6, #Perturbed by noise
                "proficient_perturbed" :  1e6,
                "expert_perturbed"     :  1e6
             }


def load_scenario_data(path, agent, value, scenarios):
    data = {}
    for scenario in scenarios:
        try:
            file_path = os.path.join(path, f"{agent}/{value}/{scenario}{SUFFIX}.csv")
            data[scenario] = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Could not find file {file_path}. Skipping scenario {scenario}.")
    return data

def add_shading_and_axes(ax, timesteps, datapoints_per_scenario, scenarios, value_to_plot):
        # Set label first as it will be used to determine the y-axis limits
        l = ax.legend(loc='upper left', fontsize=10, framealpha=1.0,  fancybox=True)
        l.set_zorder(3)

        accumulated_fill_from_timestep = 0
        ylow_fill, yhigh_fill = ax.get_ybound()
        for i, scenario in enumerate(scenarios):
            color = "white" if i % 2 == 0 else "gray"
            ax.fill_between(timesteps[accumulated_fill_from_timestep:accumulated_fill_from_timestep+datapoints_per_scenario[scenario]+1], ylow_fill, yhigh_fill, color=color, alpha=0.25, zorder=1)
            accumulated_fill_from_timestep += datapoints_per_scenario[scenario]

        ax.set_xlabel("Timesteps")
        ax.set_ylabel(value_to_plot)
        ax.set_xlim(0, timesteps[-1] + timesteps[0])
        ax.set_ylim(ylow_fill, yhigh_fill)

def add_upper_axis(ax, scenarios, timesteps, datapoints_per_scenario):
        scenario_text_positions_x = []
        upper_ax_ticks = []
        accumulated_fill_from_timestep_upper = 0
        for scenario in scenarios:
            upper_ax_ticks.append(timesteps[accumulated_fill_from_timestep_upper])
            scenario_text_positions_x.append(timesteps[accumulated_fill_from_timestep_upper + datapoints_per_scenario[scenario]//2])
            accumulated_fill_from_timestep_upper += datapoints_per_scenario[scenario]

        upper_ax_ticks = upper_ax_ticks[1:] # Remove first tick

        ax_upper = ax.twiny()
        ax_upper.set_xlim(0, timesteps[-1])
        ax_upper.set_xticks(upper_ax_ticks)
        ax_upper.grid(True, which='both', linestyle='--', linewidth=0.8, color='gray', alpha=0.3, zorder=2)
        ax_upper.set_xticklabels([' ' for i in range(len(list(scenarios.keys())) - 1)])


        # Add the scenario names
        y_lim = ax.get_ylim()[1]
        pad_y_text = 2.0
        for i, scenario in enumerate(scenarios):
            ax_upper.text(scenario_text_positions_x[i], y_lim + pad_y_text, scenario, ha='center', va='baseline', rotation=45, transform=ax_upper.transData, fontsize=8)

        # Align axes
        ax_upper.set_xbound(ax.get_xbound())


def smooth_data_ma(values, window_size):
    """ Smooth the data using a simple moving average. """
    return np.convolve(values, np.ones(window_size) / window_size, mode='same')

def exponential_moving_average(values, alpha):
    """ Compute an exponential moving average of the data. """
    ema = [values[0]]
    for v in values[1:]:
        ema.append(alpha * v + (1 - alpha) * ema[-1])
    return ema


if __name__ == "__main__":
    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # This loop is used to find the lower and upper bounds for the shading as well as 
    # plotting the values for the scenarios in each segment separated in time
    # Keep as non-function for readability as per now
    lower_bound = 0
    upper_bound = 0
    for i, (agent_path_name, agent_plot_name) in enumerate(zip(AGENT_PATH_NAMES, AGENT_PLOT_NAMES)):

        data = load_scenario_data(PATH, agent_path_name, VALUE_PATH_NAME, SCENARIOS)
        if not data:
            print(f"No data found for agent {agent_path_name}. Skipping.")
            continue

        datapoints_per_scenario = {}
        timesteps, values = [0], []
        for scenario in data.keys():
            df = data[scenario]
            timesteps_ = [timesteps[-1] + int(t) for t in df["Step"]] # Convert to int, add the final timestep of the previous scenario
            timesteps += timesteps_ # extend the list of timesteps
            datapoints_per_scenario[scenario] = len(timesteps_) # Store the number of datapoints per scenario
            values += list(df["Value"]) # Store the values of the scenario
        # Remove first element of timesteps
        timesteps = timesteps[1:]

        # For testing purposes, we subtract 1 from the values of agent1
        if agent_path_name == "agent1":
            values = [v - 1 for v in values]
        
        lower_bound = min(lower_bound, min(values))
        upper_bound = max(upper_bound, max(values))
        

        # Ensure colors are the same for the two following plots of the same agent
        current_cycler = plt.rcParams['axes.prop_cycle'].by_key()['color'] # Initialize color cycler

        values_smooth = exponential_moving_average(values, 0.03)
        #values_smooth = gaussian_filter1d(values, sigma=10)

        ax.plot(timesteps, values, color=current_cycler[i], alpha=0.3, zorder=5) 
        ax.plot(timesteps, values_smooth, color=current_cycler[i], label=agent_plot_name, zorder=10)
        
    
    # Plot shading with final timesteps vector as all agens have the same total number of timesteps
    add_shading_and_axes(ax, timesteps, datapoints_per_scenario, SCENARIOS, VALUE_TO_PLOT)

    # Add upper axis with scenario names, dividing the plot in segments based on scenarios
    add_upper_axis(ax, SCENARIOS, timesteps, datapoints_per_scenario)

    plt.savefig(SAVE_NAME, bbox_inches='tight')
