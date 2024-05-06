import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train3d import scenarios

if __name__ == "__main__":
    scenarios = {"easy"                :  1e6,
                "intermediate"         :  1e6,
                "expert"               :  1e6}

    agent_path_names = ["agent1", "agent2"]
    agent_plot_names = ["Agent 1", "Agent 2"]
    value_path_name = "reward"
    value_to_plot = "Reward"        
    savename = "./example.pdf"  

    # read data, assuming the data is stored as agent -> value -> scenario
    path = "./test_results_test/"

    # Plot the data stored at value
    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    lower_bound = 0
    upper_bound = 0
    for agent_path_name, agent_plot_name in zip(agent_path_names, agent_plot_names):

        data = {}
        for scenario in scenarios:
            data[scenario] = pd.read_csv(os.path.join(path, f"{agent_path_name}/{value_path_name}/{scenario}.csv"))

        datapoints_per_scenario = {}
        timesteps = [0]
        values = []
        for scenario in scenarios:
            final_timestep = timesteps[-1]
            timesteps_ = data[scenario]["Step"]
            timesteps_ = [final_timestep + int(t) for t in timesteps_] # Convert to int, add the final timestep of the previous scenario
            timesteps += timesteps_ # extend the list of timesteps
            datapoints_per_scenario[scenario] = len(timesteps_) # Store the number of datapoints per scenario
            values += list(data[scenario]["Value"]) # Store the values of the scenario
        # Remove the first element of timesteps
        timesteps = timesteps[1:]

        # For testing purposes, we subtract 1 from the values of agent1
        if agent_path_name == "agent1":
            values = [v - 1 for v in values]
        
        lower_bound = min(lower_bound, min(values))
        upper_bound = max(upper_bound, max(values))

        ax.plot(timesteps, values, label=agent_plot_name, zorder=10) 

    # Plot shading with final timesteps vector as all agens have the same total number of timesteps
    accumulated_fill_from_timestep = 0
    ylow_fill, yhigh_fill = ax.get_ybound()
    for i, scenario in enumerate(scenarios):
        if i % 2 == 0:
            color = "gray"
        else:
            color = "white"
        ax.fill_between(timesteps[accumulated_fill_from_timestep:accumulated_fill_from_timestep+datapoints_per_scenario[scenario]+1], ylow_fill, yhigh_fill, color=color, alpha=0.25, zorder=1)
        accumulated_fill_from_timestep += datapoints_per_scenario[scenario]

    ax.set_xlabel("Timesteps")
    ax.set_ylabel(value_to_plot)
    l = ax.legend()
    l.set_zorder(3)
    ax.set_xlim(0, timesteps[-1] + timesteps[0])
    ax.set_ylim(ylow_fill, yhigh_fill)


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
    pad_y_text = 0.05
    for i, scenario in enumerate(scenarios):
        ax_upper.text(scenario_text_positions_x[i], y_lim + pad_y_text, scenario, ha='center', va='center', rotation=0, transform=ax_upper.transData, fontsize=12)

    # Align axes
    ax_upper.set_xbound(ax.get_xbound())

    #save figure
    plt.savefig(savename, bbox_inches='tight')
