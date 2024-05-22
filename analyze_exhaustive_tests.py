import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def extract_success_rate(report_path):
    with open(report_path, 'r') as file:
        content = file.read()
        # Find the Success Rate in the report
        match = re.search(r'Success Rate \[%\]\s+([\d.]+)', content)
        if match:
            return float(match.group(1))
        else:
            return None
        
def extract_progress_rate(report_path): #TODO make it possible to plot this as well maybe collision rate also
    with open(report_path, 'r') as file:
        content = file.read()
        # Find the Success Rate in the report
        match = re.search(r'Progress Rate \[%\]\s+([\d.]+)', content)
        if match:
            return float(match.group(1))
        else:
            return None        

def collect_results(base_dir):
    results = {}
    # Traverse through each test scenario directory
    for test_scen in os.listdir(base_dir):
        test_scen_dir = os.path.join(base_dir, test_scen)
        if os.path.isdir(test_scen_dir):
            # Traverse through each agent result directory
            for agent_dir in os.listdir(test_scen_dir):
                agent_dir_path = os.path.join(test_scen_dir, agent_dir)
                report_path = os.path.join(agent_dir_path, 'report.txt')
                if os.path.exists(report_path):
                    success_rate = extract_success_rate(report_path)
                    if success_rate is not None:
                        if 'last_model' in agent_dir:
                            agent_name = 'last_model'
                        else:
                            agent_name = agent_dir.split('_model_')[1]
                        if agent_name not in results:
                            results[agent_name] = []
                        results[agent_name].append(success_rate)
    return results

def calculate_average_success(results):
    avg_results = {agent: sum(rates)/len(rates) for agent, rates in results.items()}
    return avg_results

def visualize_results(avg_results,trained_scen):
    agents = list(avg_results.keys())
    success_rates = list(avg_results.values())

    plt.figure(figsize=(10, 6))
    plt.bar(agents, success_rates, color='skyblue')
    plt.xlabel(f'Agents from {trained_scen} Scenario')
    plt.ylabel('Average Success Rate [%]')
    plt.title('Average Success Rate Across All Test Scenarios')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    exp_id = 4001
    trained_scen = "expert"

    base_dir = f'log/LV_VAE_MESH-v0/Experiment {exp_id}/{trained_scen}/results_gen'  # Base directory containing test scenario directories
    results = collect_results(base_dir)
    avg_results = calculate_average_success(results)
    visualize_results(avg_results,trained_scen)
