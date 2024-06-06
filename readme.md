# Path Following and Collision Avoidance for Quadcopters using Deep Reinforcement Learning
This repo implements a 6-DOF simulation model for a quadcopter according to the stable baselines (OpenAI) interface for reinforcement learning control.


## Quick results overview
Gifs of what the quadcopter sees with an animation of where it is in the scene do last part if possible.

#### Helix test scenario
![](media\Test_scenario_gifs_&_webp\Helix\helix.gif)

#### Cave test scenario
![](media\Test_scenario_gifs_&_webp\Cave\Cave.gif)

#### Vertical test scenario
![](media\Test_scenario_gifs_&_webp\Vertical\vertical.gif)

#### Horizontal test scenario
![](media\Test_scenario_gifs_&_webp\Horizontal\horizontal.gif)

#### Deadend test scenario
![](media\Test_scenario_gifs_&_webp\Deadend\deadend.gif)


## Getting Started

To install all packages needed in your virtual environment, run:

```
conda env create -f environment.yml
```
### If you want to download the stuff yourself or the .yml file doesnt work:
You can follow this guide or do the steps below:
https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

Essentially, on windows do these:

1.  ``` conda create -n [name] python=3.10 ```

2. Choose cuda version to use (Suggest 12.1 as it is most up to date and compatible with the rest of the packages at the time of writing). 
- If you dont have cuda: https://developer.nvidia.com/cuda-12-1-0-download-archive 
- If you have cuda and need to change version follow this guide: https://github.com/bycloudai/SwapCudaVersionWindows   
- cuda 12.1: ``` conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia ```
- cuda 11.8: ``` conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia ```

3.  ``` conda install -c fvcore -c iopath -c conda-forge fvcore iopath ```

4.  ``` pip install "git+https://github.com/facebookresearch/pytorch3d.git" ``` 

5.  ```pip install gymnasium stable-baselines3 rich numba trimesh python-fcl vispy tensorboard imageio snakeviz scipy pycollada pyglet vtk pyvista trame```

### Training an agent:
The drl environment hyperparameters can be tuned in the main config file: [gym_quad/drl_config.py]. This is then copied to the [gym_quad/train3d.py] file where one can change certain hyperparameteres to better support training mode and also select the remaining hyperparameters: curriculum training setup, PPO and feature extractor.

For training an agent, run:

```
python train3d.py --exp_id [x] --n_cpu [y]
```

- x: experiment id number
- y: number of cpus to train on

### Running an agent in the environment
Copies the [gym_quad/drl_config.py] hyperparameters and changes certain hyperparam supporting running of agent in the [gym_quad/run3d.py] file.

For running an agent in any scenario, use:
```
python run3d.py --env "" --exp_id x --run_scenario "..." --trained_scenario "..." --agent x --episodes x 
```

- env: the gym environment to use (defaults to LV_VAE_MESH-v0)
- exp_id: which experiment to retrieve agent from (defaults to 1)
- run_scenario: which scenario to run (defaults to line)
- trained_scenario: which scenario the agent was trained in (defaults to line)
- agent: The timestep of the agent (defaults to the "last_model.zip" saved model from a completed training)
- episodes: how many episodes to repeat the run. (defaults to one)

There exists some additional args. For more info view the [gym_quad/utils.py] file and view the parse_experiment_info() function.

#### Running in manual debug mode
The run3d.py script has a mode for realtime visualization where you can control the quadcopter using wasd and spacebar. 

EITHER 

open the run3d.py script uncomment the 
    ```args = Namespace(manual_control=True, env = "LV_VAE_MESH-v0", save_depth_maps=False)``` 
line and run the script using Fn+f5 

OR

run the script from the terminal and set the two arguments --RT_vis True and ----manual_control True.

This is an overview of how keyboardinputs map to moving the quadcopter:

| Key Press | Action Description | Input Mapping | Notes |
|-----------|--------------------|---------------|-------|
| `w`       | Forward            | `[1, 0.3, 0]` | Moves the drone forward |
| `a`       | Rotate left        | `[-1, 0, 1]`  | Applies maximum positive yaw, rotating the drone to the left. |
| `d`       | Rotate right       | `[-1, 0, -1]` | Applies maximum negative yaw, essentially rotating the drone to the right. |
| `s`       | Down               | `[1, -1, 0]`  | Directs the drone downwards, using a negative pitch. This is less usual as geometric control typically aims to maintain hovering. |
| `Space`   | Up                 | `[1, 1, 0]`   | Directs the drone upwards; the most positive pitch is used to achieve this. |
| `escape`  | Exit environment   | N/A           | Closes the simulation or control environment. |

### Generating results 
Copies the [gym_quad/drl_config.py] hyperparameters and changes certain hyperparam supporting resultgen in [gym_quad/result_gen.py] for mass result generation. Four modes:

1. Testing all trained agents in the trainedlist across all scenarios in the testlist
    ``` python result_gen.py --exp_id 19 --episodes 10 --trained_list expert expert_perturbed --test_list horizontal vertical deadend random_corridor house --test_all_agents True```
2. Testing a specific trained agent from a specific training scenario across all scenarios in the testlist
    ```python result_gen.py --exp_id 19 --episodes 10 --trained_list expert --test_list horizontal vertical deadend random_corridor house --agent "name"```


