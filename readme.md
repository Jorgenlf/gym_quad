# Path Following and Collision Avoidance for Quadcopters using Deep Reinforcement Learning

This repo implements a 6-DOF simulation model for a quadcopter according to the stable baselines (OpenAI) interface for reinforcement learning control.
## Getting Started

To install all packages needed in your virtual environment, run:

```
conda env create -f environment.yml
```
### If you want to download the stuff yourself or the .yml file doesnt work:
You can follow this guide or do the steps below:
https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

Essentially on windows do these:

1.  ``` conda create -n [name] python=3.10 ```

2. Choose cuda version to use (Suggest 12.1 as it is most up to date and compatible with the rest of the packages at the time of writing). 
- If you dont have cuda: https://developer.nvidia.com/cuda-12-1-0-download-archive 
- If you have cuda and need to change version follow this guide: https://github.com/bycloudai/SwapCudaVersionWindows   
- cuda 12.1: ``` conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia ```
- cuda 11.8: ``` conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia ```

3.  ``` conda install -c fvcore -c iopath -c conda-forge fvcore iopath ```

4.  ``` pip install "git+https://github.com/facebookresearch/pytorch3d.git" ``` 

5.  ```pip install gymnasium stable-baselines3 rich numba trimesh python-fcl vispy tensorboard imageio snakeviz scipy pycollada pyglet vtk pyvista```

### Training an agent:

All hyperparameters and setup can be tuned in the file [gym_quad/train3d.py] and [gym_quad/gym_quad/__init__.py].

For training an agent, run:

```
python train3d.py --exp_id [x] --n_cpu [y]
```

- x: experiment id number
- y: number of cpus to train on


### Running an agent in the environment

For running an agent in any scenario, use:

```
python run3d.py --env "" --exp_id x --run_scenario "..." --trained_scenario "..." --agent x --episodes x 
```

- env: the gym environment to use
- exp_id: which experiment to retrieve agent from
- run_scenario: which scenario to run
- trained_scenario: which scenario the agent was trained in
- agent: The timestep of the agent (if left out attempts to use the last model saved model from a completed training)
- episodes: how many episodes to repeat the run.

There exists some additional args. For more info view the [gym_quad/utils.py] file and view the parse_experiment_info() function.


