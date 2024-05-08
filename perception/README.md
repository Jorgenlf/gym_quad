# Perception Module

This module contains the code for the perception component of the drone simulation. It includes scripts for running and pre-training the perception model, as well as utilities for data handling, loading and saving models, and plotting.

(Insert fancy pictue of VAE model from the thesis here)

## Files

- `VAE/...`: Contains the encoders, decoders and VAE arcitechture.
- `run_perception.py`: Runs the perception model on the depth data. Takes parsed args.
- `train_perception.py`: Trains the perception model using depth data.
- `utils_perception/data_reader.py`: Contains utility functions for reading and processing the depth data.
- `utils_perception/plotting.py`: Plotting utilities.
- `data/...`: Depth (and rgb) data.
- `models/...`: Trained models.
- `results/...`: Results from training. Divided into numerical and plots like: results/model_name/numerical_or_plots/experiment_id/

## Run

Run pretraining example:
```
python run_perception.py train --num_seeds 5  --latent_dims 64 --beta 1.5 --batch_size 32 --learning_rate 0.001 --epochs 20 --plot losses --save_model --exp_id 7
```

For more detailed explenation of arguments:
```
python run_perception.py -h
```

<br/>

**Important:** <br/>
When loading a model for testing with `--model_name <modelname>`, make sure all parameters that affect the architecture of the net (i.e., `--latent_dim`, `--batch_size`, and ofc. `IMG_SIZE` and `NUM_CHANNELS` which are initialized in the VAE object) is set to the same values as during training of the given model. Also, make sure the encoder and decoder objects in the code are identical as during training.

## Pretraining VAE using SUN RGBD dataset:
1. Step 1 get our data from somewhere (get link)
2. put in folder on this and that format
3. run this an dthat script

