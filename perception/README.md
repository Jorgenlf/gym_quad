# Perception Module

This module contains the code for the perception component of the drone simulation. It includes scripts for running and pre-training the perception model, as well as utilities for data handling, loading and saving models, and plotting.

## Files

- `VAE/...`: Contains the encoders, decoders and VAE arcitechture.
- `run_perception.py`: Runs the perception model on the depth data. Takes parsed args.
- `train_perception.py`: Trains the perception model using depth data.
- `utils_perception/data_reader.py`: Contains utility functions for reading and processing the depth data.
- `utils_perception/plotting.py`: Plotting utilities.
- `data/...`: Depth (and rgb) data.
- `models/...`: Trained models.
- `results/...`: Results from training. Divided into numerical and plots like: results/model_name/numerical_or_plots/experiment_id/

Run pretraining example:
```
python run_perception.py train --num_seeds 5  --latent_dims 64 --beta 1.5 --batch_size 32 --learning_rate 0.001 --epochs 20 --plot losses --save_model --exp_id 7
```

For more detailed explenation of arguments:
```
python run_perception.py -h
```

### Pretraining VAE using SUN RGBD dataset:
1. Get stripped depth (and possibly rgb) images from [this repo](https://github.com/ankurhanda/sunrgbd-meta-data?tab=readme-ov-file) created by Github user ankurhanda, under "Training and test data for depth" in the readme.
2. Create directory "data/sunrgbd_stripped" and place the downloaded folders in here.
3. Run. Images should be renamed, shuffeled and reorganized into train, validate and test folders next to "sunrgbd_stripped". A folder "sunrgbd_images_depth" is also created. This contains all of the depth images from the dataset.

<br/>

Original dataset is obtained from [https://rgbd.cs.princeton.edu/](https://rgbd.cs.princeton.edu/), and is fully described in the following paper by B. Zhou et al.: <br/>
B. Zhou, A. Lapedriza, J. Xiao, A. Torralba, and A. Oliva. *Learning Deep Features for Scene Recognition using Places Database Advances* in Neural Information Processing Systems 27 (NIPS2014)

### References
[1] N. Silberman, D. Hoiem, P. Kohli, R. Fergus. *Indoor segmentation and support inference from rgbd images*. In ECCV, 2012.<br/>
[2] A. Janoch, S. Karayev, Y. Jia, J. T. Barron, M. Fritz, K. Saenko, and T. Darrell. *A category-level 3-d object dataset: Putting the kinect to work. In ICCV Workshop on Consumer Depth Cameras for Computer Vision*, 2011.<br/>
[3] J. Xiao, A. Owens, and A. Torralba. *SUN3D: A database of big spaces reconstructed using SfM and object labels*. In ICCV, 2013
