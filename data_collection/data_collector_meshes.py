import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gym_quad.objects.depth_camera as dc
import gym_quad.objects.mesh_obstacles as meshobs
from gym_quad.objects.QPMI import QPMI
import torch
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm




if __name__ == "__main__":

    # Datapath
    savepath = "./synthetic_depthmaps/"
    os.makedirs(savepath, exist_ok=True)
    figures_path = "./figures/"
    os.makedirs(figures_path, exist_ok=True)
    unit_sphere_path = "./mesh/sphere.obj"


    # Camera globals
    IMG_SIZE = (240, 320)           # (H, W) of physical depth cam images AFTER the preprocessing pipeline
    FOV = 75                        # Field of view in degrees
    MAX_MEASURABLE_DEPTH = 10.0      # Maximum measurable depth

    #init device to use gpu if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #init camera
    camera = dc.FoVPerspectiveCameras(device=device, fov=FOV)

    #Init rasterizer
    raster_settings = dc.RasterizationSettings(
    image_size=IMG_SIZE, 
    blur_radius=0.0, 
    faces_per_pixel=1, # Keep at 1, dont change
    perspective_correct=True, # Doesn't do anything(??), but seems to improve speed
    cull_backfaces=True # Do not render backfaces. MAKE SURE THIS IS OK WITH THE GIVEN MESH.
    )

    # Set distributions from which to sample sphere radii and positions wrt. camera
    # Min and max dist
    min_d = 0.5
    max_d = MAX_MEASURABLE_DEPTH

    # nOISE?
    add_noise = True

    # Set the distributions
    min_radius = 0.1
    max_radius = 4.0
    radius_distribution = torch.distributions.uniform.Uniform(min_radius, max_radius)


    n_imgs = 50000
    lambda_rate = 1.0

    alpha = np.deg2rad(FOV/2)
    ASPECT_RATIO = (4, 3)

    zs = []
    xs = []
    ys = []

    for i in tqdm(range(n_imgs), desc="Generating images"):
        n_spheres = np.random.poisson(lam=lambda_rate, size=1)[0]
        n_cubes = np.random.poisson(lam=lambda_rate, size=1)[0]
        n_cylinders = np.random.poisson(lam=lambda_rate, size=1)[0]
        
        obstacles = []
        sphere_obstacles = [] # used to calculate the room size
        for s_i in range(n_spheres):
            r = radius_distribution.sample().item()

            z = torch.distributions.uniform.Uniform(min_d + r, max_d + r).sample().item()
            phi = torch.distributions.uniform.Uniform(0, 2 * np.pi).sample().item()
            rx = z * np.tan(alpha) * (ASPECT_RATIO[0] / sum(ASPECT_RATIO))
            ry = z * np.tan(alpha) * (ASPECT_RATIO[1] / sum(ASPECT_RATIO))
            
            x = rx * np.cos(phi)
            y = ry * np.sin(phi)

            # Transform x, y from double to float
            x = x.item()
            y = y.item()

            zs.append(z)
            xs.append(x)
            ys.append(y)


            o = meshobs.SphereMeshObstacle(device=device, path=unit_sphere_path, radius=r, center_position=torch.tensor([x, y, z]))
            obstacles.append(o)
            sphere_obstacles.append(o)

        for c_i in range(n_cubes):
            width = torch.distributions.uniform.Uniform(0.5, 7.0).sample().item()
            height = torch.distributions.uniform.Uniform(0.5, 7.0).sample().item()
            depth = torch.distributions.uniform.Uniform(0.5, 7.0).sample().item()

            z = torch.distributions.uniform.Uniform(min_d + depth/2, max_d + depth/2).sample().item()
            phi = torch.distributions.uniform.Uniform(0, 2 * np.pi).sample().item()
            rx = z * np.tan(alpha) * (ASPECT_RATIO[0] / sum(ASPECT_RATIO))
            ry = z * np.tan(alpha) * (ASPECT_RATIO[1] / sum(ASPECT_RATIO))
            
            x = rx * np.cos(phi)
            y = ry * np.sin(phi)

            # Transform x, y from double to float
            x = x.item()
            y = y.item()

            zs.append(z)
            xs.append(x)
            ys.append(y)

            o = meshobs.CubeMeshObstacle(device=device, width=width, height=height, depth=depth, center_position=torch.tensor([x, y, z]), inverted=False)
            obstacles.append(o)
        
        # With a prob of 50% add a "cylinder" which now is just a high ass box
        if np.random.rand() > 0.5:
            for cy_i in range(n_cylinders):
                # radius = torch.distributions.uniform.Uniform(1.0, 3.0).sample().item()
                # height = 10.0
                #height = torch.distributions.uniform.Uniform(8.0, 16.0).sample().item()

                width = torch.distributions.uniform.Uniform(0.2, 1.5).sample().item()
                height = 30
                depth = torch.distributions.uniform.Uniform(0.5, 5.0).sample().item()

                z = torch.distributions.uniform.Uniform(min_d + depth/2, max_d + depth/2).sample().item()
                phi = torch.distributions.uniform.Uniform(0, 2 * np.pi).sample().item()
                rx = z * np.tan(alpha) * (ASPECT_RATIO[0] / sum(ASPECT_RATIO))
                ry = z * np.tan(alpha) * (ASPECT_RATIO[1] / sum(ASPECT_RATIO))
                
                x = rx * np.cos(phi)
                y = ry * np.sin(phi)

                # Transform x, y from double to float
                x = x.item()
                y = y.item()

                zs.append(z)
                xs.append(x)
                ys.append(y)

                o = meshobs.CubeMeshObstacle(device=device, width=width, height=height, depth=depth,center_position=torch.tensor([x, y, z]), inverted=False)
                obstacles.append(o)

        # Add enclosing cube of scene at random indices
        if np.random.rand() > 0.10:
            # Create dummy path from behind the camera to somewhere around max depth in the FOV
            start = np.array([0, 0, -0.2])
            mid = np.array([np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(1, 5)])
            end = np.array([np.random.uniform(-15, 15), np.random.uniform(-15, 15), np.random.uniform(8, MAX_MEASURABLE_DEPTH + 5)])
            waypoints = np.array([start, mid, end])
            path = QPMI(waypoints)
            padding = 0.5
            bounds, _ = meshobs.get_scene_bounds(sphere_obstacles, path, padding=padding)
            #calculate the size of the room
            width = bounds[1] - bounds[0] #z in tri and pt3d, x in enu
            height = bounds[3] - bounds[2] #y in tri and pt3d, z in enu
            depth = bounds[5] - bounds[4] #x in tri and pt3d y in enu
            #The room wants the coordinates in the tri/pt3d format
            room_center = torch.tensor([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
            cube = meshobs.CubeMeshObstacle(device=device,width=width, height=height, depth=depth, center_position=room_center)
            obstacles.append(cube)

        if obstacles:
            scene = meshobs.Scene(device=device, obstacles=obstacles) 
            renderer = dc.DepthMapRenderer(device=device, camera=camera, scene=scene, raster_settings=raster_settings, MAX_MEASURABLE_DEPTH=MAX_MEASURABLE_DEPTH)
            depth_map = renderer.render_depth_map().to(device)
        else:
            depth_map = torch.ones(IMG_SIZE, device=device) * MAX_MEASURABLE_DEPTH
                
        if add_noise:
            if np.random.rand() > 0.25:
                sigma = 0.1 #[m]
                noise = torch.normal(mean=0, std=sigma, size=IMG_SIZE, device=device)
                depth_map += noise
        
        # Save depth map as grayscale png with values like the actual depth
        depth_map = depth_map.squeeze().cpu()*1000 # convert to mm
        depth_map = depth_map.to(torch.float32)
        depth_map = depth_map.numpy()
        depth_map = depth_map.astype(np.uint16)
        cv2.imwrite(f"{savepath}depth_{i}.png", depth_map, [cv2.IMWRITE_PNG_COMPRESSION, 0]) # no compression

        if i%1000 == 0:
            plt.style.use('ggplot')
            plt.rc('font', family='serif')
            plt.rc('xtick', labelsize=12)
            plt.rc('ytick', labelsize=12)
            plt.rc('axes', labelsize=12)

            plt.figure(figsize=(8, 6))
            depth_map = depth_map / 1000 # convert back to meters for viz
            plt.imshow(depth_map, cmap="magma")
            plt.clim(0.0, MAX_MEASURABLE_DEPTH)
            plt.colorbar(label="Depth [m]", aspect=30, orientation="vertical", fraction=0.0235, pad=0.04)
            plt.axis("off")
            plt.savefig(f'{figures_path}depth_{i}.pdf', bbox_inches='tight')
            plt.close()
        

    # display some stats of z and the positions
    min_z = min(zs)
    max_z = max(zs)
    mean_z = np.mean(zs)
    median_z = np.median(zs)
    std_z = np.std(zs)
    print(f"Min z: {min_z}, Max z: {max_z}, Mean z: {mean_z}, Median z: {median_z}, Std z: {std_z}")

    # create plot of mean xy positions
    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Mean xy positions")
    plt.savefig(f"{figures_path}mean_xy_positions", bbox_inches='tight')

    # create plot of xyz positions, seen from different angles in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, s = 0.1, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("xyz positions")
    plt.savefig(f"{figures_path}xyz_positions.pdf", bbox_inches='tight')

    # create subplot of xyz positions seen from the different axes (approximately)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # set figsize to make the subplots bigger and quadratic
    axs[0].scatter(xs, ys, s = 0.1, alpha=0.5)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title("xy positions")
    axs[1].scatter(xs, zs, s = 0.1, alpha=0.5)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("z")
    axs[1].set_title("xz positions")
    axs[2].scatter(ys, zs, s = 0.1, alpha=0.5)
    axs[2].set_xlabel("y")
    axs[2].set_ylabel("z")
    axs[2].set_title("yz positions")
    plt.savefig(f"{figures_path}xyz_positions_subplots.pdf", bbox_inches='tight')

    # create ostogram of the depth values
    plt.figure()
    plt.hist(zs, bins=100)
    plt.xlabel("Depth [m]")
    plt.ylabel("Frequency")
    plt.title("Depth histogram")
    plt.savefig(f"{figures_path}depth_histogram.pdf", bbox_inches='tight')




    





