## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt 
import os

np.set_printoptions(threshold=sys.maxsize)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("Requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Set filters
decimation = rs.decimation_filter()
depth_to_disparity_transform = rs.disparity_transform(True)
spatial = rs.spatial_filter()
disparity_to_depth_transform = rs.disparity_transform(False)
hole_filling = rs.hole_filling_filter()

colorizer = rs.colorizer()

       

location = "glassgaarden"
os.makedirs(f"{location}/color_imgs", exist_ok=True)
os.makedirs(f"{location}/depth_imgs", exist_ok=True)


try:
    counter = 0
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        # Pass depth frame through filters
        
        """
        plt.rcParams["axes.grid"] = False
        plt.rcParams['figure.figsize'] = [8, 4]
        
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        plt.imsave("depth_imgs/1.png", colorized_depth)
        
        decimation = rs.decimation_filter()
        decimated_depth = decimation.process(depth_frame)
        colorized_decimated_depth = np.asanyarray(colorizer.colorize(decimated_depth).get_data())
        plt.imsave("depth_imgs/2.png", colorized_decimated_depth)
        
        depth_to_disparity_transform = rs.disparity_transform(True)
        disparity_depth = depth_to_disparity_transform.process(decimated_depth)
        colorized_disparity_depth = np.asanyarray(colorizer.colorize(disparity_depth).get_data())
        plt.imsave("depth_imgs/3.png", colorized_disparity_depth)
        
        spatial = rs.spatial_filter()
        filtered_depth = spatial.process(disparity_depth)
        colorzed_filtered_depth = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
        plt.imsave("depth_imgs/4.png", colorzed_filtered_depth)
        
        disparity_to_depth_transform = rs.disparity_transform(False)
        spatial_depth = disparity_to_depth_transform.process(filtered_depth)
        colorized_spatial_depth = np.asanyarray(colorizer.colorize(spatial_depth).get_data())
        plt.imsave("depth_imgs/5.png", colorized_spatial_depth)
        
        hole_filling = rs.hole_filling_filter()
        filled_depth = hole_filling.process(spatial_depth)
        colorized_filled_depth = np.asanyarray(colorizer.colorize(filled_depth).get_data())
        plt.imsave("depth_imgs/6.png", np.asanyarray(filled_depth.get_data()))
        """

        #plt.imsave("depth_imgs/test_im.png", spatial_depth)
        frame = decimation.process(depth_frame)
        frame = depth_to_disparity_transform.process(frame)
        frame = spatial.process(frame)
        frame = disparity_to_depth_transform.process(frame)
        frame = hole_filling.process(frame)
        
        depth_frame = frame

        
        # Save image every 0.5 seconds
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        if counter % 15 == 0:
            save_counter = int(counter / 15)
            cv2.imwrite(f"{location}/color_imgs/{location}_color_{save_counter}.png", color_image)
            cv2.imwrite(f"{location}/depth_imgs/{location}_depth_{save_counter}.png", depth_image)
            print(f"Saved image {save_counter}")
            
        counter += 1
        
        
            
        #print(depth_image*depth_)
        #exit()
            
        #cv2.imwrite(f"color_{counter}.png", color_image)
        #cv2.imwrite(f"depth_{counter}.png", depth_image)
        
        # Show images
        #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('RealSense', images)
        #cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()