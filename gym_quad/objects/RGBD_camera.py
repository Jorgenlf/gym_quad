import numpy as np
import pyrealsense2 as rs
import cv2
import sys
import os
#Use os and sys to access the modules inside gym_quad (imports below are from gym_quad)
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent and grandparent directory of the current script
parent_dir = os.path.dirname(script_dir)
grand_parent_dir = os.path.dirname(parent_dir)
# Add the parent directory to the Python path
sys.path.append(grand_parent_dir)

from gym_quad.objects.quad import Quad
from gym_quad.objects.obstacle3d import Obstacle
from gym_quad.utils.geomutils import Rzyx
# Assuming depth_image is a numpy array representing depth images
# depth_image = np.zeros((height, width), dtype=np.uint16)  # Example depth image, depth values in millimeters+
class RGBDCamera:
    '''Class that emulates The depth part of an Intel RealSense D455 RGBD camera.'''

    def __init__(self, resolution=(640, 480), fps=30):
        '''Initialize the RGBD camera with specified resolution and fps.'''
        self.resolution = resolution
        self.fps = fps

        # Create software device
        self.dev = rs.software_device()
    
        # Add sensors for depth and color streams
        self.depth_sensor = self.dev.add_sensor("Depth")

        # Add intrinsics for depth and color streams #TODO update to actual ones of D455
        self.depth_intrinsics = { #TODO Make use of them in this class
            "fx": 500,  # Example focal length in pixels
            "fy": 500,
            "ppx": self.resolution[0] / 2,  # Principal point
            "ppy": self.resolution[1] / 2,
            "coeffs": [0, 0, 0, 0, 0],  # Distortion coefficients
            "width": self.resolution[0],  # Image width
            "height": self.resolution[1],  # Image height
            "model": rs.distortion.brown_conrady  # Distortion model
        }

        # Configure depth stream
        self.depth_stream = self.configure_depth_stream()

        # Create pointcloud
        self.pc = rs.pointcloud()


    def configure_depth_stream(self):
        '''Configure depth stream parameters.'''
        depth_stream = rs.video_stream()
        # Add depth stream to depth sensor
        self.depth_sensor.add_video_stream(depth_stream)
        return depth_stream

    def capture_frame(self): #TODO make it come from simulation
        '''Capture a synchronized depth frame.'''
        # Create a synthetic depth frame
        depth_frame = self.create_synthetic_depth_frame()
        return depth_frame

    def create_synthetic_depth_frame(self):
        '''Create a synthetic depth frame with random depth values.'''
        depth_image = np.random.randint(0, 1000, size=self.resolution, dtype=np.uint16)
        #Cant fin method to convert nparray to depthframe so I will create an empty one and populate it
        depth_frame = rs.depth_frame()  # Create an empty depth frame
        depth_frame.init(self.resolution[0], self.resolution[1], rs.format.z16, 1, 0)  # Initialize depth frame

        # Populate depth frame data from the NumPy array
        depth_data_pointer = depth_frame.get_data()
        depth_data_pointer_ptr = depth_data_pointer.cast(rs.uint16_ptr)
        depth_data_pointer_ptr[0:self.resolution[0]*self.resolution[1]] = depth_image.flatten()


        return depth_frame

    def generate_pointcloud(self, depth_frame):
        '''Generate pointcloud from depth frame.'''
        points = self.pc.calculate(depth_frame)
        return points

    # Add more methods as needed for processing and using the captured frames and pointclouds

class StereoDepthEstimationCamera:
    '''Class that emulates the depth capturing part of an Intel RealSense D455 RGBD camera.
    Use cv2 to create a depth image'''

    def __init__(self, resolution=(640, 480), fps=30):
        '''Initialize the RGBD camera with specified resolution and fps.'''
        self.resolution = resolution
        self.fps = fps

    def capture_images(self):
        '''Capture a pair of synchronized images.'''
        #TODO make it come from simulation

        imfolder = 'C:/Users/jflin/Code/Drone3D/gym_quad/media'
        # left_image = np.zeros(self.resolution, dtype=np.uint8)
        # right_image = np.zeros(self.resolution, dtype=np.uint8)
        # left_image[50:100, 50:100] = 255
        # right_image[55:105, 55:105] = 200
        left_image = cv2.imread(imfolder + '/left.png', cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(imfolder + '/right.png', cv2.IMREAD_GRAYSCALE)

        return left_image, right_image
    
    def estimate_depth(self, left_image, right_image):
        '''Estimate depth from a pair of synchronized images.'''
        depth_image = np.zeros(self.resolution, dtype=np.uint16)
        return depth_image
    
    # def computeDepthMapBM(self, left_image, right_image): https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
    #     nDispFactor = 12 # adjust this 
    #     stereo = cv2.StereoBM.create(numDisparities=16*nDispFactor, blockSize=21)
    #     disparity = stereo.compute(left_image,right_image)
    #     depth_image = disparity

    def computeDepthMapSGBM(self,left_image, right_image): #Better than BM more comp though https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html 
        window_size = 7
        min_disp = 16
        nDispFactor = 14 # adjust this (14 is good)
        num_disp = 16*nDispFactor-min_disp

        stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                    numDisparities=num_disp,
                                    blockSize=window_size,
                                    P1=8*3*window_size**2,
                                    P2=32*3*window_size**2,
                                    disp12MaxDiff=1,
                                    uniquenessRatio=15,
                                    speckleWindowSize=0,
                                    speckleRange=2,
                                    preFilterCap=63,
                                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

        # Compute disparity map
        disparity = stereo.compute(left_image,right_image).astype(np.float32) / 16.0
        depth_image = disparity
        return depth_image


import taichi as ti
ti.init(arch=ti.gpu) #cpu or gpu? If AMD GPU works use it opencl or vulkan(?)

@ti.func
def calculate_object_distance(alpha:ti.f32, 
                              beta:ti.f32, 
                              obs_rad:ti.f32, 
                              obs_x:ti.f32, 
                              obs_y:ti.f32, 
                              obs_z:ti.f32, 
                              quad_x:ti.f32, 
                              quad_y:ti.f32, 
                              quad_z:ti.f32, 
                              sensor_range:ti.f32):
    '''Searches along a sonar ray for an object'''
    s = 0.0
    fine_search = False
    while s < sensor_range:
        x = quad_x + s*ti.cos(alpha)*ti.cos(beta)
        y = quad_y + s*ti.sin(alpha)*ti.cos(beta)
        z = quad_z + s*ti.sin(beta)
        #Rough search
        if ti.sqrt((obs_x - x)**2 + (obs_y - y)**2 + (obs_z - z)**2) <= obs_rad and fine_search == False:  # Using ti.sqrt for vectorized operation
            fine_search = True
            s -= 1
        elif fine_search == False:
            s += 1
        
        #Fine search
        if ti.sqrt((obs_x - x)**2 + (obs_y - y)**2 + (obs_z - z)**2) <= obs_rad and fine_search == True:
            break
        elif fine_search == True:
            s += 0.1
    return s 

@ti.data_oriented
class taichiDepthCam():
    def __init__(self, resolution=(240, 320), sensor_span=(85,58), fps=30,sensor_range = 6.0): #OG 480,640 but VAE takes 240,320 so I will use that
        '''Initialize the RGBD camera with specified resolution and fps. 
        Init the depth image.
        Using Taichi to speed up calculations. https://docs.taichi-lang.org/ '''
        self.fps = fps
        self.resolution = resolution
        self.sensor_span = sensor_span
        self.sensor_range = sensor_range #meters

        self.depth_image = ti.field(dtype=ti.f32, shape=self.resolution)
        self.depth_image.fill(6.0) #6.0 is the max sensor range
        
        self.n_sensor_readings = self.resolution[0]*self.resolution[1]
        max_horizontal_angle = self.sensor_span[0]/2
        max_vertical_angle = self.sensor_span[1]/2
        self.sectors_horizontal = np.linspace(-max_horizontal_angle*np.pi/180, max_horizontal_angle*np.pi/180, self.resolution[0])
        self.sectors_vertical =  np.linspace(-max_vertical_angle*np.pi/180, max_vertical_angle*np.pi/180, self.resolution[1])

        # self.position = np.array([0,0,0]) #might include this later
        # self.attitude = np.array([0,0,0])

    def update_nearby_obstacles(self, obstacles, quadcopter:Quad):
        """
        Updates the nearby_obstacles array.
        """
        nearby_obstacles = []
        for obstacle in obstacles:
            distance_vec_world = obstacle.position - quadcopter.position
            distance = np.linalg.norm(distance_vec_world)
            distance_vec_BODY = np.transpose(Rzyx(*quadcopter.attitude)).dot(distance_vec_world)
            heading_angle_BODY = np.arctan2(distance_vec_BODY[1], distance_vec_BODY[0])
            pitch_angle_BODY = np.arctan2(distance_vec_BODY[2], np.sqrt(distance_vec_BODY[0]**2 + distance_vec_BODY[1]**2))

            # check if the obstacle is inside the sonar window
            if distance - quadcopter.safety_radius - obstacle.radius <= self.sensor_range and abs(heading_angle_BODY) <= self.resolution[0]*np.pi/180 \
            and abs(pitch_angle_BODY) <= self.resolution[1]*np.pi/180:
                nearby_obstacles.append(obstacle)
            elif distance <= obstacle.radius +  quadcopter.safety_radius:
                nearby_obstacles.append(obstacle)
        # Sort the obstacles such that the closest one is first
        nearby_obstacles.sort(key=lambda x: np.linalg.norm(x.position - quadcopter.position)) 
        return nearby_obstacles
    
            
    @ti.kernel
    def update_sensor_readings(self,    quad_heading:float,
                                        quad_pitch:float,
                                        sensor_range:float,
                                        obs_rads:ti.types.ndarray(), 
                                        nearby_obstacle_positions:ti.types.ndarray(), 
                                        quadcopter_position:ti.types.ndarray(),    
                                        sectors_horizontal:ti.types.ndarray(), 
                                        sectors_vertical:ti.types.ndarray()):
        """
        Updates the depth image.
        """
        # ti.loop_config(serialize=True) #TODO check if this is necessary or parallel is better
        q_x = quadcopter_position[0]
        q_y = quadcopter_position[1]
        q_z = quadcopter_position[2]
        s_range = sensor_range

        for j in range(self.resolution[1]): #TODO determine if calling self.res is fast or if I should use a local variable or a ti.ndrange
            b = quad_pitch + sectors_vertical[j]
            
            for i in range(self.resolution[0]):
                a = quad_heading + sectors_horizontal[i]
            
                for n in ti.ndrange(nearby_obstacle_positions.shape[0]):
                    o_x = nearby_obstacle_positions[n,0] 
                    o_y = nearby_obstacle_positions[n,1]
                    o_z = nearby_obstacle_positions[n,2]
                    rad = obs_rads[n]
                    s = calculate_object_distance(alpha=a,
                                                    beta = b, 
                                                    obs_rad = rad, 
                                                    obs_x = o_x, 
                                                    obs_y = o_y, 
                                                    obs_z = o_z, 
                                                    quad_x = q_x, 
                                                    quad_y = q_y, 
                                                    quad_z = q_z, 
                                                    sensor_range = s_range)
                    #Should add a check to not draw stuff twice rather than just taking the min?
                    self.depth_image[i, j] = ti.min(s, self.depth_image[i, j])

# Example usage:
if __name__ == "__main__":
    # Initialize RGBD camera
    import matplotlib.pyplot as plt
    import time
    
    camera_to_use = "taichi" # "stereoest", "rgbdIntellisense", "taichi"

    if camera_to_use == "taichi":
        # Example usage

        depth_camera = taichiDepthCam(resolution=(240, 320), sensor_span=(85,58), fps=30)
        #Test environment
        obstacle1 = Obstacle(0.5, [4,0,0])
        obstacle2 = Obstacle(1, [0,12,0])
        obstacle3 = Obstacle(1, [5,0,0])
        obstacles = [obstacle1, obstacle2, obstacle3]
        quadcopter = Quad(step_size=0.01,init_eta=np.zeros(6)) #pos = [0,0,0], att = [0,0,0]
        print("INIT DONE")
        #calc nearby obstacles
        nearby_obs = depth_camera.update_nearby_obstacles(obstacles, quadcopter)
        print("Number of nearby obstacles: ", len(nearby_obs), "\npositions: ", [obstacle.position for obstacle in nearby_obs])

        # Preparing data for update_sensor_readings kernel call all np arrays as type.ti.ndarray() accepts it
        qh = quadcopter.heading
        qp = quadcopter.pitch
        sr = depth_camera.sensor_range
        qpt = quadcopter.position
        sh = depth_camera.sectors_horizontal
        sv = depth_camera.sectors_vertical

        obs_rads = np.array([obstacle.radius for obstacle in nearby_obs])
        nopt = ti.ndarray(dtype=ti.float32,shape=(len(nearby_obs),3))
        nopt.from_numpy(np.array([obstacle.position for obstacle in nearby_obs]))
        
        # print("Starting kernel")
        # Calling the kernel
        _s = time.time()
        depth_camera.update_sensor_readings(quad_heading=qp, quad_pitch=qp, sensor_range=sr, obs_rads=obs_rads, nearby_obstacle_positions=nopt, quadcopter_position=qpt, sectors_horizontal=sh, sectors_vertical=sv)
        # print("Kernel done")

        # Accessing the depth image
        depth_image_np = depth_camera.depth_image.to_numpy()
        print(f"Kernel to calc depthimage ran in {time.time() - _s:0.2f} seconds")

        print("Showing depth image, shape:",depth_image_np.shape)
        #viewing the depth image
        plt.imshow(depth_image_np, cmap='gray')
        plt.colorbar()
        plt.show()






    elif camera_to_use == "stereoest":
        stereo_camera = StereoDepthEstimationCamera()
        left_image, right_image = stereo_camera.capture_images()
        plt.imshow(left_image, cmap='gray')
        plt.show()
        plt.imshow(right_image, cmap='gray')
        plt.show()
        depth_image = stereo_camera.estimate_depth(left_image, right_image)
        plt.imshow(depth_image, cmap='gray')
        plt.colorbar()
        plt.show()
    elif camera_to_use == "rgbdIntellisense":
        camera = RGBDCamera()

        try:
            while True:
                # Capture a frame
                depth_frame = camera.capture_frame()

                # Generate pointcloud
                points = camera.generate_pointcloud(depth_frame)

                # Process and use captured frames and pointclouds as needed
                # Example: Display depth frame using OpenCV
                depth_image = np.asanyarray(depth_frame.get_data())
                cv2.imshow('Depth Frame', depth_image)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()



 #### EXAMPLE IN C++ FROM REALSENSE SDK ####

# // License: Apache 2.0. See LICENSE file in root directory.
# // Copyright(c) 2017 Intel Corporation. All Rights Reserved.

# #include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
# #include <librealsense2/hpp/rs_internal.hpp>
# #include "example.hpp"

# #define STB_IMAGE_WRITE_IMPLEMENTATION
# #include <stb_image_write.h>
# #include <int-rs-splash.hpp>

# #define STB_IMAGE_IMPLEMENTATION
# #include <stb_image.h>

# const int W = 640;
# const int H = 480;
# const int BPP = 2;

# struct synthetic_frame
# {
#     int x, y, bpp;
#     std::vector<uint8_t> frame;
# };

# class custom_frame_source
# {
# public:
#     custom_frame_source()
#     {
#         depth_frame.x = W;
#         depth_frame.y = H;
#         depth_frame.bpp = BPP;

#         last = std::chrono::high_resolution_clock::now();

#         std::vector<uint8_t> pixels_depth(depth_frame.x * depth_frame.y * depth_frame.bpp, 0);
#         depth_frame.frame = std::move(pixels_depth);

#         auto realsense_logo = stbi_load_from_memory(splash, (int)splash_size, &color_frame.x, &color_frame.y, &color_frame.bpp, false);

#         std::vector<uint8_t> pixels_color(color_frame.x * color_frame.y * color_frame.bpp, 0);

#         memcpy(pixels_color.data(), realsense_logo, color_frame.x*color_frame.y * 4);

#         for (auto i = 0; i< color_frame.y; i++)
#             for (auto j = 0; j < color_frame.x * 4; j += 4)
#             {
#                 if (pixels_color.data()[i*color_frame.x * 4 + j] == 0)
#                 {
#                     pixels_color.data()[i*color_frame.x * 4 + j] = 22;
#                     pixels_color.data()[i*color_frame.x * 4 + j + 1] = 115;
#                     pixels_color.data()[i*color_frame.x * 4 + j + 2] = 185;
#                 }
#             }
#         color_frame.frame = std::move(pixels_color);
#     }

#     synthetic_frame& get_synthetic_texture()
#     {
#         return color_frame;
#     }

#     synthetic_frame& get_synthetic_depth(glfw_state& app_state)
#     {
#         draw_text(50, 50, "This point-cloud is generated from a synthetic device:");

#         auto now = std::chrono::high_resolution_clock::now();
#         if (now - last > std::chrono::milliseconds(1))
#         {
#             app_state.yaw -= 1;
#             wave_base += 0.1f;
#             last = now;

#             for (int i = 0; i < depth_frame.y; i++)
#             {
#                 for (int j = 0; j < depth_frame.x; j++)
#                 {
#                     auto d = 2 + 0.1 * (1 + sin(wave_base + j / 50.f));
#                     ((uint16_t*)depth_frame.frame.data())[i*depth_frame.x + j] = (int)(d * 0xff);
#                 }
#             }
#         }
#         return depth_frame;
#     }

#     rs2_intrinsics create_texture_intrinsics()
#     {
#         rs2_intrinsics intrinsics = { color_frame.x, color_frame.y,
#             (float)color_frame.x / 2, (float)color_frame.y / 2,
#             (float)color_frame.x / 2, (float)color_frame.y / 2,
#             RS2_DISTORTION_BROWN_CONRADY ,{ 0,0,0,0,0 } };

#         return intrinsics;
#     }

#     rs2_intrinsics create_depth_intrinsics()
#     {
#         rs2_intrinsics intrinsics = { depth_frame.x, depth_frame.y,
#             (float)depth_frame.x / 2, (float)depth_frame.y / 2,
#             (float)depth_frame.x , (float)depth_frame.y ,
#             RS2_DISTORTION_BROWN_CONRADY ,{ 0,0,0,0,0 } };

#         return intrinsics;
#     }

# private:
#     synthetic_frame depth_frame;
#     synthetic_frame color_frame;

#     std::chrono::high_resolution_clock::time_point last;
#     float wave_base = 0.f;
# };

# int main(int argc, char * argv[]) try
# {
#     window app(1280, 1500, "RealSense Capture Example");
#     glfw_state app_state;
#     register_glfw_callbacks(app, app_state);
#     rs2::colorizer color_map; // Save colorized depth for preview

#     rs2::pointcloud pc;
#     rs2::points points;
#     int frame_number = 0;

#     custom_frame_source app_data;

#     auto texture = app_data.get_synthetic_texture();

#     rs2_intrinsics color_intrinsics = app_data.create_texture_intrinsics();
#     rs2_intrinsics depth_intrinsics = app_data.create_depth_intrinsics();

#     //==================================================//
#     //           Declare Software-Only Device           //
#     //==================================================//

#     rs2::software_device dev; // Create software-only device

#     auto depth_sensor = dev.add_sensor("Depth"); // Define single sensor
#     auto color_sensor = dev.add_sensor("Color"); // Define single sensor

#     auto depth_stream = depth_sensor.add_video_stream({  RS2_STREAM_DEPTH, 0, 0,
#                                 W, H, 60, BPP,
#                                 RS2_FORMAT_Z16, depth_intrinsics });

#     auto color_stream = color_sensor.add_video_stream({  RS2_STREAM_COLOR, 0, 1, texture.x,
#                                 texture.y, 60, texture.bpp,
#                                 RS2_FORMAT_RGBA8, color_intrinsics });

#     dev.create_matcher( RS2_MATCHER_DEFAULT );  // Compare all streams according to timestamp
#     rs2::syncer sync;

#     depth_sensor.open(depth_stream);
#     color_sensor.open(color_stream);

#     depth_sensor.start(sync);
#     color_sensor.start(sync);

#     depth_stream.register_extrinsics_to(color_stream, { { 1,0,0,0,1,0,0,0,1 },{ 0,0,0 } });

#     while (app) // Application still alive?
#     {
#         synthetic_frame& depth_frame = app_data.get_synthetic_depth(app_state);

#         // The timestamp jumps are closely correlated to the FPS passed above to the video streams:
#         // syncer expects frames to arrive every 1000/FPS milliseconds!
#         rs2_time_t timestamp = (rs2_time_t)frame_number * 16;
#         auto domain = RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK;

#         depth_sensor.on_video_frame( { depth_frame.frame.data(),  // Frame pixels from capture API
#                                        []( void * ) {},           // Custom deleter (if required)
#                                        depth_frame.x * depth_frame.bpp,  // Stride
#                                        depth_frame.bpp,
#                                        timestamp, domain, frame_number,
#                                        depth_stream, 0.001f } );  // depth unit


#         color_sensor.on_video_frame( { texture.frame.data(),     // Frame pixels from capture API
#                                        []( void * ) {},          // Custom deleter (if required)
#                                        texture.x * texture.bpp,  // Stride
#                                        texture.bpp,
#                                        timestamp, domain, frame_number,
#                                        color_stream } );

#         ++frame_number;

#         rs2::frameset fset = sync.wait_for_frames();
#         rs2::frame depth = fset.first_or_default(RS2_STREAM_DEPTH);
#         rs2::frame color = fset.first_or_default(RS2_STREAM_COLOR);
#         // We cannot expect the syncer to always output both depth and color -- especially on the
#         // first few frames! Hiccups can always occur: OS stalls, processing demands, etc...
#         if (depth && color)
#         {
#             if (auto as_depth = depth.as<rs2::depth_frame>())
#             {
#                 pc.map_to(color);
#                 points = pc.calculate(as_depth);
#             }

#             // Upload the color frame to OpenGL
#             app_state.tex.upload(color);
#         }
#         draw_pointcloud(app.width(), app.height(), app_state, points);
#     }

#     return EXIT_SUCCESS;
# }
# catch (const rs2::error & e)
# {
#     std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
#     return EXIT_FAILURE;
# }
# catch (const std::exception& e)
# {
#     std::cerr << e.what() << std::endl;
#     return EXIT_FAILURE;
# }       