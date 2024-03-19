import numpy as np
import pyrealsense2 as rs

# Assuming depth_image and color_image are numpy arrays representing depth and color images
# depth_image = np.zeros((height, width), dtype=np.uint16)  # Example depth image, depth values in millimeters
# color_image = np.zeros((height, width, 3), dtype=np.uint8)  # Example color image

class RGBDCamera:
    '''Class that emulates The depth part of an Intel RealSense D455 RGBD camera.'''

    def __init__(self, resolution=(1280, 720), fps=30):
        '''Initialize the RGBD camera with specified resolution and fps.'''
        self.resolution = resolution
        self.fps = fps

        # Create software device
        self.ctx = rs.context()
        self.dev = rs.software_device(self.ctx)

        # Add sensors for depth and color streams
        self.depth_sensor = self.dev.add_sensor("Depth")

        # Add intrinsics for depth and color streams #TODO update to actual ones of D455
        self.depth_intrinsics = {
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
        self.depth_stream = self.configure_depth_stream(intrinsics=self.depth_intrinsics)

        # Create pointcloud
        self.pc = rs.pointcloud()

    def configure_depth_stream(self, intrinsics):
        '''Configure depth stream parameters.'''
        depth_stream = {
            "intrinsics": intrinsics,
            "fps": self.fps,
            "format": rs.format.z16
            # Add more parameters as needed
        }

        # Add depth stream to depth sensor
        self.depth_sensor.add_video_stream(depth_stream)
        return depth_stream

    def start(self):
        '''Start capturing frames from the camera.'''
        self.dev.start()

    def stop(self):
        '''Stop capturing frames from the camera.'''
        self.dev.stop()

    def capture_frame(self):
        '''Capture a synchronized depth frame.'''
        frames = self.syncer.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        return depth_frame

    def generate_pointcloud(self, depth_frame):
        '''Generate pointcloud from depth frame.'''
        points = self.pc.calculate(depth_frame)
        return points

    # Add more methods as needed for processing and using the captured frames and pointclouds

# Example usage:
if __name__ == "__main__":
    # Initialize RGBD camera
    camera = RGBDCamera()

    # Start capturing frames
    camera.start()

    try:
        while True:
            # Capture a frame
            depth_frame, color_frame = camera.capture_frame()

            # Generate pointcloud
            points = camera.generate_pointcloud(depth_frame, color_frame)

            # Process and use captured frames and pointclouds as needed
            # Example: Display frames using OpenCV
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow('Depth Frame', depth_image)
            cv2.imshow('Color Frame', color_image)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Stop capturing frames
        camera.stop()
        cv2.destroyAllWindows()
