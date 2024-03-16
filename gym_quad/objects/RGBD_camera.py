import numpy as np

class RGBDCamera:
    '''Class that simulates an RGBD camera. 
    The camera captures depth and RGB images in the training environment which consists of 3D meshes.'''
    def __init__(self, resolution=(640, 480), fov=60):
        self.resolution = resolution
        self.fov = fov

    def capture_depth_image(self, mesh):
        # Generate depth image from 3D mesh
        depth_image = np.zeros(self.resolution, dtype=np.float32)
        # Perform depth calculation logic here
        
        return depth_image

    def capture_rgb_image(self, mesh): #PRobs not needed as we care about the depth image
        # Generate RGB image from 3D mesh
        rgb_image = np.zeros((*self.resolution, 3), dtype=np.uint8)
        # Perform RGB image generation logic here
        
        return rgb_image

    def capture_rgbd_image(self, mesh):
        # Generate RGBD image from 3D mesh
        depth_image = self.capture_depth_image(mesh)
        rgb_image = self.capture_rgb_image(mesh)
        rgbd_image = np.concatenate((rgb_image, depth_image[..., np.newaxis]), axis=-1)
        
        return rgbd_image