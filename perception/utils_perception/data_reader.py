import cv2
import os
import json
import numpy as np
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image



class SunRGBD:
    """
    Original dataset is obtained from https://rgbd.cs.princeton.edu/, and is fully described in the following paper by B. Zhou et al.:
    B. Zhou, A. Lapedriza, J. Xiao, A. Torralba, and A. Oliva. Learning Deep Features for Scene Recognition using Places Database Advances in Neural Information Processing Systems 27 (NIPS2014)

    The dataset contains RGB-D images from NYU depth v2 [1], Berkeley B3DO [2], and SUN3D [3].
    Stripped rgb and depth images have been obtained from the following repo: https://github.com/ankurhanda/sunrgbd-meta-data.

    References:
    [1] N. Silberman, D. Hoiem, P. Kohli, R. Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012.
    [2] A. Janoch, S. Karayev, Y. Jia, J. T. Barron, M. Fritz, K. Saenko, and T. Darrell. A category-level 3-d object dataset: Putting the kinect to work. In ICCV Workshop on Consumer Depth Cameras for Computer Vision, 2011.
    [3] J. Xiao, A. Owens, and A. Torralba. SUN3D: A database of big spaces reconstructed using SfM and object labels. In ICCV, 2013
    """
    def __init__(self, orig_data_path:str, data_path_depth:str, data_path_rgb:str) -> None:
        self.orig_data_path = orig_data_path
        self.data_path_depth = data_path_depth
        self.data_path_rgb = data_path_rgb
        self.depth_divide_by_to_get_meters = 10000

    def merge_folders_from_stripped(self, folder1, folder2, output_folder, prefix='img_d', suffix='png'):
        """For merging folders with data from https://github.com/ankurhanda/sunrgbd-meta-data"""
        if os.path.exists(output_folder):
            print(f"Data already exists in {output_folder   }")
            return
        
        print(f"Merging {folder1} and {folder2} into {output_folder}")

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if prefix == 'img_d': # pngs must be converted to sortable names
            for filename in os.listdir(folder1):
                _, extension = os.path.splitext(filename)
                new_filename = f"{filename.split('.')[0]:0>5}{extension}" # Generate the new filename with leading zeros
                os.rename(os.path.join(folder1, filename), os.path.join(folder1, new_filename))
            for filename in os.listdir(folder2):
                _, extension = os.path.splitext(filename)
                new_filename = f"{filename.split('.')[0]:0>5}{extension}" # Generate the new filename with leading zeros
                os.rename(os.path.join(folder2, filename), os.path.join(folder2, new_filename))

        # Get the list of files in both folders
        files1 = sorted(os.listdir(folder1))
        files2 = sorted(os.listdir(folder2))

        # Copy files from folder 1 to the output folder with new names
        for i, file in enumerate(files1, start=1):
            src = os.path.join(folder1, file)
            dst = os.path.join(output_folder, f"{prefix}_{i:05d}.{suffix}")  
            shutil.copyfile(src, dst)

        # Copy files from folder 2 to the output folder with new names
        for i, file in enumerate(files2, start=len(files1) + 1):
            src = os.path.join(folder2, file)
            dst = os.path.join(output_folder, f"{prefix}_{i:05d}.{suffix}")  
            shutil.copyfile(src, dst)
    
    def split_train_validate_test(self, train_test_split:float, train_val_split:float) -> None:
        """Splits the data into train, validate and test sets"""
        data_size = len(os.listdir(self.data_path_depth))
        train_size = int(data_size * train_test_split)
        val_size = int(train_size * train_val_split)
        test_size = data_size - train_size
        train_size = train_size - val_size

        # Create the train, validate and test folders for depth images
        if not os.path.exists(f"{self.data_path_depth}_train"):
            os.makedirs(os.path.join(f"{self.data_path_depth}_train","dummyclass"))
        if not os.path.exists(f"{self.data_path_depth}_val"):
            os.makedirs(f"{self.data_path_depth}_val/dummyclass")
        if not os.path.exists(f"{self.data_path_depth}_test"):
            os.makedirs(f"{self.data_path_depth}_test/dummyclass")
        
        # Shuffle the files and copy them to the train, validate and test folders
        files = os.listdir(self.data_path_depth)
        np.random.shuffle(files)
        for i, file in enumerate(files):
            if i < train_size:
                shutil.copyfile(os.path.join(self.data_path_depth, file), os.path.join(f"{self.data_path_depth}_train","dummyclass", file))
            elif i < train_size + val_size:
                shutil.copyfile(os.path.join(self.data_path_depth, file), os.path.join(f"{self.data_path_depth}_val/dummyclass", file))
            else:
                shutil.copyfile(os.path.join(self.data_path_depth, file), os.path.join(f"{self.data_path_depth}_test/dummyclass", file))
        
        print('Splitted dataset into train, validate and test sets')
        print(f"Train size: {train_size}, Validate size: {val_size}, Test size: {test_size}")
        
    

"""
sun = SunRGBD(data_path_depth="../data/sunrgbd_images_depth", data_path_rgb="../data/sunrgbd_images_rgb")
orig_data_path = "../data/sunrgbd_stripped"
sun.merge_folders_from_stripped(folder1=f"{orig_data_path}/sunrgbd_train_depth", folder2=f"{orig_data_path}/sunrgbd_test_depth", output_folder="../data/sunrgbd_images_depth", prefix='img_d', suffix='png')
sun.merge_folders_from_stripped(folder1=f"{orig_data_path}/sunrgbd_train_img", folder2=f"{orig_data_path}/sunrgbd_test_img", output_folder="../data/sunrgbd_images_rgb", prefix='img_rgb', suffix='jpg')
sun.split_train_validate_test(train_test_split=0.7, train_val_split=0.2)
"""

class sjef:
    def __init__(self,
                 path_img_depth:str,
                 path_img_rgb:str,
                 batch_size:int,
                 train_test_split:float,
                 train_val_split:float,
                 transforms_train:transforms.Compose,
                 transforms_validate:transforms.Compose,
                 ) -> None:
        self.path_img_depth = path_img_depth
        self.path_img_rgb = path_img_rgb
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.train_val_split = train_val_split
        self.transforms_train = transforms_train
        self.transforms_validate = transforms_validate

    def load_data_sunrgbd(self, sun:SunRGBD):
        
        orig_data_path = sun.orig_data_path

        sun.merge_folders_from_stripped(folder1=f"{orig_data_path}/sunrgbd_train_depth", folder2=f"{orig_data_path}/sunrgbd_test_depth", output_folder="data/sunrgbd_images_depth", prefix='img_d', suffix='png')
        sun.merge_folders_from_stripped(folder1=f"{orig_data_path}/sunrgbd_train_img", folder2=f"{orig_data_path}/sunrgbd_test_img", output_folder="data/sunrgbd_images_rgb", prefix='img_rgb', suffix='jpg')
        
        sun.split_train_validate_test(train_test_split=self.train_test_split, train_val_split=self.train_val_split)


        train_data = torchvision.datasets.ImageFolder(root=self.path_img_depth + "_train", 
                                                      transform=self.transforms_train)
        val_data = torchvision.datasets.ImageFolder(root=self.path_img_depth + "_val",
                                                    transform=self.transforms_validate)
        test_data = torchvision.datasets.ImageFolder(root=self.path_img_depth + "_test",
                                                     transform=self.transforms_validate)
        

        train_loader = DataLoader(train_data, 
                                  batch_size=self.batch_size, 
                                  shuffle=True)
        
        val_loader = DataLoader(val_data, 
                                batch_size=self.batch_size, 
                                shuffle=True)
        
        test_loader = DataLoader(test_data, 
                                 batch_size=1, 
                                 shuffle=True)

        return train_loader, val_loader, test_loader
    

