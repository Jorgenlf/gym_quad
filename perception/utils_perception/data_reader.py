import cv2
import os
import json
import numpy as np
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image



class SunRGBD:
    """
    Class for handling the SUN RGB-D dataset, methods are specific to the dataset.

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

    def merge_folders_from_stripped(self, folder1, folder2, output_folder, prefix='img_d', suffix='png'):
        """For merging folders with data from https://github.com/ankurhanda/sunrgbd-meta-data"""
        if os.path.exists(output_folder):
            print(f"Data already exists in {output_folder}")
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
    

    def split_train_validate_test(self, train_test_split:float, train_val_split:float, shuffle:bool = True, seed:int=42) -> None:
        """Splits the data into train, validate and test sets"""
        data_size = len(os.listdir(self.data_path_depth))
        train_size = int(data_size * train_test_split)
        val_size = int(train_size * train_val_split)
        test_size = data_size - train_size
        train_size = train_size - val_size

        # Create the train, validate and test folders for depth images 
        if not os.path.exists(f"{self.data_path_depth}_train"):
            os.makedirs(f"{self.data_path_depth}_train")
        if not os.path.exists(f"{self.data_path_depth}_val"):
            os.makedirs(f"{self.data_path_depth}_val")
        if not os.path.exists(f"{self.data_path_depth}_test"):
            os.makedirs(f"{self.data_path_depth}_test")
        
        # Shuffle the files and copy them to the train, validate and test folders
        files = os.listdir(self.data_path_depth)
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(files)
        for i, file in enumerate(files):
            if i < train_size:
                shutil.copyfile(os.path.join(self.data_path_depth, file), os.path.join(f"{self.data_path_depth}_train", file))
            elif i < train_size + val_size:
                shutil.copyfile(os.path.join(self.data_path_depth, file), os.path.join(f"{self.data_path_depth}_val", file))
            else:
                shutil.copyfile(os.path.join(self.data_path_depth, file), os.path.join(f"{self.data_path_depth}_test", file))
                

class DataReader:
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

    def load_split_data_sunrgbd(self, sun:SunRGBD, seed, shuffle:bool=True):
        
        orig_data_path = sun.orig_data_path

        if not os.path.exists(self.path_img_depth):
            sun.merge_folders_from_stripped(folder1=f"{orig_data_path}/sunrgbd_train_depth", 
                                            folder2=f"{orig_data_path}/sunrgbd_test_depth", 
                                            output_folder=self.path_img_depth, 
                                            prefix='img_d', suffix='png')
            
            sun.merge_folders_from_stripped(folder1=f"{orig_data_path}/sunrgbd_train_img", 
                                            folder2=f"{orig_data_path}/sunrgbd_test_img", 
                                            output_folder=self.path_img_depth, 
                                            prefix='img_rgb', suffix='jpg')
            
        if not os.path.exists(self.path_img_depth + "_train"): # If exists, assume that the data has already been split
            sun.split_train_validate_test(train_test_split=self.train_test_split, train_val_split=self.train_val_split, shuffle=shuffle)


        train_data = CustomDepthDataset(root_dir=self.path_img_depth + "_train",
                                        transform=self.transforms_train)
        val_data   = CustomDepthDataset(root_dir=self.path_img_depth + "_val",
                                        transform=self.transforms_validate)
        test_data  = CustomDepthDataset(root_dir=self.path_img_depth + "_test",
                                        transform=self.transforms_validate)
        
        rng = torch.Generator()
        if seed: rng.manual_seed(seed)
        
        #"""
        train_loader = DataLoader(train_data, 
                                  batch_size=self.batch_size, 
                                  shuffle=shuffle,
                                  generator=rng)     
        val_loader   = DataLoader(val_data, 
                                  batch_size=self.batch_size, 
                                  shuffle=shuffle,
                                  generator=rng)
        test_loader  = DataLoader(test_data, 
                                  batch_size=1, 
                                  shuffle=shuffle,
                                  generator=rng)
        """
        
        # USED FOR FAST ITERATION ON PLOTTING CODE
        indices_1 = list(range(1000))
        indices_2 = list(range(400))
        train_loader = DataLoader(Subset(train_data, indices_1), 
                                  batch_size=self.batch_size, 
                                  shuffle=shuffle,
                                  generator=rng)     
        val_loader   = DataLoader(Subset(val_data, indices_2), 
                                  batch_size=self.batch_size, 
                                  shuffle=shuffle,
                                  generator=rng)
        test_loader  = DataLoader(test_data, 
                                  batch_size=1, 
                                  shuffle=shuffle,
                                  generator=rng)
        
        #"""


        return train_loader, val_loader, test_loader
    

class CustomDepthDataset(Dataset):
    """
    - Custom depth image dataset for the SUN RGB-D dataset. Images are assumed to be 16-bit depth images.
    - Images are converted to meters and normalized to [0, 1] range, and then converted to tensors.
    - Output tensors have one channel.
    - Additional transformations can be added to the transform argument in the constructor.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the depth images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]
        self.depth_divide_by_to_get_meters = 10000
        self.max_measurable_distance = 65535
        self.min_measurable_distance = 0
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path)
        img_np = np.array(img, dtype=np.float32)  # Convert to numpy array and ensure it's float32

        # Convert to meters and normalize
        #img_np = img_np / self.depth_divide_by_to_get_meters  # Conversion to meters
        img_np = img_np / self.max_measurable_distance  # Normalization

        # Due to a bug, some pixels have values slightly larger than 1 (1,0000001 or 1.0000002), set these to 1
        #img_np[img_np > 1] = 1.0

        # Convert to 1-channeled tensor
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # Add channel dimension

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor
    
class RealSenseDataset(Dataset):
    """
    - Custom depth image dataset for the collected Intel Realsense deph images. 
    - Images are normalized to [0, 1] range, and then converted to tensors.
    - Output tensors have one channel.
    - Additional transformations can be added to the transform argument in the constructor.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the depth images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]
        self.max_measurable_distance = 65535
        self.min_measurable_distance = 0
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        img_np = np.array(img, dtype=np.float32)  # Convert to numpy array and ensure it's float

        # Normalize
        #img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        # set all pixel values above 6500 in img_np to 6500
        img_np[img_np > 6500] = 6500
        img_np = (img_np - self.min_measurable_distance) / 6500

        # Convert to 1-channeled tensor
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # Add channel dimension

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor
    

class RealSenseDataset_v2(Dataset):
    """
    - Custom depth image dataset for the collected Intel Realsense deph images. 
    - Images are normalized to [0, 1] range, and then converted to tensors.
    - Output tensors have one channel.
    - Additional transformations can be added to the transform argument in the constructor.
    - Max measurable distance is 10 meters.
    - Can be used for training instad of SUN RGBD
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the depth images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]
        self.max_measurable_distance = 10000 #8000
        self.min_measurable_distance = 0
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        img_np = np.array(img, dtype=np.float32)  # Convert to numpy array and ensure it's float

        # Truncate to 10 meters
        img_np[img_np > self.max_measurable_distance] = self.max_measurable_distance
        # Normalize to [0, 1]
        img_np = img_np / self.max_measurable_distance

        # Convert to 1-channeled tensor
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # Add channel dimension

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor

class DataReaderRealSensev2:
    def __init__(self,
                 path_img_depth:str,
                 batch_size:int,
                 train_test_split:float,
                 train_val_split:float,
                 transforms_train:transforms.Compose,
                 transforms_validate:transforms.Compose,
                 ) -> None:
        self.path_img_depth = path_img_depth
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.train_val_split = train_val_split
        self.transforms_train = transforms_train
        self.transforms_validate = transforms_validate

    def load_split_data_realsense(self, seed, shuffle:bool=True):
        if not os.path.exists(self.path_img_depth + "_train"): # If exists, assume that the data has already been split
            self.split_train_validate_test(train_test_split=self.train_test_split, train_val_split=self.train_val_split, shuffle=shuffle)

        train_data = RealSenseDataset_v2(root_dir=self.path_img_depth + "_train",
                                        transform=self.transforms_train)
        val_data   = RealSenseDataset_v2(root_dir=self.path_img_depth + "_val",
                                        transform=self.transforms_validate)
        test_data  = RealSenseDataset_v2(root_dir=self.path_img_depth + "_test",
                                        transform=self.transforms_validate)
        
        rng = torch.Generator()
        if seed: rng.manual_seed(seed)
        #"""
        train_loader = DataLoader(train_data, 
                                  batch_size=self.batch_size, 
                                  shuffle=shuffle,
                                  generator=rng)     
        val_loader   = DataLoader(val_data, 
                                  batch_size=self.batch_size, 
                                  shuffle=shuffle,
                                  generator=rng)
        test_loader  = DataLoader(test_data, 
                                  batch_size=1, 
                                  shuffle=shuffle,
                                  generator=rng)
        """
        indices_1 = list(range(100))
        indices_2 = list(range(40))
        train_loader = DataLoader(Subset(train_data, indices_1), 
                                  batch_size=self.batch_size, 
                                  shuffle=shuffle,
                                  generator=rng)     
        val_loader   = DataLoader(Subset(val_data, indices_2), 
                                  batch_size=self.batch_size, 
                                  shuffle=shuffle,
                                  generator=rng)
        test_loader  = DataLoader(Subset(test_data, indices_2),
                                  batch_size=1, 
                                  shuffle=shuffle,
                                  generator=rng)#"""
        return train_loader, val_loader, test_loader
    
    def split_train_validate_test(self, train_test_split:float, train_val_split:float, shuffle:bool = True, seed:int=42) -> None:
        """Splits the data into train, validate and test sets"""
        data_size = len(os.listdir(self.path_img_depth))
        train_size = int(data_size * train_test_split)
        val_size = int(train_size * train_val_split)
        test_size = data_size - train_size
        train_size = train_size - val_size

        # Create the train, validate and test folders for depth images 
        if not os.path.exists(f"{self.path_img_depth}_train"):
            os.makedirs(f"{self.path_img_depth}_train")
        if not os.path.exists(f"{self.path_img_depth}_val"):
            os.makedirs(f"{self.path_img_depth}_val")
        if not os.path.exists(f"{self.path_img_depth}_test"):
            os.makedirs(f"{self.path_img_depth}_test")
        
        # Shuffle the files and copy them to the train, validate and test folders
        files = os.listdir(self.path_img_depth)
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(files)
        
        for i, file in enumerate(files):
            if i < train_size:
                shutil.copyfile(os.path.join(self.path_img_depth, file), os.path.join(f"{self.path_img_depth}_train", file))
            elif i < train_size + val_size:
                shutil.copyfile(os.path.join(self.path_img_depth, file), os.path.join(f"{self.path_img_depth}_val", file))
            else:
                shutil.copyfile(os.path.join(self.path_img_depth, file), os.path.join(f"{self.path_img_depth}_test", file))
        
