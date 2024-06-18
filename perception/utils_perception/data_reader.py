"""
This module contains classes for custom depth image datasets. The classes are used to load and preprocess depth images for training and testing.
"""

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
        
        
        # This can be uncommented to only load a subset of the data for quick tests
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
        
class SyntheticDataset(Dataset):
    """
    - Custom depth image dataset with synthetic depth images. 
    - Images are normalized to [0, 1] range, and then converted to tensors.
    - Output tensors have one channel.
    - Additional transformations can be added to the transform argument in the constructor.
    - Max measurable distance is 10 meters.
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
        self.min_measurable_distance = 0
        self.max_measurable_distance = 10000 #mm
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        img_np = np.array(img, dtype=np.float32)  # Convert to numpy array and ensure it's float

        # Normalize to [0, 1]
        img_np = img_np / self.max_measurable_distance

        # Convert to 1-channeled tensor
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # Add channel dimension

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor

class DataReaderSynthetic:
    def __init__(self,
                 path_depth:str,
                 batch_size:int,
                 train_test_split:float,
                 train_val_split:float,
                 transforms_train:transforms.Compose,
                 transforms_validate:transforms.Compose,
                 ) -> None:
        self.path_img_depth = path_depth
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.train_val_split = train_val_split
        self.transforms_train = transforms_train
        self.transforms_validate = transforms_validate

    def load_split_data_synthetic(self, seed, shuffle:bool=True):
        if not os.path.exists(self.path_img_depth + "_train"): # If exists, assume that the data has already been split
            self.split_train_validate_test(train_test_split=self.train_test_split, train_val_split=self.train_val_split, shuffle=shuffle)

        train_data = SyntheticDataset(root_dir=self.path_img_depth + "_train",
                                        transform=self.transforms_train)
        val_data   = SyntheticDataset(root_dir=self.path_img_depth + "_val",
                                        transform=self.transforms_validate)
        test_data  = SyntheticDataset(root_dir=self.path_img_depth + "_test",
                                        transform=self.transforms_validate)
        
        rng = torch.Generator()
        if seed: rng.manual_seed(seed)

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
        



class CustomDepthDataset(Dataset): # NOT IN USE AS PER 18.06.2024
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
    
class RealSenseDataset(Dataset): # NOT IN USE AS PER 18.06.2024
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
    

