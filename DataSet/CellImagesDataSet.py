import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torchvision.io import read_image
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

class CellImagesDataSet(Dataset):
    def __init__(
            self, 
            metadata_dir, 
            images_dir,  
            plate_num= 24277,
            output_channel = "Mito",
            augmentation=None, 
            preprocessing=None,
    ):
        self.channels = ["Mito", "ER", "DNA", "RNA", "AGP"]
        self.metadata_dir = metadata_dir
        self.metadata_csv = pd.read_csv(metadata_dir+str(plate_num)+".csv")
        self.image_paths = images_dir
        self.output_channel = output_channel
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        dict_tensor = {}
        for channel in self.channels:

            channel_path = self.image_paths + self.metadata_csv[channel].iloc[i]
            # print(channel_path)
            if channel == self.output_channel:
                # https://discuss.pytorch.org/t/training-a-cnn-with-tiff-images-in-pytorch/9531
                # from PIL import Image
                # from torchvision.transforms import ToTensor
                # image = Image.open('/path/to/image.tif')
                # image = ToTensor()(image)
                output_tensor = ToTensor()(Image.open(channel_path))
            else:
                dict_tensor[channel]= ToTensor()(Image.open(channel_path))
        #https://www.tutorialspoint.com/how-to-join-tensors-in-pytorch
        #https://stackoverflow.com/questions/54307225/whats-the-difference-between-torch-stack-and-torch-cat-functions
        #stack-joining on new dimension, cat-joining on existing dimension
        total_tensor = torch.stack(tuple(dict_tensor.values()),1) 
        # for key in self.channels[1:]:
        #     if key== self.output_channel:
        #         output_tensor = value
        #     else:
        #         total_tensor = torch.tensor([total_tensor,dict_tensor[key]])
        return total_tensor, output_tensor
    def __len__(self):
        return self.metadata_csv.shape[1]