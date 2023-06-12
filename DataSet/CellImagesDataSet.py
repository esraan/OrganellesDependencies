
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
#https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class CellImagesDataSet(Dataset):
    def __init__(
            self,
            metadata_df,
            images_dir,
            output_channel="Mito",
            augmentation=None,
            preprocessing=None,
    ):
        self.channels = ["Mito", "ER", "DNA", "RNA", "AGP"]
        self.output_channel = output_channel
        self.channels.remove(self.output_channel)
        self.images_dir = images_dir
        self.metadata_df = metadata_df
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # https://discuss.pytorch.org/t/training-a-cnn-with-tiff-images-in-pytorch/9531
        dict_tensor = {channel: ToTensor()(Image.open(self.images_dir + self.metadata_df[channel].iloc[i])) for channel in self.channels}
        output_tensor = ToTensor()(Image.open(self.images_dir + self.metadata_df[self.output_channel].iloc[i]))
        # https://www.tutorialspoint.com/how-to-join-tensors-in-pytorch
        # https://stackoverflow.com/questions/54307225/whats-the-difference-between-torch-stack-and-torch-cat-functions
        # stack-joining on new dimension, cat-joining on existing dimension
        total_tensor = torch.stack(tuple(dict_tensor.values()), 1)

        return total_tensor, output_tensor

    def __len__(self):
        return self.metadata_df.shape[1]
