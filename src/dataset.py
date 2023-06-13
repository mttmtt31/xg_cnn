import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.datasets import ImageFolder
    
class TensorDataset(Dataset):
    def __init__(self, data_path, labels_path, angle, augmentation=None):
        # load the numpy arrays
        data = np.load(data_path)[:, int(not angle):, :, :]
        labels = np.load(labels_path)
        # turn them into tensors
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float()
        # check augmentation 
        self.augmentation = augmentation
        # find real length
        self.real_length = self.data.shape[0]

    def __len__(self):
        return self.real_length * (self.augmentation + 1)
    
    def __getitem__(self, index):
        # check if augmentation is on
        if index >= self.real_length:
            # locate tensor at the right index
            tensor = self.data[index - self.real_length]
            # perform augmentation
            tensor = tensor.flip(dims = [2])
            # locate label at the right index
            label = self.labels[index - self.real_length]
        else:
            # locate tensor 
            tensor = self.data[index]
            # locate label
            label = self.labels[index]

        return tensor, label

class PictureDataset(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, augmentation:bool=False):
        super().__init__(root, transform, target_transform)
        self.augmentation = augmentation
        self.real_length = len(self.samples)

    def __getitem__(self, index):
        # check if augmentation is on
        if index >= self.real_length:
            path, target = self.samples[index - self.real_length]
        else:
            # retrieve image and label
            path, target = self.samples[index]
        image = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)

        # perform data augmentation
        if index >= self.real_length:
            image = torch.flip(image, [-1]) 

        if self.target_transform is not None:
            target = self.target_transform(target)

        # target is reserved, so that goal->1, non_goal->0
        return image, 1 - target

    def __len__(self):
        return len(self.samples)*(self.augmentation+1)