import torch
from torchvision.datasets import ImageFolder

class FreezeFrameDataset(ImageFolder):
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