import torch
from torchvision import datasets

class FreezeFrameDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, augmentation:bool=False):
        super().__init__(root, transform, target_transform)
        self.augmentation = augmentation

    def __getitem__(self, index):
        # check if augmentation is on
        if self.augmentation:
            index = index // 2
        # retrieve image and label
        path, target = self.samples[index]
        image = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)

        # perform data augmentation
        if self.augmentation:
            # check if index is even or odd. If odd -> flip along y-axis
            if index % 2 == 1:
                image = torch.flip(image, [-1]) 

        if self.target_transform is not None:
            target = self.target_transform(target)

        # target is reserved, so that goal->1, non_goal->0
        return image, 1 - target

    def __len__(self):
        return len(self.samples)*(self.augmentation+1)
            