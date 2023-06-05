import random
from torch.utils.data import Subset
import torch
from torchvision import transforms
from .dataset import TensorDataset, PictureDataset

def train_val_split(dataset, train_size:float=0.8):
    # Perform train/val split (on the original set)
    augmentation = dataset.augmentation
    all_indices = list(range(len(dataset) if not augmentation else int(len(dataset) / 2)))
    # Randomly split the remaining elements
    random.shuffle(all_indices)
    split_point = int(len(all_indices) * train_size)
    # divide train and validation indices (with augmentation if needed)
    if augmentation:
        train_indices = all_indices[:split_point] + [idx + dataset.real_length for idx in all_indices[:split_point]]
    else:
        train_indices = all_indices[:split_point]
    val_indices = all_indices[split_point:]
    # subset dataset
    train_dataset = Subset(dataset, indices = train_indices)
    val_dataset = Subset(dataset, indices = val_indices)

    return train_dataset, val_dataset

def normalise_distance(dataset):
    dist_min = torch.tensor([distance for _, _, distance, _ in dataset]).min()
    dist_max = torch.tensor([distance for _, _, distance, _ in dataset]).max()
    normalise = lambda x: (x - dist_min) / (dist_max - dist_min)

    return normalise

def normalise_angle(dataset):
    angle_min = torch.tensor([angle for _, _, _, angle in dataset]).min()
    angle_max = torch.tensor([angle for _, _, _, angle in dataset]).max()
    normalise = lambda x: (x - angle_min) / (angle_max - angle_min)

    return normalise

def load_model(version:str, in_channels:int=3, dropout:float=0.0):
    """Load the correct model based on the user's input.
    If path is specified, load a pretrained model.

    Args:
        version (str): model to load

    Returns:
        nn.Module: model used to encode the heatmaps
    """
    if version == 'v1':
        from .models.model_v1 import XGCNN
        model = XGCNN(dropout=dropout, in_channels=in_channels)
    else:
        raise ValueError(f'Architecture {version} not implemented.')

    return model

def load_tensor_dataset(angle, augmentation):
    data_path = 'tensors/shots.npy'
    labels_path = 'tensors/labels.npy'
    # Load the dataset
    return TensorDataset(data_path=data_path, labels_path=labels_path, angle=angle, augmentation=augmentation)


def load_picture_dataset(angle, augmentation):
    picture_type = 'angle' if angle else 'white'
    folder_path = f'images/{picture_type}'
    # Load the dataset
    return PictureDataset(folder_path, transform=transforms.ToTensor(), augmentation=augmentation)

def load_dataset(input_type, angle, augmentation):
    if input_type == 'tensor':
        dataset = load_tensor_dataset(angle, augmentation)
    else:
        dataset = load_picture_dataset(angle, augmentation)

    return dataset

def set_optimiser(model, optim, learning_rate, weight_decay):
    if optim.lower() == 'adam': 
        optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)  
    elif optim.lower() == 'adamw':
        optimiser = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=weight_decay) 
    elif optim.lower() == 'sgd':
        optimiser = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f'Specified optimiser {optim} not implemented. Should be one of ["adam", "adamw", "sgd"]')

    return optimiser