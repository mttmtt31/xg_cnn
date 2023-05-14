import random
from torch.utils.data import Subset

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