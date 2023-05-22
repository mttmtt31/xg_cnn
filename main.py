import torch.nn as nn
import torch.optim as optim
from src import FreezeFrameDataset, train, val, train_val_split, load_model
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type = int, default = 32)
    parser.add_argument('--learning-rate', type = float, default = 0.001)
    parser.add_argument('--dropout', type = float, default = 0.0)
    parser.add_argument('--epochs', type = int, default = 10)
    parser.add_argument('--version', type = str, default = 'v2')
    parser.add_argument('--picture-type', type = str, choices = ['white', 'all', 'visible', 'cones', 'angle'], help = 'Path to folder where pictures are stored. Should be one of (white, all, visible, cones).', default='white')
    parser.add_argument('--augmentation', action='store_true', help = 'Whether you want to perform data augmentation')
    parser.add_argument('--wandb', action='store_true', help = 'Whether you want to log results in wandb')

    return parser.parse_args()

def main(device, batch_size, lr, num_epochs, picture_type, log_wandb, augmentation, dropout, version):
    if log_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="xg-cnn",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": lr,
            "epochs": num_epochs,
            "batch_size" : batch_size, 
            "learning_rate" : lr, 
            "num_epochs" : num_epochs, 
            "picture_type" : picture_type,
            "dropout" : dropout,
            "version" : version,
            "augmentation" : augmentation
            }
        )
    # Specify transformation for loading the images
    transform = transforms.Compose([
        transforms.Resize((93, 140)),
        transforms.ToTensor(),
    ])
    folder_path = f'images/{picture_type}'

    # Load the dataset
    dataset = FreezeFrameDataset(folder_path, transform=transform, augmentation=augmentation)

    # Split train/val dataset
    train_dataset, val_dataset = train_val_split(dataset=dataset, train_size=0.8)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = load_model(version=version, dropout=dropout)
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Train the model
    for i in range(num_epochs):
        model, train_loss = train(train_loader=train_loader, model=model, epoch=i, device=device, optimizer=optimizer, criterion=criterion)

        # evaluate the model on the validation set
        roc_score, log_loss = val(model=model, val_loader=val_loader, device=device, epoch=i, criterion=criterion)

        # log in wandb
        if log_wandb:
            wandb.log({"Train Loss": train_loss, "Validation ROC-AUC score:" : roc_score, "Validation loss" : log_loss})

if __name__ == '__main__':
    args = parse_args()
    main(
        device=args.device,
        picture_type=args.picture_type,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        log_wandb=args.wandb,
        num_epochs=args.epochs,       
        augmentation=args.augmentation,
        dropout=args.dropout,
        version=args.version
    )
