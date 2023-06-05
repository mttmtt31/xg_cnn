import torch.nn as nn
from src import FreezeFrameDataset, train, val, train_val_split, load_model, set_optimiser
from torch.utils.data import DataLoader
import wandb
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type = int, default = 64)
    parser.add_argument('--learning-rate', type = float, default = 0.01)
    parser.add_argument('--dropout', type = float, default = 0.0)
    parser.add_argument('--epochs', type = int, default = 10)
    parser.add_argument('--version', type = str, default = 'v1')
    parser.add_argument('--augmentation', action='store_true', help = 'Whether you want to perform data augmentation')
    parser.add_argument('--wandb', action='store_true', help = 'Whether you want to log results in wandb')
    parser.add_argument('--optim', type = str, default = 'adam', help = 'Optimiser to use')
    parser.add_argument('--weight-decay', type = float, default = 0.0, help = 'Weight decay')
    parser.add_argument('--angle', action = 'store_true', default = 'Whether to consider a channel for the shot angle')

    return parser.parse_args()

def main(device, batch_size, lr, num_epochs, log_wandb, augmentation, angle, dropout, version, optimiser, wd):
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
            "angle" : angle,
            "dropout" : dropout,
            "version" : version,
            "augmentation" : augmentation,
            "optimiser" : optimiser,
            "weight_decay" : wd
            }
        )

    data_path = 'data/shots.npy'
    labels_path = 'data/labels.npy'
    # Load the dataset
    dataset = FreezeFrameDataset(data_path=data_path, labels_path=labels_path, angle=angle, augmentation=augmentation)

    # Split train/val dataset
    train_dataset, val_dataset = train_val_split(dataset=dataset, train_size=0.8)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = load_model(version=version, dropout=dropout, in_channels=3 if angle else 2)
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = set_optimiser(model=model, optim=optimiser, learning_rate=lr, weight_decay=wd)

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
        batch_size=args.batch_size,
        lr=args.learning_rate,
        log_wandb=args.wandb,
        num_epochs=args.epochs,       
        augmentation=args.augmentation,
        dropout=args.dropout,
        version=args.version,
        optimiser=args.optim,
        wd = args.weight_decay,
        angle = args.angle
    )
