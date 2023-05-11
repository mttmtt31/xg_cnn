import torch
import torch.nn as nn
import torch.optim as optim
from src.model import SimpleCNN
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type = int, default = 64)
    parser.add_argument('--learning-rate', type = float, default = 0.001)
    parser.add_argument('--epochs', type = int, default = 10)
    parser.add_argument('--picture_type', type = str, choices = ['white', 'all', 'visible', 'cones'], help = 'Path to folder where pictures are stored. Should be one of (white, all, visible, cones).', default='white')
    parser.add_argument('--wandb', action='store_false', help = 'Whether you want to log results in wandb')

    return parser.parse_args()

def main(device, batch_size, lr, num_epochs, picture_type, log_wandb):
    # Specify transformation for loading the images
    transform = transforms.Compose([
        transforms.Resize((210, 140)),
        transforms.ToTensor(),
    ])
    folder_path = f'images/{picture_type}'

    # Load the dataset
    dataset = ImageFolder(folder_path, transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = SimpleCNN()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    model.to(device)

    for epoch in range(num_epochs):
        # sets into training mode
        model.train()
        # initialise parameter to track the training performance
        running_loss = 0.0
        correct = 0
        total = 0

        # loop over the images
        for images, labels in tqdm(train_loader, total = len(train_loader), desc = f'Training epoch #{epoch+1}'):
            # send them to the device
            images, labels = images.to(device), labels.to(device)
            # zero the gradients
            optimizer.zero_grad()
            # forward pass
            outputs = model(images)
            # loss
            loss = criterion(outputs, labels)
            # backpropagation
            loss.backward()
            # optimisation
            optimizer.step()

            # track loss
            running_loss = running_loss + loss.item()
            # find all predicted labels
            _, predicted = torch.max(outputs.data, 1)
            # find the total number of predictions
            total = total + labels.size(0)
            # check how many correct
            correct = correct + (predicted == labels).sum().item()
            break

        # average the loss to get the training loss
        train_loss = running_loss / len(train_loader)
        # compute the accuracy
        train_acc = correct / total

        # evaluate the model on the validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                # Calculate the loss
                loss = criterion(outputs, labels)
                val_loss = val_loss + loss.item()

                # Track the total and correct predictions
                _, predicted = torch.max(outputs.data, 1)
                total = total + labels.size(0)
                correct = correct + (predicted == labels).sum().item()

            # Calculate average loss and accuracy on the validation set
            val_loss = val_loss / len(val_loader)
            val_acc = correct / total

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
                "picture_type" : picture_type
                }
            )
            # log in wandb
            wandb.log({"Train Loss": train_loss, "Train Accuracy": train_acc, "Validation Loss": val_loss, "Validation Accuracy": val_acc})

if __name__ == '__main__':
    args = parse_args()
    main(
        device=args.device,
        picture_type=args.picture_type,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        log_wandb=args.wandb,
        num_epochs=args.epochs        
    )
