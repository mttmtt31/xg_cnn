import torch
from tqdm import tqdm

def val(model, val_loader, device, criterion, epoch):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, total = len(val_loader), desc = f'Validating epoch #{epoch+1}'):
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

    return val_loss, val_acc