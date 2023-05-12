from tqdm import tqdm
import torch

def train(train_loader, model, epoch, device, optimizer, criterion):
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

    # average the loss to get the training loss
    train_loss = running_loss / len(train_loader)
    # compute the accuracy
    train_acc = correct / total

    return model, train_loss, train_acc
