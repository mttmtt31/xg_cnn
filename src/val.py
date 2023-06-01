import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def val(model, val_loader, device, epoch, criterion):
    model.eval()
    xGs = []
    outcomes = []

    with torch.no_grad():
        for images, labels, distance, angle in tqdm(val_loader, total = len(val_loader), desc = f'Validating epoch #{epoch+1}'):
            # send to device
            images, labels, distance, angle = images.to(device), labels.to(device), distance.to(device), angle.to(device)
            outputs = model(images, distance=distance, angle=angle)

            # append the xG
            xGs.append(outputs.squeeze())
            outcomes.append(labels.float())

        # calculate the roc-auc score
        xGs = torch.cat(xGs).cpu()
        outcomes = torch.cat(outcomes).cpu()
        roc_auc = roc_auc_score(outcomes, xGs)
        log_loss = criterion(outcomes, xGs)
        
    return roc_auc, log_loss