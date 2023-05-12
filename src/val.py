import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def val(model, val_loader, device, epoch):
    model.eval()
    xGs = []
    outcomes = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, total = len(val_loader), desc = f'Validating epoch #{epoch+1}'):
            # send to device
            images, labels = images.to(device), labels.to(device)
            # append the xG
            xGs.append(model(images).squeeze())
            outcomes.append(labels)

        # calculate the roc-auc score
        xGs = torch.cat(xGs).cpu()
        outcomes = torch.cat(outcomes).cpu()
        score = roc_auc_score(outcomes, xGs)
        
    return score