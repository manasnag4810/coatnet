# Training loop script
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from utils.utils import dice_score, jaccard_index

class StrokeDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images  # List of image tensors
        self.masks = masks    # List of mask tensors

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for img, mask in tqdm(dataloader):
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        pred = model(img)
        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    dice_total, jaccard_total = 0, 0
    with torch.no_grad():
        for img, mask in dataloader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            dice_total += dice_score(pred, mask)
            jaccard_total += jaccard_index(pred, mask)
    return dice_total / len(dataloader), jaccard_total / len(dataloader)

def train_model(model, train_loader, val_loader, epochs, lr, device, save_path):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        dice, jaccard = validate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Dice: {dice:.4f} | Jaccard: {jaccard:.4f}")

        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch{epoch+1}.pth"))
