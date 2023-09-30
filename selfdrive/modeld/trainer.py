import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from simplecnn import SimpleCNN
import numpy as np
import cv2

element = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))

def skeletonize(img):
    """Compute the skeleton of a binary image using morphological operations."""
    skel = np.zeros(img.shape, np.uint8)
    size = np.size(img)
    
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        done = (cv2.countNonZero(img) == 0)
        
    return skel


class NPZDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), self.labels[idx]


# Create the model, criterion, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

base_dir = "dataset_segment/"
folders = ["straight", "left", "right"]

data = []
labels = []
for label, folder in enumerate(folders):
    npy_files = glob.glob(base_dir + folder + "/*.npy")
    color_imgs = glob.glob(base_dir + folder + "/*.jpg")
    
    for npy_file, clr_file in zip(npy_files, color_imgs):
        mask = np.load(npy_file).astype(np.uint8) # they will be zero and ones
        im = cv2.imread(clr_file)
        
        rgb_size = im.shape
        mask_size = mask.shape

        offset  = 20
        croped_mask = mask[mask_size[0] - rgb_size[0]: mask_size[0], rgb_size[1] + offset: rgb_size[1]*2 + offset]
        
        mask = cv2.dilate(croped_mask, element)
        sck = skeletonize(mask)
        sck = cv2.dilate(sck, element)

        data.append(np.expand_dims(sck, axis=0))
        labels.append(label)

data = np.array(data)
labels = np.array(labels)
# print(f'Shape of data is {data.shape}')
# print(f'Shape of data from torch -> {torch.from_numpy(data[1]).shape}')
# Splitting data into training and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(data))
val_size = len(data) - train_size
dataset = NPZDataset(data, labels)
# exit(1)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# Assuming the SimpleCNN model and other training components (criterion, optimizer) are already defined
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

torch.save(model.state_dict(), 'nav_model_hackathon_1.pth')
