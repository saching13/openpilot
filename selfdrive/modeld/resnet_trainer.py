from torchvision import models
import torch
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import cv2

# Define transformations - normalization values are based on pre-trained models from torchvision
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, labels transform=None):
        self.dataset = original_dataset
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image_tensor = torch.from_numpy(self.data[idx]).float()
        image_tensor = image_tensor.permute(2, 0, 1)

        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, self.labels[idx]


# Load pre-trained ResNet (e.g., ResNet-50)
model = models.resnet50(pretrained=True)

# Modify the final layer to have 3 classes
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 3)


criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

data = []
labels = []
for label, folder in enumerate(folders):
    folderSubs =glob.glob(base_dir + folder + "/*")
    # classCount = [0, 0, 0]
    # print(f'Analyzing class {folder}')
    for subFolder in folderSubs:
        print(subFolder)
        npy_files = glob.glob(subFolder + "/*.npy")
        color_imgs = glob.glob(subFolder + "/*.jpg")
        npy_files.sort()
        color_imgs.sort()
    
        for npy_file, clr_file in zip(npy_files, color_imgs):
            mask = np.load(npy_file).astype(np.uint8) # they will be zero and ones
            im = cv2.imread(clr_file)
            
            rgb_size = im.shape
            mask_size = mask.shape

            offset  = 20
            croped_mask = mask[mask_size[0] - rgb_size[0]: mask_size[0], rgb_size[1] + offset: rgb_size[1]*2 + offset]
            
            # mask = cv2.dilate(croped_mask, element)
            # sck = skeletonize(croped_mask)
            # cv2.imshow("mask", croped_mask * 255)
            # cv2.imshow('img', im)
            # cv2.waitKey(0)
            # im = im.astype(np.float64)
            # im = im / 255.0
            # data.append(np.expand_dims(im, axis=0))
            data.append(im)
            labels.append(label)

data = np.array(data)
labels = np.array(labels)

dataset = TransformedDataset(original_dataset=data, transform=transform)
train_size = int(0.8 * len(data))
val_size = len(data) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

num_epochs = 50

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

torch.save(model.state_dict(), 'resnet_hackathon_org_data.pth')
