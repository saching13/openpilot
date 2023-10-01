# Define the Neural Network
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

element = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))

def skeletonize(img):
    """Compute the skeleton of a binary image using morphological operations."""
    skel = np.zeros(img.shape, np.uint8)
    size = np.size(img)
    img = cv2.dilate(img, element)
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        done = (cv2.countNonZero(img) == 0)
        skel = cv2.dilate(skel, element)
    return skel

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # Input has 1 channel (grayscale)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 21 * 19, 512)  # After 3 pooling layers, the size is 170/8 x 153/8
        self.fc2 = nn.Linear(512, 3)



    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 21 * 19)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        logits = self.fc2(x)
        probabilities = F.softmax(logits, dim=1)
        return probabilities
