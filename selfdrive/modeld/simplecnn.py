# Define the Neural Network
import torch.nn as nn
import torch.nn.functional as F

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
