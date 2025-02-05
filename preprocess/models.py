import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherFCNNMNIST(nn.Module):
    def __init__(self):
        super(TeacherFCNNMNIST, self).__init__()
        self.fc1 = nn.Linear(784, 400)  # First fully connected layer
        self.fc2 = nn.Linear(400, 400)  # Second fully connected layer
        self.fc3 = nn.Linear(400, 10)   # Output layer (10 classes for MNIST)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the image to a 1D vector
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = F.relu(self.fc2(x))  # Apply ReLU activation
        x = self.fc3(x)  # No activation, since we'll use CrossEntropyLoss
        return x

class StudentFCNN(nn.Module):
    def __init__(self, K1=1, K2=1):
        super(StudentFCNN, self).__init__()
        self.fc1 = nn.Linear(784, int(400 * K1))  # First fully connected layer
        self.fc2 = nn.Linear(int(400 * K1), int(400 * K2))  # Second fully connected layer
        self.fc3 = nn.Linear(int(400 * K2), 10)  # Output layer (10 classes)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class TeacherCNNMNIST(nn.Module):
    def __init__(self):
        super(TeacherCNNMNIST, self).__init__()
        
        # First convolutional layer: 10 filters, kernel size 4x4, stride 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=4, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2)  # Max pooling layer (2x2)
        
        # Second convolutional layer: 20 filters, kernel size 4x4, stride 1
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=4, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(20 * 5 * 5, 80)  # Flattened conv output to 80 neurons
        self.fc2 = nn.Linear(80, 10)  # Output layer (10 classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + MaxPool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + MaxPool
        x = x.view(-1, 20 * 5 * 5)  # Flatten the feature maps
        x = F.relu(self.fc1(x))  # Fully connected layer with ReLU
        x = self.fc2(x)  # Output layer (no activation, since we use CrossEntropyLoss)
        return x


class StudentCNN_MNIST(nn.Module):
    def __init__(self, K1=1, K2=1):
        super(StudentCNN_MNIST, self).__init__()
        
        # First convolutional layer: 10*K1 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=int(10 * K1), kernel_size=4, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Second convolutional layer: 20*K1 filters
        self.conv2 = nn.Conv2d(in_channels=int(10 * K1), out_channels=int(20 * K1), kernel_size=4, stride=1)
        
        # Fully connected layer
        self.fc1 = nn.Linear(int(20 * K1) * 5 * 5, int(80 * K2))
        self.fc2 = nn.Linear(int(80 * K2), 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, int(20 * self.K1) * 5 * 5)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




class TeacherCIFAR10_CNN(nn.Module):
    def __init__(self):
        super(TeacherCIFAR10_CNN, self).__init__()
        
        # First convolutional layer: 16 filters, kernel size 5x5
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)  # Max pooling layer (2x2)
        
        # Second convolutional layer: 32 filters, kernel size 5x5
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 5 * 5, 200)  # Flattened conv output to 200 neurons
        self.fc2 = nn.Linear(200, 50)  # Fully connected layer with 50 neurons
        self.fc3 = nn.Linear(50, 10)  # Output layer (10 classes for CIFAR-10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + MaxPool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + MaxPool
        x = x.view(-1, 32 * 5 * 5)  # Flatten the feature maps
        x = F.relu(self.fc1(x))  # Fully connected layer with ReLU
        x = F.relu(self.fc2(x))  # Fully connected layer with ReLU
        x = self.fc3(x)  # Output layer (no activation, since we use CrossEntropyLoss)
        return x

class StudentCNNCIFAR10(nn.Module):
    def __init__(self, K1=1, K2=1):
        super(StudentCNNCIFAR10, self).__init__()

        # First convolutional layer: 16*K1 filters
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=int(16 * K1), kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Second convolutional layer: 16*K1 filters
        self.conv2 = nn.Conv2d(in_channels=int(16 * K1), out_channels=int(16 * K1), kernel_size=5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(int(16 * K1) * 5 * 5, int(200 * K2))
        self.fc2 = nn.Linear(int(200 * K2), int(50 * K2))
        self.fc3 = nn.Linear(int(50 * K2), 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, int(16 * self.K1) * 5 * 5)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

