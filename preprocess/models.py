import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherFCNNMNIST(nn.Module):
    def __init__(self):
        super(TeacherFCNNMNIST, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 400) 
        self.fc3 = nn.Linear(400, 10)  

    def forward(self, x):
        x = x.view(-1, 784)  
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)  # No activation, since we'll use CrossEntropyLoss
        return x

class StudentFCNNMNIST(nn.Module):
    def __init__(self, K1=1, K2=1):
        super(StudentFCNNMNIST, self).__init__()
        self.fc1 = nn.Linear(784, int(400 * K1))  
        self.fc2 = nn.Linear(int(400 * K1), int(400 * K2))  
        self.fc3 = nn.Linear(int(400 * K2), 10)  

    def forward(self, x):
        x = x.view(-1, 784) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class TeacherCNNMNIST(nn.Module):
    def __init__(self):
        super(TeacherCNNMNIST, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=4, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2)  
        
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=4, stride=1)
        
        self.fc1 = nn.Linear(20 * 5 * 5, 80)  
        self.fc2 = nn.Linear(80, 10) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(-1, 20 * 5 * 5) 
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)  # Output layer (no activation, since we use CrossEntropyLoss)
        return x


class StudentCNN_MNIST(nn.Module):
    def __init__(self, K1=1, K2=1):
        super(StudentCNN_MNIST, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=int(10 * K1), kernel_size=4, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=int(10 * K1), out_channels=int(20 * K1), kernel_size=4, stride=1)
        
        self.fc1 = nn.Linear(int(20 * K1) * 5 * 5, int(80 * K2))
        self.fc2 = nn.Linear(int(80 * K2), 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, int(20 * self.K1) * 5 * 5)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




class TeacherCIFAR10_CNN(nn.Module):
    def __init__(self):
        super(TeacherCIFAR10_CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)  # Max pooling layer (2x2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        
        self.fc1 = nn.Linear(32 * 5 * 5, 200)  
        self.fc2 = nn.Linear(200, 50) 
        self.fc3 = nn.Linear(50, 10) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(-1, 32 * 5 * 5)  
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)  
        return x

class StudentCNNCIFAR10(nn.Module):
    def __init__(self, K1=1, K2=1):
        super(StudentCNNCIFAR10, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=int(16 * K1), kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=int(16 * K1), out_channels=int(16 * K1), kernel_size=5)
        
        self.fc1 = nn.Linear(int(16 * K1) * 5 * 5, int(200 * K2))
        self.fc2 = nn.Linear(int(200 * K2), int(50 * K2))
        self.fc3 = nn.Linear(int(50 * K2), 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, int(16 * self.K1) * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

