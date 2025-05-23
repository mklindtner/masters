import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import random
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import numpy as np


def MNIST_distillation_data(data_dir="./data", batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    

    return train_loader, test_loader


def MNIST_teacher_data(sub=60000):
    #Test that the data is like tehy specified in the paper
    train_loader = torch.load(f"./data/MNIST/manipulated/manipulated_train_loader_{sub}.pth")
    test_loader = torch.load(f"./data/MNIST/manipulated/manipulated_train_loader_{sub}.pth")
    return train_loader, test_loader

def CIFAR10_teacher_data(sub=50000):
    train_loader = torch.load(f"./data/CIFAR10/manipulated/manipulated_train_loader_{sub}.pth")
    test_loader = torch.load(f"./data/CIFAR10/manipulated/manipulated_train_loader_{sub}.pth")
    return train_loader, test_loader


def data_preprocess_cifar10(data_dir="./data",batch_size=64,sub=60000, save = True):
    #Subsampling for train, always use full dataset for 
    allowed_sizes = {10000, 20000, 50000}
    if sub not in allowed_sizes:
        raise ValueError(f"Must specifiy size of {allowed_sizes} for subsamples")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,transform=transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    
    indices = np.random.choice(len(train_dataset), sub, replace=False)  # Random indices
    train_dataset_sub = Subset(train_dataset, indices)  # Create a subset

    # Data loader
    train_loader = DataLoader(train_dataset_sub, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  

    if save:
        # Save the loaders for reuse
        os.makedirs(data_dir, exist_ok=True)
        torch.save(train_loader, os.path.join(data_dir, f"CIFAR10/manipulated/manipulated_train_loader_{sub}.pth"))
        torch.save(test_loader, os.path.join(data_dir, f"CIFAR10/manipulated/manipulated_test_loader_{sub}.pth"))
        print(f"Data saved at {data_dir}")

    return train_loader,test_loader 


def data_preprocess_mnist(data_dir="./data", batch_size=64, save=True, sub = 60000, should_mask=True):
    # Define transformations for MNIST
    # This might be unnecessary and take a large amount of time
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and prepare MNIST dataset
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)


    #Subsampling for train, always use full dataset for 
    allowed_sizes = {10000, 20000, 30000, 60000}
    if sub not in allowed_sizes:
        raise ValueError(f"Must specifiy size of {allowed_sizes} for subsamples")

    indices = np.random.choice(len(train_dataset), sub, replace=False)  # Random indices
    train_dataset_sub = Subset(train_dataset, indices)  # Create a subset

    # Data loader
    train_loader = DataLoader(train_dataset_sub, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    
    

    #Mask Data
    #We want a random patch of 14x14 and put it to -1
    if should_mask:
        random.seed(42)
        m = 14
        r = m*m/(28*28) #percentage of image masked

        manipulated_train = []
        labels = []
        for images, label in train_loader:
            
            rng_row = torch.randint(0,28-m,(1,))
            rng_col = torch.randint(0,28-m,(1,))
            for i in range(images.shape[0]):
                #just to make which image im manipulating clear
                image = images[i].clone()
                image[rng_row:rng_row+m,rng_col:rng_col+m] = -1
                manipulated_train.append(image)
                labels.append(label[i])
            #labels.append(label)
        man_img = torch.stack(manipulated_train)
        man_lab = torch.stack(labels)
        man_dataset = TensorDataset(man_img, man_lab)
        train_loader = DataLoader(man_dataset, batch_size=batch_size, shuffle=True)

    #Visual test if manipulation was done correct
    # tmp = next(iter(man_train_loader))[0][0].view(28,28)
    # tmp[rng_row:rng_row+m,rng_col:rng_col+m] = -1
    # # torch.save(tmp, "./data/tmp.pt")
    # plt.imshow(tmp, cmap="gray")
    # plt.axis("off")  # Hide axes
    # plt.show()
    # print("showing image")


    #When training I should probably store the images and load them for so it'll go faster
    if save:
        # Save the loaders for reuse
        os.makedirs(data_dir, exist_ok=True)
        torch.save(train_loader, os.path.join(data_dir, f"MNIST/manipulated/manipulated_train_loader_{sub}.pth"))
        torch.save(test_loader, os.path.join(data_dir, f"MNIST/manipulated/manipulated_test_loader_{sub}.pth"))
        print(f"Data saved at {data_dir}")



    return train_loader, test_loader



if __name__ == "__main__":
    # data_preprocess_mnist(sub=60000)
    # MNIST_teacher_data(sub=20000)
    data_preprocess_cifar10(sub=20000)
