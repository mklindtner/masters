import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import random

def load_data(data_dir="./data", batch_size=64, save=True):
    # Define transformations for MNIST
    # This might be unnecessary and take a large amount of time
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and prepare MNIST dataset
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    


    #Mask Data
    random.seed(42)
    m = 14
    r = m*m/(28*28) #percentage of image masked
    #For one
        #rng = random.randrange(0,27-m)
        #tv[0][rng+m:,rng+m:] = 0
    #Do masking operation on the entire set


    #

    # if save:
    #     # Save the loaders for reuse
    #     os.makedirs(data_dir, exist_ok=True)
    #     torch.save(train_loader, os.path.join(data_dir, "train_loader.pth"))
    #     torch.save(test_loader, os.path.join(data_dir, "test_loader.pth"))
    #     print(f"Data saved at {data_dir}")



    return train_loader, test_loader



if __name__ == "__main__":
    load_data()
