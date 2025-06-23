import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from experiment.experiment1_models import FFC_Regression, BayesianRegression
import numpy as np
import random

#For reproducabilæity I'll just set the seed everywhere
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)



def MNIST_dataloaders(bsz):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
    trainloader = DataLoader(trainset, batch_size=bsz, shuffle=True, num_workers=0)
    testset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)
    testloader = DataLoader(testset, batch_size=bsz, shuffle=False, num_workers=0)

    return trainloader, testloader

#cat1 hyp par
DEFAULT_T = int(1e5)
DEFAULT_H = 100
DEFAULT_B = 1000
DEFAULT_TAU = 15
DEFAULT_BATCH_SIZE = 256
DEFAULT_N = 60000

#cat2 TR poly
DEFAULT_TR_POLY_INIT_LR = 4e-6
DEFAULT_TR_POLY_DECAY_GAMMA = 0.55
DEFAULT_TR_POLY_LR_B = 17.0


def setup_experiment(batch_size, tau, N):
    print("--- Setting up experiment with the following parameters: ---")
    print(f"  Data size (N): {N}, Batch Size (M): {batch_size}, Prior Precision (tau): {tau}")
    print("---------------------------------------------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    trainloader, testloader = MNIST_dataloaders(batch_size)

    INPUT_FEATURES = 784  
    OUTPUT_FEATURES = 10
    
    tr_model = FFC_Regression(input_size=INPUT_FEATURES, output_size=OUTPUT_FEATURES, dropout_rate=0).to(device)
    tr_bayers = BayesianRegression(f=tr_model, n=N, m=batch_size, tau=tau)
    
    # Use a criterion with 'mean' reduction for validation, as it's a metric
    val_criterion = nn.CrossEntropyLoss(reduction='mean')
    
    return tr_bayers, tr_model, trainloader, testloader, val_criterion, device
