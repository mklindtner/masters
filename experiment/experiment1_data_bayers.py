import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from experiment.experiment1_models import FFC_Regression, BayesianRegression
import numpy as np
import random
import torch.optim as optim

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
        transforms.Lambda(lambda x: x*(255.0/ 126.0))
    ])


    trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
    trainloader = DataLoader(trainset, batch_size=bsz, shuffle=True, num_workers=0)
    testset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)
    testloader = DataLoader(testset, batch_size=bsz, shuffle=False, num_workers=0)

    return trainloader, testloader

#cat1 hyp par
DEFAULT_T = int(2e5)
DEFAULT_H = 100
DEFAULT_B = 2000
DEFAULT_TAU = 45
DEFAULT_BATCH_SIZE = 100
DEFAULT_N = 60000

#cat2 TR poly
DEFAULT_TR_POLY_INIT_LR = 4e-6
DEFAULT_TR_POLY_DECAY_GAMMA = 0.55
DEFAULT_TR_POLY_LR_B = 17.0



def setup_experiment(batch_size, tau, N, H):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"running on: {device}")
    trainloader, testloader = MNIST_dataloaders(batch_size)

    INPUT_FEATURES = 784  
    OUTPUT_FEATURES = 10
    
    tr_model = FFC_Regression(input_size=INPUT_FEATURES, output_size=OUTPUT_FEATURES, dropout_rate=0).to(device)
    tr_bayers = BayesianRegression(f=tr_model, n=N, m=batch_size, tau=tau)

    st_model = FFC_Regression(input_size=INPUT_FEATURES, output_size=OUTPUT_FEATURES, dropout_rate=0.5).to(device)
    tr_st_criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
    
    st_lr_init = 1e-3 
    st_prior_precision = 0.001
    st_optim = optim.Adam(st_model.parameters(), lr=st_lr_init, weight_decay=st_prior_precision)

    #Every 200 epochs as described in the paper
    steps_per_epoch = N / batch_size
    scheduler_step_size_in_teacher_iterations = (200 * steps_per_epoch) * H
    st_scheduler = optim.lr_scheduler.StepLR(st_optim, step_size=int(scheduler_step_size_in_teacher_iterations), gamma=0.5)

  
    # Use a criterion with 'mean' reduction for validation, as it's a metric
    val_criterion = nn.CrossEntropyLoss(reduction='mean')
    tr_items = [tr_bayers, tr_model, trainloader, testloader]
    st_items = [st_model, st_optim, st_scheduler, tr_st_criterion]

    
    return tr_items, st_items, val_criterion, device
