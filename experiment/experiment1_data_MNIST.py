import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from experiment.experiment1_models import SGLD
import torch
from experiment.experiment1_models import FFC_Regression, U_s, BayesianRegression

##Student parameters these actually worked
# st_droput = 0.5
# st_model = FFC_Regression(input_size=INPUT_FEATURES, output_size=OUTPUT_FEATURES,dropout_rate=st_droput).to(device)
# st_lr_init = 1e-3
# st_optim = optim.Adam(st_model.parameters(), lr=st_lr_init)
# st_scheduler = optim.lr_scheduler.StepLR(st_optim, step_size=step_128_bz, gamma=0.5)
# tr_st_criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
# st_items = [st_model, st_optim, st_scheduler, U_s, tr_st_criterion]

EXP_MNIST_T = 1e6
EXP_MNIST_B = 1000
EXP_MNIST_H = 100
EXP_MNIST_TAU = 10.0
EXP_MNIST_TR_LR = 4e-6
EXP_MNIST_DATASZ = 60000
EXP_MNIST_BZ = 256


def setup_distillation_experiment(batch_size):
    print("--- Setting up experiment with fixed hyperparameters ---")
    
    #(A.2 section "Model and Distillation Hyper-Parameters")
    N = EXP_MNIST_DATASZ
    T_total = int(1e6) 
    B = EXP_MNIST_B
    H = EXP_MNIST_H
    tau = EXP_MNIST_TAU      
    tr_lr = EXP_MNIST_TR_LR    
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    INPUT_FEATURES = 784
    OUTPUT_FEATURES = 10
    tr_model = FFC_Regression(input_size=INPUT_FEATURES, output_size=OUTPUT_FEATURES, dropout_rate=0).to(device)    
    tr_optim = SGLD(tr_model.parameters(), lr=tr_lr, weight_decay=tau, N=N)    
    criterion = nn.CrossEntropyLoss(reduction='mean')

    #Later
    #1 epoch = 60k/128 = 468; 468*200 = half learning = 93600
    # step_128_bz = T_total * 0.096
    # tr_scheduler = optim.lr_scheduler.StepLR(tr_optim, step_size=step_128_bz, gamma=0.5)

    tr_items = [tr_optim, tr_model, train_loader, val_loader]
    msc_items = [B, H, criterion, device]
    
    return tr_items, msc_items, T_total