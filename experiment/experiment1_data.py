import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from experiment.experiment1_models import SGLD
import torch
from experiment.experiment1_models import FFC_Regression, U_s
import numpy as np
import random

#For reproducabilæity I'll just set the seed everywhere
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Internet says 
# mean = 0.1307, std = 0.3081.
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
])
BATCH_SIZE = 128

trainset = torchvision.datasets.MNIST(root='./data',
                                      train=True,
                                      download=True,
                                      transform=transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data',
                                     train=False,
                                     download=True,
                                     transform=transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


device = None

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Apple MPS is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("No GPU available. Using CPU.")


#(A.2 section "Model and Distillation Hyper-Parameters")


INPUT_FEATURES = 784  
OUTPUT_FEATURES = 10

st_droput = 0.5
st_model = FFC_Regression(input_size=INPUT_FEATURES, output_size=OUTPUT_FEATURES,dropout_rate=st_droput).to(device)
tr_model = FFC_Regression(input_size=INPUT_FEATURES, output_size=OUTPUT_FEATURES, dropout_rate=0).to(device)


#Msc Parameters
criterion = nn.CrossEntropyLoss()
T = int(1e6)
B = 1000
H = 100

#teacher parameters 
tr_lr = 4e-6
tau = 10
N = 60000 
M = BATCH_SIZE 
#I think the 1/2 cancels here because the gra
# l2_regularization = 1/2 * tau 
l2_regularization = tau
tr_optim = SGLD(tr_model.parameters(), lr=tr_lr, weight_decay=l2_regularization, N=N)


#Student parameters
st_lr_init = 1e-3


st_optim = optim.Adam(st_model.parameters(), lr=st_lr_init)
step_128_bz = 93600 #1 epoch = 60k/128 = 468; 468*200 = half learning = 93600
st_scheduler = optim.lr_scheduler.StepLR(st_optim, step_size=step_128_bz, gamma=0.5)
tr_st_criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)

#Setup relevant lists
msc_items = [B,H, criterion, device]
st_items = [st_model, st_optim, st_scheduler, U_s, tr_st_criterion]
tr_items = [tr_optim, tr_model, trainloader, testloader]





