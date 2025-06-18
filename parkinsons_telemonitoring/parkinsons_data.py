from parkinsons_telemonitoring.data.parkinsons_dataloaders import parkinsons_dataloaders 
from parkinsons_telemonitoring.parkinsons_model import SGLD, FFC_Regression_Parkinsons, U_s
import torch
import torch.nn as nn
import torch.optim as optim

BATCH_SIZE = 64

trainloader, testloader, scaler_X, scaler_y = parkinsons_dataloaders(BATCH_SIZE)

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

INPUT_FEATURES = 18
st_droput = 0.5
st_model = FFC_Regression_Parkinsons(input_size=INPUT_FEATURES, dropout_rate=st_droput).to(device)
tr_model = FFC_Regression_Parkinsons(input_size=INPUT_FEATURES, dropout_rate=0).to(device)


#Msc hyper parameters
T = int(1e5)
B = 2000
H = 100


#teacher Hyper parameters
tr_lr = 4e-6
tau = 10
N = len(trainloader.dataset)
M = BATCH_SIZE
l2_regularization = tau 
tr_optim = SGLD(tr_model.parameters(), lr=tr_lr, weight_decay=l2_regularization, N=N, M=BATCH_SIZE)
tr_criterion = nn.GaussianNLLLoss(reduction='sum', eps=1e-7)
tr_eval = int(len(trainloader))


#Student hyper parameters
st_lr_init = 1e-3

st_optim = optim.Adam(st_model.parameters(), lr=st_lr_init)
st_scheduler = optim.lr_scheduler.StepLR(st_optim, step_size=200, gamma=0.5)
tr_st_criterion = nn.KLDivLoss(reduction="batchmean")

#Setup for distillation
tr_list = [tr_optim, tr_model, trainloader, testloader]
st_list = [st_model, st_optim, st_scheduler, U_s, tr_st_criterion]
msc_list = [B,H, tr_criterion, device]