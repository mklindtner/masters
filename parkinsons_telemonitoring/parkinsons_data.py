from parkinsons_telemonitoring.data.parkinsons_dataloaders import parkinsons_dataloaders 
from parkinsons_telemonitoring.parkinsons_model import SGLD, FFC_Regression_Parkinsons, U_s, BayesianRegressionParkin
import torch
import torch.nn as nn
import torch.optim as optim

# --- Default Hyperparameters ---
# These values are kept here as the defaults, as you wanted.
DEFAULT_BATCH_SIZE = 256
DEFAULT_TR_LR = 4e-6
DEFAULT_TAU = 10
DEFAULT_ST_DROPOUT = 0.5
DEFAULT_ST_LR_INIT = 1e-3
DEFAULT_BURNIN = 0
DEFAULT_H = 100
DEFAULT_T = 2000

DEFAULT_TR_POLY_A = 5e-6
DEFAULT_TR_POLY_DECAY_GAMMA = 0.55
DEFAULT_TR_POLY_LR_B = 3000

DEFAULT_OUTPUT_FOLDER = "parkinsons_telemonitoring/SGLD_testing"

def setup_experiment(batch_size, tau, st_dropout, st_lr_init, B, H, T, poly_a, poly_gamma, poly_b):

    print("--- Setting up PARKINSONS_TELEMONITORING with the following parameters: ---")
    print(f"  Batch Size: {batch_size}, Tau: {tau}")
    print(f"  Poly_a: {poly_a}, Poly_b: {poly_b}, Poly_gamma: {poly_gamma}")
    print(f"  Student Dropout: {st_dropout}, Student LR: {st_lr_init}")
    print(f" T: {T}, Burn-in: {B}, H: {H}")
    print("---------------------------------------------------------")

    trainloader, testloader, _, _ = parkinsons_dataloaders(batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    INPUT_FEATURES = 18
    tr_model = FFC_Regression_Parkinsons(input_size=INPUT_FEATURES, dropout_rate=0).to(device)

    # Teacher Hyper parameters
    N = len(trainloader.dataset)
    tr_criterion = nn.GaussianNLLLoss(reduction='sum', eps=1e-7)
    tr_optim = BayesianRegressionParkin(tr_model, n=N, m=batch_size, likelihood_criterion=tr_criterion, tau=tau)

    # Setup for distillation
    tr_list = [tr_optim, tr_model, trainloader, testloader]
    msc_list = [B, H, tr_criterion, device]
    tr_hyp_param = [poly_a, poly_gamma, poly_b]

    return tr_list, tr_hyp_param, msc_list

#Old setup_experiment w. constant lr that halves
# def setup_experiment(batch_size, tr_lr, tau, st_dropout, st_lr_init, B, H, T):

#     print("--- Setting up experiment with the following parameters: ---")
#     print(f"  Batch Size: {batch_size}, Teacher LR: {tr_lr}, Tau: {tau}")
#     print(f"  Student Dropout: {st_dropout}, Student LR: {st_lr_init}")
#     print(f" T: {T}, Burn-in: {B}, H: {H}")
#     print("---------------------------------------------------------")

#     trainloader, testloader, _, _ = parkinsons_dataloaders(batch_size)

#     device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#     print(f"Using device: {device}")

#     INPUT_FEATURES = 18
#     st_model = FFC_Regression_Parkinsons(input_size=INPUT_FEATURES, dropout_rate=st_dropout).to(device)
#     tr_model = FFC_Regression_Parkinsons(input_size=INPUT_FEATURES, dropout_rate=0).to(device)

#     # Teacher Hyper parameters
#     N = len(trainloader.dataset)
#     l2_regularization = tau
#     tr_optim = SGLD(tr_model.parameters(), lr=tr_lr, weight_decay=l2_regularization, N=N, M=batch_size)
#     tr_criterion = nn.GaussianNLLLoss(reduction='sum', eps=1e-7)

#     # Student hyper parameters
#     st_optim = optim.Adam(st_model.parameters(), lr=st_lr_init)
#     st_scheduler = optim.lr_scheduler.StepLR(st_optim, step_size=200, gamma=0.5)
#     tr_st_criterion = nn.KLDivLoss(reduction="batchmean")

#     # Setup for distillation
#     tr_list = [tr_optim, tr_model, trainloader, testloader]
#     st_list = [st_model, st_optim, st_scheduler, U_s, tr_st_criterion]
#     msc_list = [B, H, tr_criterion, device]
    
#     return tr_list, st_list, msc_list