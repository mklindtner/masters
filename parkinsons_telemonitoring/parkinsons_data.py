from parkinsons_telemonitoring.data.parkinsons_dataloaders import parkinsons_dataloaders 
from parkinsons_telemonitoring.parkinsons_model import FFC_Regression_Parkinsons, BayesianRegressionParkin, MeanVarianceStudentTrainer, VarianceOnlyStudentTrainer, StudentVarOnly, StudentMeanOnly, MeanOnlyStudentTrainer
import torch
import torch.nn as nn
import torch.optim as optim

# --- Default Hyperparameters ---
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

DEFAULT_VAL_STEP = 5000

DEFAULT_OUTPUT_FOLDER = "parkinsons_telemonitoring/SGLD_testing"

def setup_experiment(batch_size, tau, st_dropout, st_lr_init, B, H, T, poly_a, poly_gamma, poly_b, student_mode, val_step):

    print("--- Setting up PARKINSONS_TELEMONITORING with the following parameters: ---")
    print(f"  Batch Size: {batch_size}, Tau: {tau}")
    print(f"  Poly_a: {poly_a}, Poly_b: {poly_b}, Poly_gamma: {poly_gamma}")
    print(f"  Student Dropout: {st_dropout}, Student LR: {st_lr_init}")
    print(f" T: {T}, Burn-in: {B}, H: {H}, val_step: {val_step}")
    print("---------------------------------------------------------")

    trainloader, testloader, _, _ = parkinsons_dataloaders(batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    INPUT_FEATURES = 18

    # Teacher Setup
    tr_model = FFC_Regression_Parkinsons(input_size=INPUT_FEATURES, dropout_rate=0).to(device)
    N = len(trainloader.dataset)
    tr_criterion = nn.GaussianNLLLoss(reduction='sum', eps=1e-7)
    tr_optim = BayesianRegressionParkin(tr_model, n=N, m=batch_size, likelihood_criterion=tr_criterion, tau=tau)
    tr_list = [tr_optim, tr_model, trainloader, testloader]

    #Student Setup
    st_network = None
    st_optim = None

    if student_mode == 'mean_and_variance':
        print("Initializing student for Mean and Variance.")
        st_network = FFC_Regression_Parkinsons(input_size=INPUT_FEATURES, dropout_rate=st_dropout).to(device)
        st_optim = optim.Adam(st_network.parameters(), lr=st_lr_init)
        student_trainer = MeanVarianceStudentTrainer(st_network, st_optim)

    elif student_mode == 'variance_only':
        print("Initializing student for Variance Only.")
        # This student still needs two outputs, but only one is trained
        st_network = StudentVarOnly(input_size=INPUT_FEATURES, dropout_rate=st_dropout).to(device)
        st_optim = optim.Adam(st_network.parameters(), lr=st_lr_init)
        student_trainer = VarianceOnlyStudentTrainer(st_network, st_optim)

    elif student_mode == 'mean_only':
        print("Initializing student for Mean Only.")
        st_network = StudentMeanOnly(input_size=INPUT_FEATURES, dropout_rate=st_dropout).to(device)
        st_optim = optim.Adam(st_network.parameters(), lr=st_lr_init)
        student_trainer = MeanOnlyStudentTrainer(st_network, st_optim)

    else:
        raise ValueError(f"Unknown student mode: {student_mode}")
    
    st_list = [st_network, st_optim, student_trainer] 

    #MSC Setup
    msc_list = [B, H, tr_criterion, device]
    tr_hyp_param = [poly_a, poly_gamma, poly_b]

    return tr_list, st_list, tr_hyp_param, msc_list

