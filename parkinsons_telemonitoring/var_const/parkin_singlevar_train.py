import argparse
from datetime import datetime
from parkinsons_telemonitoring.data.parkinsons_dataloaders import parkinsons_dataloaders 
import torch
from torch import nn
import itertools
from tqdm.auto import tqdm
import torch.nn.functional as F
import math
from constants import path_parkin_stat
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import FuncFormatter



def plot_results_bayers(results_data, timestamp, hp, output_dir):   
    if not results_data:
        print("No data available to plot.")
        return

    if not output_dir:
        print("no output dir specified, cancelling")
        return

    print("\n--- Plotting Results ---")     

    df = pd.DataFrame(results_data)

    t_steps = df['t']
    teacher_nll_val = df['tr_nll_val']
    teacher_nll_train = df['tr_nll_train']
    window_size = 10
    teacher_nll_train_smooth = teacher_nll_train.rolling(window=window_size).mean()
    teacher_nll_val_smooth = teacher_nll_val.rolling(window=window_size).mean()
    
    title_str = (
        f"Telemonitoring Dataset\n"
        f"(sigmasq: {hp['tr_var']}) | ({hp['iterations']} iterations) | (Poly Decay: a={hp['tr_poly_a']:.2e}, b={hp['tr_poly_b']:.0f}, γ={hp['tr_poly_gamma']}) | "
        f"(Data: M={hp['batch_size']}, τ={hp['tau']})"
    )

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    ax.plot(t_steps, teacher_nll_val_smooth, marker='o', linestyle='-', label=f'Teacher NLL Validation (Smoothed, w={window_size})')
    # ax.plot(t_steps, teacher_nll_train, marker='o', color="pink", linestyle='None', alpha=0.3, label='Teacher NLL Train (Raw)')
    ax.plot(t_steps, teacher_nll_train_smooth, color="red", linestyle='-', label=f'Teacher NLL Train (Smoothed, w={window_size})')
    

    #it looks ugly af if I dont make some sort of writing on the x-axis
    num_iterations = hp.get('iterations', t_steps.max())
    tick_locations = np.linspace(0, num_iterations, 11)

    def k_formatter(x, pos):
        if x == 0:
            return '0'
        elif x == 1_000_000:
             return '1M'
        return f'{int(x*1e-3)}k'
    
    ax.set_xticks(tick_locations)
    ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))

    ax.set_title(title_str)
    ax.set_xlabel('Training Iterations (t)') # Corrected label
    ax.set_ylabel('Average Negative Log-Likelihood (NLL)')
    ax.grid(True)
    ax.legend()

    plot_filename = (
        f"PARKINSONS_{timestamp}_"
        f"T={hp['iterations']}_"
        f"var={hp['tr_var']}"
        f"M={hp['batch_size']}_"
        f"tau={hp['tau']}_"
        f"a={hp['tr_poly_a']:.1e}_"
        f"b={hp['tr_poly_b']}_"
        f"g={hp['tr_poly_gamma']}.png"
    )
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, plot_filename)
    
    plt.savefig(full_path)
    print(f"Plot saved to {full_path}")


def save_results_to_csv_bayers(results_data, hp, timestamp, output_dir):
    if not results_data:
        print("No results to save.")
        return None
    
    if not output_dir:
        print("no output dir specified, cancelling")
        return

    print("\n--- Saving Results to CSV ---")
    
    results_df = pd.DataFrame(results_data)    
    results_df = results_df.assign(**hp)
    

    csv_filename = (
        f"PARKINSONS_{timestamp}_"
        f"T={hp['iterations']}_"
        f"M={hp['batch_size']}_"
        f"tau={hp['tau']}_"
        f"a={hp['tr_poly_a']:.1e}_"
        f"b={hp['tr_poly_b']}_"
        f"g={hp['tr_poly_gamma']}.csv"
    )

    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, csv_filename)
    results_df.to_csv(full_path, index=False)

    print(f"Results saved to {full_path}")
    
    print("\nResults Preview (with hyperparameters):")
    print(results_df.head())
    
    return results_df



def save_results_to_csv(results_data, timestamp, T, results_dir=path_parkin_stat):
    if not results_data:
        print("No results to save.")
        return None

    print("\n--- Saving Results to CSV ---")
    
    results_df = pd.DataFrame(results_data)
    
    csv_filename = f"distillation_results_{timestamp}_T={T}.csv"
    os.makedirs(results_dir, exist_ok=True)
    full_path = os.path.join(results_dir, csv_filename)
    
    results_df.to_csv(full_path, index=False)
    print(f"Results saved to {full_path}")
    
    print("\nResults Preview:")
    print(results_df.head())
    
    return results_df

class FCC_Regression_Parkinsons_Mean(nn.Module):
    def __init__(self, input_size, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 400)
        self.fc2 = nn.Linear(400, 400)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc3_mean = nn.Linear(400, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))      
        # x = self.dropout(x)  
        x = F.relu(self.fc2(x))        
        # x = self.dropout(x)        
        mean = self.fc3_mean(x)
        return mean


class BayesianRegressionParkin():
    
    def __init__(self, f, n,m, likelihood_criterion, sigmasq, tau=10):
        self.tau = tau
        self.lik_criterion = likelihood_criterion
        self.f = f
        self.N = n
        self.M = m
        self.sigmasq = sigmasq
        

    def log_prior(self):
        W = torch.cat([w.view(-1) for w in self.f.parameters()])
        W_sq = torch.dot(W, W)
        return -1/2 * self.tau * W_sq


    def log_likelihood(self, x,y):
        mean = self.f(x)
        return -self.lik_criterion(input=mean, target=y, var=self.sigmasq)

    def log_joint(self, x,y):
        return self.log_prior() + self.log_likelihood(x,y)
    
    def log_joint_gradient(self, x,y):        

        #prior: analytical grad
        w_grad_prior_list = [-self.tau* w.data.view(-1) for w in self.f.parameters()]
        w_grad_prior = torch.cat([w.view(-1) for w in w_grad_prior_list])

        #likelihood: autodiff grad
        self.f.zero_grad()
        w_grad_likelihood_loss = self.log_likelihood(x,y)
        w_grad_likelihood_loss.backward()
        w_grad_likelihood = torch.cat([w.grad.view(-1) for w in self.f.parameters()])

        return w_grad_prior + self.N/self.M * w_grad_likelihood

    
    def sgld_step(self, x, y, lr):
        grad = self.log_joint_gradient(x,y)

        with torch.no_grad():
            noise = torch.randn_like(grad) * math.sqrt(lr)
            w_deltas = lr/2 * grad + noise

            offset = 0
            for w in self.f.parameters():
                wslice = w.view(-1).shape[0]
                w_delta = w_deltas[offset : offset+wslice].view(w.shape)  
                w.add_(w_delta)
                offset += wslice


def validate_network(model, validation_loader, criterion, device, sigmasq, verbose=True):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    val_loop = tqdm(validation_loader, desc="Validating", leave=False, disable=not verbose)
    
    with torch.no_grad():
        for inputs, labels in val_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            
            mean = model(inputs)
            loss = criterion(input=mean, target=labels, var=sigmasq)
            
            total_loss += loss.item()
            total_samples += len(labels)
            
    return total_loss / total_samples


def bayesian_distillation_parkin(tr_items, msc_items, tr_hyp_par, T_total=1e6, verbose=True):
    tr_bayers, tr_network, tr_loader_train, tr_loader_valid, tr_var = tr_items
    B, H, eval_criterion, device = msc_items
    lr_init, decay_gamma, lr_b = tr_hyp_par
    train_iterator = itertools.cycle(tr_loader_train)
    results = []


    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

    print(f"--- Starting Distillation Process for {T_total} steps ---")
    
    T = tqdm(range(T_total), desc="Total Steps", disable=not verbose, bar_format=bar_format)

    for t in T:
        lr = lr_init * (lr_b + t)**(-decay_gamma)
        T.set_postfix(LR=f"{lr:.2e}") 
        inputs, labels = next(train_iterator)
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(inputs.size(0), -1)

        #The tr_network is inside tr_bayers
        tr_bayers.sgld_step(inputs, labels, lr)

        if t >= B and (t % H == 0):
            with torch.no_grad():            
                mean = tr_network(inputs)
                tr_nll_train = eval_criterion(input=mean, target=labels, var=tr_var)
                tr_nll_avg = 1/len(labels) * tr_nll_train.item()

                tr_network.eval()  
                teacher_nll = validate_network(tr_network, tr_loader_valid, eval_criterion, device, sigmasq=tr_var, verbose=False)
                
            results.append({
                't': t + 1,
                'tr_nll_val': teacher_nll,
                'tr_nll_train': tr_nll_avg,
            })        

            tr_network.train()
    return results, tr_network.state_dict()


# --- Default Hyperparameters ---
DEFAULT_BATCH_SIZE = 256
DEFAULT_TR_LR = 4e-6
DEFAULT_TAU = 10
DEFAULT_ST_DROPOUT = 0.5
DEFAULT_ST_LR_INIT = 1e-3
DEFAULT_BURNIN = 0
DEFAULT_H = 100
DEFAULT_T = 2000

DEFAULT_SIGMASQ = 0.055

DEFAULT_TR_POLY_A = 5e-6
DEFAULT_TR_POLY_DECAY_GAMMA = 0.55
DEFAULT_TR_POLY_LR_B = 3000


DEFAULT_OUTPUT_FOLDER = "parkinsons_telemonitoring/var_const/singlevar_runs/testing"

def setup_experiment(batch_size, tau, st_dropout, st_lr_init, B, H, T, poly_a, poly_gamma, poly_b, sigmasq):

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
    tr_model = FCC_Regression_Parkinsons_Mean(input_size=INPUT_FEATURES, dropout_rate=0).to(device)

    # Teacher Hyper parameters
    N = len(trainloader.dataset)
    tr_criterion = nn.GaussianNLLLoss(reduction='sum', eps=1e-7)
    tr_optim = BayesianRegressionParkin(tr_model, n=N, m=batch_size, likelihood_criterion=tr_criterion, tau=tau, sigmasq=sigmasq)

    # Setup for distillation
    tr_list = [tr_optim, tr_model, trainloader, testloader, sigmasq]
    msc_list = [B, H, tr_criterion, device]
    tr_hyp_param = [poly_a, poly_gamma, poly_b]

    return tr_list, tr_hyp_param, msc_list

def main(args):
    
    tr_list, tr_hyp_param, msc_list = setup_experiment(
        batch_size=args.batch_size,
        tau=args.tau,
        st_dropout=args.st_dropout,
        st_lr_init=args.st_lr_init,
        B=args.B,
        H=args.H,
        T=args.iterations,
        poly_a=args.tr_poly_a,
        poly_gamma=args.tr_poly_gamma,
        poly_b=args.tr_poly_b,
        sigmasq=args.tr_var
    )

    if args.output_dir == None:
        print("no output dir given, exiting")
        return
    
    #new2
    results, _ = bayesian_distillation_parkin(tr_list, msc_items=msc_list, tr_hyp_par=tr_hyp_param, T_total=args.iterations)
    hp_dict = vars(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_results_bayers(results_data=results, timestamp=timestamp, hp=hp_dict, output_dir=args.output_dir)
    save_results_to_csv_bayers(results_data=results, hp=hp_dict, timestamp=timestamp, output_dir=args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Parkinsons Telemonitoring distillation experiment.')

    parser.add_argument('--iterations', type=int, default=DEFAULT_T, help='Total training iterations')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size (M)')
    parser.add_argument('--tau', type=float, default=DEFAULT_TAU, help='Precision of the prior (tau)')
    parser.add_argument('--st_dropout', type=float, default=DEFAULT_ST_DROPOUT, help='Student dropout rate')
    parser.add_argument('--st_lr_init', type=float, default=DEFAULT_ST_LR_INIT, help='Student initial learning rate')
    parser.add_argument('--B', type=int, default=DEFAULT_BURNIN, help='Burn-in period')
    parser.add_argument('--H', type=int, default=DEFAULT_H, help='Distillation frequency')
    
    parser.add_argument('--tr_var', type=float, default=DEFAULT_SIGMASQ, help='Distillation frequency')
    
    parser.add_argument('--tr_poly_a', type=float, default=DEFAULT_TR_POLY_A, help='(Polynomial decay): Initial Teacher learning rate')
    parser.add_argument('--tr_poly_gamma',  type=float, default=DEFAULT_TR_POLY_DECAY_GAMMA, help='(Polynomial decay): gamma decay')
    parser.add_argument('--tr_poly_b', type=float, default=DEFAULT_TR_POLY_LR_B, help='(Polynomial decay): b decay')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_FOLDER, help='Directory to save all run artifacts')



    args = parser.parse_args()
    main(args)
