# parkinsons_SGLD_ablation.py

import argparse
import os
import json
import torch
import pandas as pd
import numpy as np # Import numpy for calculating the mean
import matplotlib.pyplot as plt # Import matplotlib for plotting
from datetime import datetime

# Import your project's functions and default values
from parkinsons_telemonitoring.parkinsons_data import (
    setup_experiment, 
    DEFAULT_BATCH_SIZE, 
    DEFAULT_TR_LR, 
    DEFAULT_TAU, 
    DEFAULT_BURNIN,
    DEFAULT_H,
    DEFAULT_T
)
from parkinsons_telemonitoring.parkinsons_model import train_teacher_network

# --- NEW: Plotting function ---
def plot_nll_results(log_results, hyperparams, output_path):
    """
    Generates and saves a plot of the validation NLL over iterations.
    """
    # Extract only the data points where validation was performed
    val_logs = [log for log in log_results if 'tr_gauss_nll_loss_val' in log]
    if not val_logs:
        print("No validation data to plot.")
        return

    iterations = [log['t_val'] for log in val_logs]
    nll_values = [log['tr_gauss_nll_loss_val'] for log in val_logs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, nll_values, color='blue', label='Validation NLL')
    
    # Create a descriptive title from the hyperparameters
    title = (f"NLL vs. Iterations\n"
             f"LR={hyperparams['tr_lr']}, Batch Size={hyperparams['batch_size']}, Weight Decay={hyperparams['weight_decay']}")
    plt.title(title)
    
    plt.xlabel("Iterations")
    plt.ylabel("NLL")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot to the specified path
    plt.savefig(output_path)
    plt.close() # Close the plot to free up memory

# --- MODIFIED: Artifact saving function now includes plotting ---
def save_trial_artifacts(log_results, teacher_weights, hyperparams, base_dir):
    """Saves all artifacts for a single trial into its dedicated directory."""
    print(f"--- Saving artifacts to: {base_dir} ---")
    
    # 1. Save detailed time-series log to CSV
    log_df = pd.DataFrame(log_results)
    log_df.to_csv(os.path.join(base_dir, "detailed_log.csv"), index=False)
    
    # 2. Save final teacher weights
    weights_path = os.path.join(base_dir, "teacher_weights.pth")
    torch.save(teacher_weights, weights_path)

    # 3. Generate and save the NLL plot (NEW)
    plot_path = os.path.join(base_dir, "nll_plot.png")
    plot_nll_results(log_results, hyperparams, plot_path)
    
    print("--- Artifacts saved successfully ---")

def main(args):
    """Main function to run a single SGLD ablation trial."""
    
    # 1. Create a unique directory for this specific trial
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}_T_{args.T}_lr_{args.tr_lr}_bs_{args.batch_size}_wd_{args.weight_decay}"
    cwd = os.getcwd() 
    # Define the relative path to your ablation results folder
    ablation_base_path = "parkinsons_telemonitoring/SGLD_ablation"
    # Join them to create a full, absolute path
    output_dir = os.path.join(cwd, ablation_base_path, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Setup and run the experiment
    tr_list, _, msc_list = setup_experiment(
        batch_size=args.batch_size, tr_lr=args.tr_lr, tau=args.weight_decay,
        st_dropout=0.5, st_lr_init=1e-3, B=args.B, H=DEFAULT_H, T=args.T
    )

    tr_optim, tr_network, tr_loader_train, tr_loader_valid = tr_list
    _, _, tr_criterion, device = msc_list
    tr_eval = len(tr_loader_train)

    log_results, final_tr_w = train_teacher_network(
        tr_optim=tr_optim, tr_network=tr_network, T_steps=args.T,
        tr_loader_train=tr_loader_train, tr_loader_valid=tr_loader_valid,
        criterion=tr_criterion, device=device, tr_eval=tr_eval
    )
    
    # 3. Save all artifacts, now passing hyperparameters for the plot title
    hyperparams = vars(args) # Convert argparse namespace to a dictionary
    save_trial_artifacts(log_results, final_tr_w, hyperparams, output_dir)

    # --- MODIFIED: Calculate post-burn-in average NLL ---
    # 4. Filter for validation results after the burn-in period
    post_burn_in_nlls = [
        log['tr_gauss_nll_loss_val'] for log in log_results 
        if 'tr_gauss_nll_loss_val' in log and log['t_val'] > args.B
    ]
    
    # Calculate the average, handling the case where there are no results
    avg_nll = np.mean(post_burn_in_nlls) if post_burn_in_nlls else float('nan')
    print(f"Average NLL after burn-in (t > {args.B}): {avg_nll:.4f}")

    # 5. Save the final summary for this trial
    summary_data = {
        'T-B': args.T - args.B,
        'batch_size': args.batch_size,
        'tr_lr': args.tr_lr,
        'weight_decay': args.weight_decay,
        'burn_in_B': args.B,
        'avg_nll_post_burn_in': avg_nll, # The new metric
        'timestamp': timestamp,
        'output_dir': output_dir
    }
    
    summary_filepath = os.path.join(output_dir, "summary.json")
    with open(summary_filepath, 'w') as f:
        json.dump(summary_data, f, indent=4)
    
    print(f"Final summary for this trial saved to {summary_filepath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SGLD ablation trial.')

    parser.add_argument('--T', type=int, default=DEFAULT_T, help='Total training iterations.')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size (M).')
    parser.add_argument('--tr_lr', type=float, default=DEFAULT_TR_LR, help='Teacher learning rate.')
    #weight_decay = tau
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_TAU, help='L2 regularization / prior precision (tau).')
    parser.add_argument('--B', type=int, default=DEFAULT_BURNIN, help='Burn-in period.')

    args = parser.parse_args()
    main(args)