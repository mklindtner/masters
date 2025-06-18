import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
from constants import path_parkin_stat, path_parkin_fig


def save_results_to_csv(results_data, results_dir=path_parkin_stat):
    if not results_data:
        print("No results to save.")
        return None

    print("\n--- Saving Results to CSV ---")
    
    results_df = pd.DataFrame(results_data)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"distillation_results_{timestamp}.csv"
    os.makedirs(results_dir, exist_ok=True)
    full_path = os.path.join(results_dir, csv_filename)
    
    results_df.to_csv(full_path, index=False)
    print(f"Results saved to {full_path}")
    
    print("\nResults Preview:")
    print(results_df.head())
    
    return results_df


def plot_results(results_data, figs_dir=path_parkin_fig):
    if not results_data:
        print("No data available to plot.")
        return

    print("\n--- Plotting Results ---")
    
    # //New
    # Extract the data for plotting from the list of dictionaries.
    t_steps = [record['t'] for record in results_data]
    teacher_nll = [record.get('tr_nll') for record in results_data]
    student_nll = [record.get('st_nll') for record in results_data]

    plt.figure(figsize=(12, 7))

    if any(nll is not None for nll in teacher_nll):
        plt.plot(t_steps, teacher_nll, marker='o', linestyle='-', label='Teacher NLL')


    if any(nll is not None for nll in student_nll):
        plt.plot(t_steps, student_nll, marker='x', linestyle='--', label='Student NLL')

    plt.title('Network Validation NLL vs. Training Step')
    plt.xlabel('Training Step (t)')
    plt.ylabel('Average Negative Log-Likelihood')
    plt.grid(True)
    plt.legend()
    
    # --- Save the plot with a unique filename ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'distillation_nll_plot_{timestamp}.png'
    full_path = os.path.join(figs_dir, plot_filename)
    
    plt.savefig(full_path)
    print(f"Plot saved to {full_path}")



def plot_tr_results_teacher(results_data, figs_dir=path_parkin_fig, label="teacher_nll_plot"):
    train_steps = range(len(results_data))
    gauss_nll_train = [r['tr_gauss_nll_loss_train'] for r in results_data]
    
    val_steps = []
    gauss_nll_val = []
    for record in results_data:
        if 'tr_gauss_nll_loss_val' in record:
            val_steps.append(record['t_val'])
            gauss_nll_val.append(record['tr_gauss_nll_loss_val'])
            
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(train_steps, gauss_nll_train, label='Training Loss (per batch)', color='cyan', alpha=0.7, linewidth=1)

    ax.plot(val_steps, gauss_nll_val, label='Validation Loss (per epoch)', color='blue', marker='o', linestyle='-', linewidth=2,markersize=6)

    ax.set_title('Teacher Network: Training & Validation Loss', fontsize=16)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Average Gaussian NLL', fontsize=12)
    ax.legend(fontsize=12)

    fig.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'{label}_{timestamp}.png'
    full_path = os.path.join(figs_dir, plot_filename)
    
    fig.savefig(full_path, dpi=150 )
    print(f"Plot saved to {full_path}")


def plot_tr_results_distillation(results_data, figs_dir=path_parkin_fig, label="nll_kl_plots"):
    teacher_nlls = [r['tr_nll'] for r in results_data]
    kl_divergences = [r['tr_st_kl'] for r in results_data]
    t = np.arange(len(teacher_nlls))
    t_total = len(t)

    plt.style.use('seaborn-v0_8-whitegrid')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig1, ax1 = plt.subplots(figsize=(12, 7))

    ax1.plot(t, teacher_nlls, label="Teacher Validation NLL", color='blue', marker='o', linestyle='-')
    ax1.set_title(f'Parkinsons Telmonitoring\n Teacher Performance NLL: {t_total}-steps', fontsize=16)
    ax1.set_ylabel('Average Gaussian NLL', fontsize=12)
    ax1.set_xlabel('Student Distillation Step', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_filename_nll = f'{label}_NLL_{timestamp}.png'
    full_path_nll = os.path.join(figs_dir, plot_filename_nll)
    fig1.savefig(full_path_nll, dpi=150)
    print(f"Teacher NLL plot saved to {full_path_nll}")


    fig2, ax2 = plt.subplots(figsize=(12, 7))

    ax2.plot(t, kl_divergences, label="KL Divergence (Teacher || Student)", color='green', marker='.', linestyle='-')
    ax2.set_title(f'Parkinsons telemonitoring\n Posterior Distillation Expectation: {t_total}-steps', fontsize=14)
    ax2.set_ylabel('KL Divergence', fontsize=12)
    ax2.set_xlabel('Studen Distillaiton Step', fontsize=12) 
    ax2.legend(loc='best')
    # ax2.set_ylim((0,20))
    ax2.grid(True, linestyle='--', alpha=0.7)


    plt.tight_layout()
    plot_filename_kl = f'{label}_KL_{timestamp}.png'
    full_path_kl = os.path.join(figs_dir, plot_filename_kl)
    fig2.savefig(full_path_kl, dpi=150)
    print(f"Student KL plot saved to {full_path_kl}")
    
    plt.show()
