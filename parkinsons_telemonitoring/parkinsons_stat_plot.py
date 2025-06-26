import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from constants import path_parkin_stat, path_parkin_fig, path_parkin_weights
import torch
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
    
    title_str = (
        f"Parkinsons Teacher GaussNLL: ({hp['iterations']} iterations)\n"
        f"(Poly Decay: a={hp['tr_poly_a']:.2e}, b={hp['tr_poly_b']:.0f}, γ={hp['tr_poly_gamma']}) | "
        f"(Data: M={hp['batch_size']}, τ={hp['tau']})"
    )

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    ax.plot(t_steps, teacher_nll_val, marker='o', linestyle='-', label='Teacher NLL Validation')
    ax.plot(t_steps, teacher_nll_train, marker='o', color="pink", linestyle='None', alpha=0.3, label='Teacher NLL Train (Raw)')
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


def plot_results(results_data, timestamp, T, figs_dir=path_parkin_fig):
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

    plt.title('Network Validation NLL\n ({T} iterations)')
    plt.xlabel('Student Training Steps (t)')
    plt.ylabel('Average Negative Log-Likelihood')
    plt.grid(True)
    plt.legend()
    
    # --- Save the plot with a unique filename ---
    plot_filename = f'distillation_nll_plot_{timestamp}_T={T}.png'
    full_path = os.path.join(figs_dir, plot_filename)
    
    plt.savefig(full_path)
    print(f"Plot saved to {full_path}")



def plot_tr_results_teacher(results_data, timestamp, T, figs_dir=path_parkin_fig, label="teacher_nll_plot"):
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

    plot_filename = f'{label}_{timestamp}_T={T}.png'
    full_path = os.path.join(figs_dir, plot_filename)
    
    fig.savefig(full_path, dpi=150 )
    print(f"Plot saved to {full_path}")


def plot_tr_results_distillation(results_data, timestamp, T, figs_dir=path_parkin_fig, label="parkin_tele"):
    teacher_nlls = [r['tr_nll'] for r in results_data]
    kl_divergences = [r['tr_st_kl'] for r in results_data]
    t = np.arange(len(teacher_nlls))

    plt.style.use('seaborn-v0_8-whitegrid')

    fig1, ax1 = plt.subplots(figsize=(12, 7))

    ax1.plot(t, teacher_nlls, label="Teacher Validation NLL", color='blue', marker='o', linestyle='-')
    ax1.set_title(f'Parkinsons Telmonitoring Teacher NLL:\n ({T} iterations)', fontsize=16)
    ax1.set_ylabel('Average Gaussian NLL', fontsize=12)
    ax1.set_xlabel(f'Student Distillation iterations\n ({len(t)} iterations)', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_filename_nll = f'{label}_NLL_{timestamp}_T={T}.png'
    full_path_nll = os.path.join(figs_dir, plot_filename_nll)
    fig1.savefig(full_path_nll, dpi=150)
    print(f"Teacher NLL plot saved to {full_path_nll}")


    fig2, ax2 = plt.subplots(figsize=(12, 7))

    ax2.plot(t, kl_divergences, label="KL Divergence (Teacher || Student)\n ({T} iterations)", color='green', marker='.', linestyle='-')
    ax2.set_title(f'(Parkinsons) Posterior Distillation Expectation\n ({T} iterations)', fontsize=14)
    ax2.set_ylabel('KL Divergence', fontsize=12)
    ax2.set_xlabel(f'Student Distillation Step\n ({len(t)})', fontsize=12) 
    ax2.legend(loc='best')
    # ax2.set_ylim((0,20))
    ax2.grid(True, linestyle='--', alpha=0.7)


    plt.tight_layout()
    plot_filename_kl = f'{label}_KL_{timestamp}_T={T}.png'
    full_path_kl = os.path.join(figs_dir, plot_filename_kl)
    fig2.savefig(full_path_kl, dpi=150)
    print(f"Student KL plot saved to {full_path_kl}")
    
    plt.show()


def store_weights(tr_w, st_w, timestamp, T, weights_dir=path_parkin_weights):
    student_weights_path = os.path.join(weights_dir, f"final_student_weights_{timestamp}_T={T}.pth")
    teacher_weights_path = os.path.join(weights_dir, f"final_teacher_weights_{timestamp}_T={T}.pth")

    print(f"Saving final student weights to: {student_weights_path}")
    torch.save(st_w, student_weights_path)
    
    print(f"Saving final teacher weights to: {teacher_weights_path}")
    torch.save(tr_w, teacher_weights_path)