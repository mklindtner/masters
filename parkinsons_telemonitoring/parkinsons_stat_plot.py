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
    teacher_nll_val_smooth = teacher_nll_val.rolling(window_window_size).mean()
    
    title_str = (
        f"Parkinsons Teacher GaussNLL: ({hp['iterations']} iterations)\n"
        f"(Poly Decay: a={hp['tr_poly_a']:.2e}, b={hp['tr_poly_b']:.0f}, γ={hp['tr_poly_gamma']}) | "
        f"(Data: M={hp['batch_size']}, τ={hp['tau']})"
    )

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    ax.plot(t_steps, teacher_nll_val_smooth, marker='o', linestyle='-', label='Teacher NLL Validation (Smoothed, w={window_size})')
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



def store_weights(tr_w, st_w, timestamp, T, weights_dir=path_parkin_weights):
    student_weights_path = os.path.join(weights_dir, f"final_student_weights_{timestamp}_T={T}.pth")
    teacher_weights_path = os.path.join(weights_dir, f"final_teacher_weights_{timestamp}_T={T}.pth")

    print(f"Saving final student weights to: {student_weights_path}")
    torch.save(st_w, student_weights_path)
    
    print(f"Saving final teacher weights to: {teacher_weights_path}")
    torch.save(tr_w, teacher_weights_path)