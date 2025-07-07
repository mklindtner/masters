# experiment1_stat_plot.py

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
from constants import path_exp_fig, path_exp_stat, path_exp_weights, path_exp_fig_sens_figs, path_exp_fig_sens_stat, path_exp_MNIST_default
import torch
from matplotlib.ticker import FuncFormatter




def save_results_to_csv_bayers(results_data, hp, timestamp, output_dir=path_exp_fig_sens_stat):
    if not results_data:
        print("No results to save.")
        return None

    print("\n--- Saving Results to CSV ---")
    results_df = pd.DataFrame(results_data)    
    final_validation_accuracy = results_df['tr_acc_val'].iloc[-1]

    final_error_rate_pct = (1 - final_validation_accuracy) * 100
    
    hp_with_summary = hp.copy()
    hp_with_summary['final_error_rate_pct'] = final_error_rate_pct
    results_df = results_df.assign(**hp_with_summary)
    

    csv_filename = (
        f"MNIST_{timestamp}_"
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
    print(f"Final Validation Error Rate: {final_error_rate_pct:.3f}%")
    
    print("\nResults Preview (with hyperparameters):")
    print(results_df.head())
    
    return results_df


def plot_teacher_performance(df, hp, output_dir, timestamp):
    """Plots the teacher's training and validation NLL, with accuracy in the title."""
    t_steps = df['t']
    tr_nll_val = df['tr_nll_val']
    tr_nll_train = df['tr_nll_train']
    
    # Smooth the noisy training curve for better visualization
    window_size = 5
    tr_nll_train_smooth = tr_nll_train.rolling(window=window_size).mean()

    # Get final accuracy values for the title, with checks for existence
    final_val_acc = df['tr_acc_val'].iloc[-1] if 'tr_acc_val' in df.columns and not df['tr_acc_val'].empty else 0
    final_train_acc = df['tr_acc_train'].iloc[-1] if 'tr_acc_train' in df.columns and not df['tr_acc_train'].empty else 0

    title_str = (
        f"Teacher Performance (Final Err Rate Train/Val: {(1-final_train_acc)*100:.2f}% / {(1-final_val_acc)*100:.2f}%)\n"
        f"Run Parameters: T={hp['iterations']}, M={hp['batch_size']}, τ={hp['tau']}"
    )

    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    
    ax.plot(t_steps, tr_nll_val, marker='o', markersize=4, linestyle='-', label='Teacher Validation NLL')
    ax.plot(t_steps, tr_nll_train, marker='o', markersize=4, color="pink", linestyle='None', alpha=0.3, label='Teacher Train NLL (Raw Batch)')
    ax.plot(t_steps, tr_nll_train_smooth, color="red", linestyle='-', label=f'Teacher Train NLL (Smoothed, w={window_size})')



    def k_formatter(x, pos):
        """Formats ticks into human-readable k/M/B units."""
        x = int(x)
        if x >= 1e6:
            # For millions, format as '1.5M' or '2M'
            return f'{x/1e6:.1f}'.replace('.0', '') + 'M'
        elif x >= 1e3:
            # For thousands, format as '1.5k' or '10k'
            return f'{x/1e3:.1f}'.replace('.0', '') + 'k'
        return x

    num_iterations = hp.get('iterations', t_steps.max())
    tick_locations = np.linspace(0, num_iterations, 11)
    ax.set_xticks(tick_locations)
    ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))

    ax.set_title(title_str)
    ax.set_xlabel("Training Iterations (t)")
    ax.set_ylabel("Average Negative Log-Likelihood (NLL)")
    ax.legend()
    ax.grid(True)
    
    # Save the figure
    filename = f"teacher_performance_{timestamp}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()
    print(f"Teacher performance plot saved to {os.path.join(output_dir, filename)}")

# --- Helper Function 2: Distillation NLL Comparison ---
def plot_distillation_nll(df, hp, output_dir, timestamp):
    """Plots and compares the validation NLL of the teacher and student."""
    if 'st_nll_val' not in df.columns: 
        print("Student NLL data not found, skipping distillation plot.")
        return

    t_steps = df['t']
    tr_nll_val = df['tr_nll_val']
    st_nll_val = df['st_nll_val']

    # Get final accuracy values for the title
    final_tr_acc = df['tr_acc_val'].iloc[-1] if 'tr_acc_val' in df.columns and not df['tr_acc_val'].empty else 0
    final_st_acc = df['st_acc_val'].iloc[-1] if 'st_acc_val' in df.columns and not df['st_acc_val'].empty else 0

    title_str = (
        f"Distillation Performance (Final Err Rate Teacher/Student: {(1-final_tr_acc)*100:.2f}% / {(1-final_st_acc)*100:.2f}%)\n"
        f"Run Parameters: T={hp['iterations']}, M={hp['batch_size']}, τ={hp['tau']}"
    )

    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    
    ax.plot(t_steps, tr_nll_val, marker='o', markersize=4, linestyle='-', label='Teacher Validation NLL')
    ax.plot(t_steps, st_nll_val, marker='s', markersize=4, linestyle='--', label='Student Validation NLL')
    

    ax.set_title(title_str)
    ax.set_xlabel("Training Iterations (t)")
    ax.set_ylabel("Average Negative Log-Likelihood (NLL)")
    ax.grid(True)
    ax.legend()

    filename = f"distillation_nll_{timestamp}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()
    print(f"Distillation NLL plot saved to {os.path.join(output_dir, filename)}")


def plot_results_bayers(results_data, timestamp, hp, output_dir=path_exp_fig_sens_figs):
    """
    Main plotting function that calls specialized helper functions
    to generate a suite of plots for the experiment.
    """
    if not results_data:
        print("No data available to plot.")
        return

    print("\n--- Generating All Result Plots ---")
    
    df = pd.DataFrame(results_data)
    os.makedirs(output_dir, exist_ok=True)

    # Call each specific plotting function
    plot_teacher_performance(df, hp, output_dir, timestamp)
    plot_distillation_nll(df, hp, output_dir, timestamp)
    
    print("--- All plots generated. ---")

#Old bayes'
# def plot_results_bayers(results_data, timestamp, hp, output_dir=path_exp_fig_sens_figs):   
#     if not results_data:
#         print("No data available to plot.")
#         return

#     print("\n--- Plotting Results ---")

#     df = pd.DataFrame(results_data)

#     t_steps = df['t']
#     tr_nll_val = df['tr_nll_val']
#     tr_nll_train = df['tr_nll_train']
#     st_nll_val = df['st_nll_val']
#     window_size = 10
#     teacher_nll_train_smooth = tr_nll_train.rolling(window=window_size).mean()
#     # tr_nll_val_smooth = tr_nll_val.rolling(window=window_size).mean()
#     # st_nll_val_smooth = st_nll_val.rolling(window=window_size).mean()    

#     tr_final_val_acc = df['tr_acc_val'].iloc[-1] 
#     # final_train_acc = df['tr_acc_train'].iloc[-1]
#     st_final_val_acc = df['st_acc_val'].iloc[-1] 

#     title_str = (
#         f"Teacher/Student NLL (own SGLD)\n ({hp['iterations']} iterations) |  (Final err rate Val tr/st: {(1-tr_final_val_acc)*100:.2f}%/{(1-st_final_val_acc) * 100:.2f}%)\n"
#         f"(Poly Decay: a={hp['tr_poly_a']:.2e}, b={hp['tr_poly_b']:.0f}, γ={hp['tr_poly_gamma']}) | "
#         f"(Data: M={hp['batch_size']}, τ={hp['tau']})"
#     )

#     plt.figure(figsize=(12, 7))
#     ax = plt.gca()

#     ax.plot(t_steps, tr_nll_val, marker='o', linestyle='-', label='Teacher NLL Validation')
#     ax.plot(t_steps, tr_nll_train, marker='o', color="pink", linestyle='None', alpha=0.3, label='Teacher NLL Train (Raw)')
#     ax.plot(t_steps, st_nll_val,  color="green", marker='s', markersize=3, linestyle='--', label="Student NLL")
#     # ax.plot(t_steps, tr_nll_val_smooth, marker='o', linestyle='-', label='Teacher NLL Validation')
#     # ax.plot(t_steps, st_nll_val_smooth, color="green", marker='s', markersize=3, linestyle='--', label=f'Student NLL Validation (Smoothed, w={window_size})')

#     # ax.plot(t_steps, teacher_nll_train_smooth, color="red", linestyle='-', label=f'Teacher NLL Train (Smoothed, w={window_size})')
    

#     #it looks ugly af if I dont make some sort of writing on the x-axis
#     num_iterations = hp.get('iterations', t_steps.max())
#     tick_locations = np.linspace(0, num_iterations, 11)

#     def k_formatter(x, pos):
#         if x == 0:
#             return '0'
#         elif x == 1_000_000:
#              return '1M'
#         return f'{int(x*1e-3)}k'
    
#     ax.set_xticks(tick_locations)
#     ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))

#     ax.set_title(title_str)
#     ax.set_xlabel('Training Iterations (t)') # Corrected label
#     ax.set_ylabel('Average Negative Log-Likelihood (NLL)')
#     ax.grid(True)
#     ax.legend()

#     plot_filename = (
#         f"MNIST_{timestamp}_"
#         f"T={hp['iterations']}_"
#         f"M={hp['batch_size']}_"
#         f"tau={hp['tau']}_"
#         f"a={hp['tr_poly_a']:.1e}_"
#         f"b={hp['tr_poly_b']}_"
#         f"g={hp['tr_poly_gamma']}.png"
#     )
#     os.makedirs(output_dir, exist_ok=True)
#     full_path = os.path.join(output_dir, plot_filename)
    
#     plt.savefig(full_path, bbox_inches='tight')
#     plt.close()
#     print(f"Plot saved to {full_path}")




def csv_results_MNIST(results_data, timestamp, hp, output_dir=path_exp_MNIST_default):
    if not results_data:
        print("No results to save.")
        return None

    print("\n--- Saving Results to CSV ---")
    
    results_df = pd.DataFrame(results_data)
    
    csv_filename = (
        f"MNIST_fixed_lr_{timestamp}_"
        f"T={hp['iterations']}_"
        f"M={hp['batch_size']}.csv"
    )
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, csv_filename)
    
    results_df.to_csv(full_path, index=False)
    print(f"Results saved to {full_path}")
    
    print("\nResults Preview:")
    print(results_df.head())
    
    return results_df



def plot_results_MNIST(results_data, timestamp, hp, output_dir=path_exp_MNIST_default):
    if not results_data:
        print("No data available to plot.")
        return

    print("\n--- Plotting Results ---")
    
    df = pd.DataFrame(results_data)
    t_steps = df['t']
    teacher_nll_val = df['tr_nll_val']
    teacher_nll_train = df['tr_nll_train']
    
    window_size = 10
    teacher_nll_train_smooth = teacher_nll_train.rolling(window=window_size).mean()

    title_str = (
        f"Teacher NLL (Fixed LR): ({hp['iterations']} iterations)\n"
        f"(lr={hp['tr_lr']:.1e}, τ={hp['tau']}) | "
        f"(Batch Size M={hp['batch_size']})"
    )
    
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    ax.plot(t_steps, teacher_nll_val, marker='o', linestyle='-', label='Teacher NLL Validation')
    ax.plot(t_steps, teacher_nll_train, marker='o', color="pink", linestyle='None', alpha=0.3, label='Teacher NLL Train (Raw)')
    ax.plot(t_steps, teacher_nll_train_smooth, color="red", linestyle='-', label=f'Teacher NLL Train (Smoothed, w={window_size})')
    
    num_iterations = hp.get('iterations', t_steps.max())
    tick_locations = np.linspace(0, num_iterations, 11)

    def k_formatter(x, pos):
        if x == 0:
            return '0'
        elif x >= 1_000_000:
             return f'{x*1e-6:.0f}M'
        return f'{int(x*1e-3)}k'
    
    ax.set_xticks(tick_locations)
    ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))
        
    ax.set_title(title_str)
    ax.set_xlabel('Training Iterations (t)')
    ax.set_ylabel('Average Negative Log-Likelihood (NLL)')
    ax.grid(True)
    ax.legend()
        
    plot_filename = (
        f"MNIST_fixed_lr_{timestamp}_"
        f"T={hp['iterations']}_"
        f"M={hp['batch_size']}.png"
    )

    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, plot_filename)
    
    plt.savefig(full_path)
    print(f"Plot saved to {full_path}")


def store_weights(st_w, tr_w, timestamp, T, output_dir=path_exp_weights):
    student_weights_path = os.path.join(output_dir, f"final_student_weights_{timestamp}_T={T}.pth")
    teacher_weights_path = os.path.join(output_dir, f"final_teacher_weights_{timestamp}_T={T}.pth")

    print(f"Saving final student weights to: {student_weights_path}")
    torch.save(st_w, student_weights_path)
    
    print(f"Saving final teacher weights to: {teacher_weights_path}")
    torch.save(tr_w, teacher_weights_path)
