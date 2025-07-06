import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from constants import path_parkin_stat, path_parkin_fig, path_parkin_weights
import torch
from matplotlib.ticker import FuncFormatter

def format_x_axis(ax, total_iterations):
    """
    Applies neat, evenly-spaced, and human-readable formatting to a plot's x-axis.
    This is a self-contained function.
    
    Args:
        ax (matplotlib.axes.Axes): The axes object of the plot to format.
        total_iterations (int): The maximum value for the x-axis (e.g., 1,000,000).
    """
    
    # 1. Define the formatting logic as a nested helper function
    def formatter(x, pos):
        """Formats ticks into k/M/B units."""
        x = int(x)
        if x >= 1_000_000:
            return f'{x/1_000_000:.1f}'.replace('.0', '') + 'M'
        elif x >= 1_000:
            return f'{x/1_000:.1f}'.replace('.0', '') + 'k'
        return str(x)

    # 2. Define 11 evenly spaced tick locations
    tick_locations = np.linspace(0, total_iterations, 11)
    ax.set_xticks(tick_locations)

    # 3. Apply the custom formatter function to the tick labels
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(formatter))
    
    # 4. Rotate labels slightly to prevent overlap
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def plot_tr_GNLL(results_df, output_dir, timestamp):
    fig, ax = plt.subplots(figsize=(10, 6))
    tsteps = results_df['t']
    smoothing_window = 10
    train_nll_smooth = results_df['tr_nll_train'].rolling(window=smoothing_window).mean()

    ax.plot(tsteps, results_df['tr_nll_val'], marker='o', linestyle='--', color="blue", label='Teacher GNLL Validation')
    ax.plot(tsteps, train_nll_smooth, marker='o', linestyle='-', color="red", label=f'Teacher NLL Train (Smoothed, w={smoothing_window})')
    
    ax.set_title(f"Teacher GNLL Performance")
    ax.set_xlabel("Training Iterations (t)")
    ax.set_ylabel("Average GNLL")

    total_iterations = results_df['t'].max()
    format_x_axis(ax, total_iterations)

    ax.legend()
    ax.grid(True, linestyle="--")
    plt.savefig(os.path.join(output_dir, f"plot_teacher_perf_{timestamp}.png"))
    plt.close()

def plot_gnll_comparison(results_df, output_dir, timestamp):
    """
    Compares the validation GNLL of the student and teacher models on the same plot.
    """
    st_metric = 'st_nll_val'
    tr_metric = 'tr_nll_val'


    if (st_metric not in results_df.columns or results_df[st_metric].isnull().all() or tr_metric not in results_df.columns or results_df[tr_metric].isnull().all()):
        print("Notice: Missing student or teacher GNLL data, skipping comparison plot.")
        return

    # 2. Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(results_df['t'], results_df[st_metric], marker='o', linestyle='-', color='purple', label='Student Validation NLL')
    ax.plot(results_df['t'], results_df[tr_metric], marker='x', linestyle='--', color='blue', label='Teacher Validation NLL')

    # 3. Add titles and labels for clarity
    ax.set_title("Student vs. Teacher Validation Performance (NLL)")
    ax.set_xlabel("Training Iterations (t)")
    ax.set_ylabel("Average NLL")
    
    total_iterations = results_df['t'].max()
    format_x_axis(ax, total_iterations)

    ax.legend()
    ax.grid(True, linestyle='--')
    
    # 4. Save and close the plot to free up memory
    filename = f"plot_gnll_comparison_{timestamp}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_student_mean_mse(results_df, output_dir, timestamp):
    """Plots the MSE loss for the student's mean prediction."""
    metric_column = 'st_mse_val'
    
    # Check if the required column is missing or has no valid data
    if metric_column not in results_df.columns or results_df[metric_column].isnull().all():
        print("Notice: No student MSE mean found to plot.")
        return 

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df['t'], results_df[metric_column], color='green', label='Student Validation MSE')
    

    ax.set_title("Student Mean Approximation vs. Teacher")
    ax.set_xlabel("Training Iterations (t)")
    ax.set_ylabel("Mean Squared Error (MSE)")

    total_iterations = results_df['t'].max()
    format_x_axis(ax, total_iterations)

    ax.legend()
    ax.grid(True, linestyle='--')
    plt.savefig(os.path.join(output_dir, f"plot_student_mean_mse_{timestamp}.png"))
    plt.close()

# --- Plot 3: Student Variance Approximation (MSE) ---
def plot_student_variance_mse(results_df, output_dir, timestamp):
    """Plots the MSE loss for the student's variance prediction."""
    metric_column = 'st_mse_val'
    
    # Check if the required column is missing or has no valid data
    if metric_column not in results_df.columns or results_df[metric_column].isnull().all():
        print("Notice: No student MSE var found to plot.")
        return 

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df['t'], results_df[metric_column], color='purple', label='MSE(student_log_var || teacher_log_var)')    
    ax.set_title("Student Variance Approximation vs. Teacher")
    ax.set_xlabel("Training Iterations (t)")
    ax.set_ylabel("Mean Squared Error (MSE)")

    total_iterations = results_df['t'].max()
    format_x_axis(ax, total_iterations)

    ax.legend()
    ax.grid(True, linestyle='--')

    plt.savefig(os.path.join(output_dir, f"plot_student_var_mse_{timestamp}.png"))
    plt.close()

# --- Plot 3: KL Divergence ---
def plot_kl_divergence(results_df, output_dir, timestamp):
    if 'kl_divergence' not in results_df.columns or results_df['kl_divergence'].isnull().all():
        print("No KL divergence data to plot.")
        return        

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df['t'], results_df['kl_divergence'], marker='.', linestyle='-', color='green', label='KL(student || teacher)')
    ax.set_title(f"KL Divergence (Student || Teacher)")
    ax.set_xlabel("Training Iterations (t)")
    ax.set_ylabel("KL Divergence")

    total_iterations = results_df['t'].max()
    format_x_axis(ax, total_iterations)

    ax.legend()
    ax.grid(True)
    plt.savefig(os.path.join(output_dir, f"plot_kl_divergence_{timestamp}.png"))
    plt.close()

# --- Main Plotting Function ---
def create_and_save_plots(results_data, hp_dict, output_dir, timestamp):
    """Main function to generate and save all relevant plots."""
    
    if not results_data:
        print("No results to plot.")
        return

    # Convert list of dicts to a pandas DataFrame for easier handling
    results_df = pd.DataFrame(results_data)
    student_mode = hp_dict.get('student_mode')

    print(f"--- Generating plots for student_mode: '{student_mode}' ---")

    plot_tr_GNLL(results_df, output_dir, timestamp)
        
    # 2. Conditionally plot the relevant student metric based on the mode
    if student_mode == 'mean_and_variance':
        print("Plotting Student GNLL...")
        plot_gnll_comparison(results_df, output_dir, timestamp)
        print("Plotting KL Divergence...")
        plot_kl_divergence(results_df, output_dir, timestamp)
    
    elif student_mode == 'variance_only':
        print("Plotting Student Variance MSE...")
        plot_student_variance_mse(results_df, output_dir,timestamp)
        
    elif student_mode == 'mean_only':
        print("Plotting Student Mean MSE...")
        plot_student_mean_mse(results_df, output_dir, timestamp)        
    else:
        print(f"Warning: Unknown student_mode '{student_mode}'. No specific student plot generated.")

    print("--- Plots saved successfully ---")




#Old
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
    st_nll_val = df['st_nll_val']
    window_size = 10
    # teacher_nll_train_smooth = teacher_nll_train.rolling(window=window_size).mean()
    # teacher_nll_val_smooth = teacher_nll_val.rolling(window=window_size).mean()
    
    title_str = (
        f"Parkinsons Teacher GaussNLL: ({hp['iterations']} iterations)\n"
        f"(Poly Decay: a={hp['tr_poly_a']:.2e}, b={hp['tr_poly_b']:.0f}, γ={hp['tr_poly_gamma']}) | "
        f"(Data: M={hp['batch_size']}, τ={hp['tau']})"
    )

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    

    # ax.plot(t_steps, teacher_nll_val_smooth, marker='o', linestyle='-', label='Teacher NLL Validation (Smoothed, w={window_size})')
    ax.plot(t_steps, teacher_nll_train, marker='o', color="pink", linestyle='None', alpha=0.3, label='Teacher NLL Train')
    ax.plot(t_steps, teacher_nll_val, color="red", linestyle="-", label="Teacher NLL val")
    # ax.plot(t_steps, teacher_nll_train_smooth, color="red", linestyle='-', label=f'Teacher NLL Train (Smoothed, w={window_size})')
    ax.plot(t_steps, st_nll_val, color='blue', linestyle='-', label=f"Student NLL")

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