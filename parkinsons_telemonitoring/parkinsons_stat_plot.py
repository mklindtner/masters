import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from constants import path_parkin_stat, path_parkin_fig, path_parkin_weights
import torch
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mticker
import uncertainty_toolbox as uct
from tqdm.auto import tqdm

# In plotting.py
def plot_metric_over_time(results_df, metric_column, title, ylabel, color, output_dir, timestamp):
    """
    A generic function to plot any single metric over time from the results DataFrame.
    """
    # Guard Clause: Check if the required column exists and has data
    if metric_column not in results_df.columns or results_df[metric_column].isnull().all():
        print(f"Notice: No data found for metric '{metric_column}', skipping plot.")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Smooth the metric for better visualization
    smoothing_window = 5
    metric_smooth = results_df[metric_column].rolling(window=smoothing_window).mean()

    ax.plot(
        results_df['t'], 
        metric_smooth, 
        linestyle='-', 
        color=color, 
        label=f'Smoothed {ylabel} (w={smoothing_window})'
    )
    
    # Add titles, labels, and formatting
    ax.set_title(title)
    ax.set_xlabel("Training Iterations (t)")
    ax.set_ylabel(f"{ylabel}")
    
    total_iterations = results_df['t'].max()
    format_x_axis(ax, total_iterations) # Use your existing helper

    ax.legend()
    ax.grid(True, linestyle='--')
    
    # Save the figure
    filename = f"plot_{metric_column}_{timestamp}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_student_mean_mse_train(results_df, output_dir, timestamp):
    """Plots the smoothed training MSE for the student's mean prediction."""
    plot_metric_over_time(
        results_df=results_df,
        metric_column='st_mse_mean_train',
        title='Student Training Loss',
        ylabel='MSE (Mean)',
        color='green',
        output_dir=output_dir,
        timestamp=timestamp
    )

def plot_student_variance_mse_train(results_df, output_dir, timestamp):
    """Plots the smoothed training MSE for the student's variance prediction."""
    plot_metric_over_time(
        results_df=results_df,
        metric_column='st_mse_var_train',
        title='Student Training Loss',
        ylabel='MSE (Variance)',
        color='purple',
        output_dir=output_dir,
        timestamp=timestamp
    )

def plot_student_total_loss(results_df, output_dir, timestamp):
    """Plots the smoothed total training loss for the MeanVariance student."""
    plot_metric_over_time(
        results_df=results_df,
        metric_column='st_total_loss_train',
        title='Student Training Loss (Total MSE)',
        ylabel='Total MSE Loss',
        color='orange',
        output_dir=output_dir,
        timestamp=timestamp
    )


def save_uncertainty_metrics_to_csv(metrics, output_dir, timestamp):
    """
    Flattens the nested metrics dictionary and saves it as a single-row CSV file.
    """
    flat_metrics = {}
    
    # Loop through the main categories to flatten the dictionary
    for category, sub_dict in metrics.items():
        if isinstance(sub_dict, dict):
             for metric_name, value in sub_dict.items():
                # Create a new, unique key (e.g., 'accuracy_rmse')
                flat_key = f"{category}_{metric_name}"
                flat_metrics[flat_key] = value
            
    # Convert the flattened dictionary to a pandas DataFrame
    metrics_df = pd.DataFrame(flat_metrics, index=[0])
    
    # Save to CSV
    filename = f"uncertainty_metrics_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    metrics_df.to_csv(filepath, index=False)
    
    print(f"\nUncertainty metrics saved to {filepath}")


def bayes_predictions(network, weight_samples, loader, device):
    """
    Performs Bayesian Model Averaging to get final predictions and uncertainty.
    Returns:
        (np.array, np.array, np.array): y_pred_mean, y_pred_std, y_true
    """
    network.eval()
    all_means_across_samples = []

    # Get predictions for each weight sample
    for sample_weights in tqdm(weight_samples, desc="Getting BMA Predictions"):
        network.load_state_dict({k: v.to(device) for k, v in sample_weights.items()})
        batch_means = []
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(device)
                mean, _ = network(data) # We only need the mean from each model
                batch_means.append(mean)
        all_means_across_samples.append(torch.cat(batch_means))

    # Collect ground truth labels once
    y_true = torch.cat([target for _, target in loader]).cpu().numpy().flatten()    
    
    # Calculate the final predictive mean and standard deviation
    stacked_means = torch.stack(all_means_across_samples, dim=0)
    y_pred_mean = torch.mean(stacked_means, dim=0).cpu().numpy().flatten()
    y_pred_std = torch.std(stacked_means, dim=0).cpu().numpy().flatten()

    return y_pred_mean, y_pred_std, y_true


def calculate_and_print_uncertainty_metrics(y_pred_mean, y_pred_std, y_true):
    """
    Uses uncertainty-toolbox to calculate and print all standard metrics.
    """
    print("\n" + "="*50)
    print("  Uncertainty and Calibration Metrics")
    print("="*50)

    # Get all metrics from the toolbox
    metrics = uct.metrics.get_all_metrics(y_pred_mean, y_pred_std, y_true)
       
    scoring_rule_metrics = metrics.get("scoring_rule", {})
    accuracy_metrics = metrics.get("accuracy", {})
    calibration_metrics = metrics.get("avg_calibration", {})
    sharpness_metrics = metrics.get("sharpness", {})
    print(f"  Average NLL:                {scoring_rule_metrics.get('nll', float('nan')):.4f}")
    print(f"  RMSE:                       {accuracy_metrics.get('rmse', float('nan')):.4f}")
    print(f"  Miscalibration Area:        {calibration_metrics.get('miscal_area', float('nan')):.4f}")
    print(f"  Sharpness (Avg. Std Dev):   {sharpness_metrics.get('sharp', float('nan')):.4f}")
    print("="*50)
    
    return metrics


def plot_uncertainty_visualizations(y_pred_mean, y_pred_std, y_true, output_dir, timestamp):
    """
    Uses uncertainty-toolbox to generate and save standard plots individually.
    """
    print("\nGenerating uncertainty plots...")

    # This is the most important plot for uncertainty quality.
    print("  - Generating calibration plot...")
    fig_cal, ax_cal = plt.subplots(1, 1, figsize=(8, 8))
    uct.viz.plot_calibration(y_pred_mean, y_pred_std, y_true, ax=ax_cal)
    
    cal_filename = f"plot_calibration_{timestamp}.png"
    cal_filepath = os.path.join(output_dir, cal_filename)
    fig_cal.savefig(cal_filepath, dpi=300)
    plt.close(fig_cal)
    print(f"    Calibration plot saved to {cal_filepath}")

    # This plot visually shows the uncertainty for each prediction.
    print("  - Generating prediction interval plot...")
    fig_pi, ax_pi = plt.subplots(1, 1, figsize=(8, 8))
    uct.viz.plot_intervals(y_pred_mean, y_pred_std, y_true, ax=ax_pi)
    
    pi_filename = f"plot_intervals_{timestamp}.png"
    pi_filepath = os.path.join(output_dir, pi_filename)
    fig_pi.savefig(pi_filepath, dpi=300)
    plt.close(fig_pi)
    print(f"    Prediction interval plot saved to {pi_filepath}")


def bayes_uncertainty_analysis(tr_items, msc_items, final_teacher_samples, output_dir, timestamp):
    """
    Takes the final teacher samples and runs a full uncertainty analysis,
    generating metrics and plots.
    """
    if not final_teacher_samples:
        print("No teacher samples provided. Skipping uncertainty analysis.")
        return

    print("\n" + "#"*60)
    print("  Running Post-Hoc Bayesian Uncertainty Analysis on Teacher Model")
    print("#"*60)

    # Unpack the necessary components
    _, teacher_model, _, val_loader = tr_items
    device = msc_items[3]

    y_pred_mean, y_pred_std, y_true = bayes_predictions(teacher_model, final_teacher_samples, val_loader, device)
    metrics = calculate_and_print_uncertainty_metrics(y_pred_mean, y_pred_std, y_true)
    plot_uncertainty_visualizations(y_pred_mean, y_pred_std, y_true, output_dir, timestamp)
    save_uncertainty_metrics_to_csv(metrics, output_dir, timestamp)

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

    ax.plot(tsteps, results_df['tr_nll_val'], marker='o', linestyle='--', color="blue", label='Teacher NLL Validation')
    ax.plot(tsteps, train_nll_smooth, marker='o', linestyle='-', color="pink", label=f'Teacher NLL Train (Smoothed)')
    # ax.plot(tsteps, results_df['tr_nll_train'], linestyle='-', color="red", label='Teacher NLL Train')

    ax.set_title(f"Teacher NLL Performance")
    ax.set_xlabel("Training Iterations (t)")
    ax.set_ylabel("Average NLL")

    total_iterations = results_df['t'].max()
    format_x_axis(ax, total_iterations)

    ax.legend()
    ax.grid(True, linestyle="--")
    plt.savefig(os.path.join(output_dir, f"plot_teacher_perf_{timestamp}.png"))
    plt.close()

def plot_gnll_comparison(results_df, output_dir, timestamp):
    """
    Compares the validation NLL of the student and teacher models on the same plot.
    """
    st_metric = 'st_nll_val'
    tr_metric = 'tr_nll_val'


    if (st_metric not in results_df.columns or results_df[st_metric].isnull().all() or tr_metric not in results_df.columns or results_df[tr_metric].isnull().all()):
        print("Notice: Missing student or teacher GNLL data, skipping comparison plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(results_df['t'], results_df[st_metric], marker='o', linestyle='-', color='purple', label='Student Validation NLL')
    ax.plot(results_df['t'], results_df[tr_metric], marker='x', linestyle='--', color='blue', label='Teacher Validation NLL')

    ax.set_title("Teacher Vs. Student Validation")
    ax.set_xlabel("Training Iterations (t)")
    ax.set_ylabel("NLL")
    
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

def plot_kl_divergence(results_df, output_dir, timestamp):
    if 'kl_divergence' not in results_df.columns or results_df['kl_divergence'].isnull().all():
        print("No KL divergence data to plot.")
        return        

    fig, ax = plt.subplots(figsize=(10, 6))

    smoothing_window = 3
    kl_smooth = results_df['kl_divergence'].rolling(window=smoothing_window).mean()
 
    ax.plot(results_df['t'], kl_smooth, marker='.', linestyle='-', color='green', label=f'KL Divergence (Smoothed, w={smoothing_window})')
    # ax.plot(results_df['t'], results_df['kl_divergence'], marker='.', linestyle='-', color='green', label='KL(student || teacher)')

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

        print("Plotting Student Total Training Loss (MSE)...")
        plot_student_total_loss(results_df, output_dir, timestamp)

        print("Plotting Student Mean MSE Training Loss...")
        plot_student_mean_mse_train(results_df, output_dir, timestamp)

        print("Plotting Student Variance MSE Training Loss...")
        plot_student_variance_mse_train(results_df, output_dir, timestamp)

    
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
        f"val_step={hp['val_step']}"
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