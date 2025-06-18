# experiment1_stat_plot.py

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
from constants import path_exp_fig, path_exp_stat

def save_results_to_csv(results_data, results_dir=path_exp_stat):
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


def plot_results(results_data, figs_dir=path_exp_fig):
    if not results_data:
        print("No data available to plot.")
        return

    print("\n--- Plotting Results ---")

    teacher_nll = [record.get('tr_nll') for record in results_data]
    student_nll = [record.get('st_nll') for record in results_data]
    t_steps = np.arange(len(teacher_nll))
    plt.figure(figsize=(12, 7))
    plt.plot(t_steps, teacher_nll, marker='o', linestyle='-', label='Teacher NLL')
    plt.plot(t_steps, student_nll, marker='x', linestyle='--', label='Student NLL')

    plt.title(f'Teacher/Student NLL: {len(t_steps)}-steps')
    plt.xlabel('Student Training Steps (t)')
    plt.ylabel('Average Negative Log-Likelihood (NLL)')
    plt.grid(True)
    plt.legend()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'distillation_nll_plot_{timestamp}.png'
    full_path = os.path.join(figs_dir, plot_filename)
    
    plt.savefig(full_path)
    print(f"Plot saved to {full_path}")