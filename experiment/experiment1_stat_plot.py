# experiment1_stat_plot.py

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
from constants import path_exp_fig, path_exp_stat, path_exp_weights
import torch

def save_results_to_csv(results_data, timestamp, T, results_dir=path_exp_stat):
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


# def plot_results_training(results_data,T,tiemstamp,figs_dir=path_exp_fig):
#     if not results_data:
#         print("No data available to plot.")
#         return
#     teacher_nll_train = [record.get('tr_nll_train') for record in results_data]

#     t_steps = np.arange(len(teacher_nll_train))
#     plt.figure(figsize=(12, 7))
#     plt.plot(t_steps, teacher_nll_train, marker='o', linestyle='-', label='Teacher NLL')



#     return None


def plot_results(results_data, T, timestamp, figs_dir=path_exp_fig):
    if not results_data:
        print("No data available to plot.")
        return

    print("\n--- Plotting Results ---")

    teacher_nll = [record.get('tr_nll') for record in results_data]
    # student_nll = [record.get('st_nll') for record in results_data]
    teacher_nll_train = [record.get('tr_nll_train') for record in results_data]
    t_steps = np.arange(len(teacher_nll))
    plt.figure(figsize=(12, 7))

    plt.plot(t_steps, teacher_nll, marker='o', linestyle='-', label='Teacher NLL')
    # plt.plot(t_steps, student_nll, marker='x', linestyle='--', label='Student NLL')
    plt.plot(t_steps,teacher_nll_train, marker='o', color="pink", linestyle='-', label='Teacher NLL Train')

    plt.title(f'Teacher/Student NLL: \n ({T} iterations)')
    plt.xlabel('Student Training Steps (t)')
    plt.ylabel('Average Negative Log-Likelihood (NLL)')
    plt.grid(True)
    plt.legend()


    
    plot_filename = f'distillation_nll_plot_{timestamp}_T={T}.png'
    full_path = os.path.join(figs_dir, plot_filename)
    
    plt.savefig(full_path)
    print(f"Plot saved to {full_path}")

def plot_results_tr(results_data, T, timestamp, figs_dir=path_exp_fig):
    teacher_nll = [record.get('tr_nll') for record in results_data]
    student_nll = [record.get('st_nll') for record in results_data]
    t_steps = np.arange(len(teacher_nll))
    plt.figure(figsize=(12, 7))
    plt.plot(t_steps, teacher_nll, marker='o', linestyle='-', label='Teacher NLL')
    plt.plot(t_steps, student_nll, marker='x', linestyle='--', label='Student NLL')

    plt.title(f'Teacher/Student NLL: \n ({T} iterations)')
    plt.xlabel('Student Training Steps (t)')
    plt.ylabel('Average Negative Log-Likelihood (NLL)')
    plt.grid(True)
    plt.legend()
    
    plot_filename = f'tr_nll_{timestamp}_T={T}.png'
    full_path = os.path.join(figs_dir, plot_filename)
    
    plt.savefig(full_path)
    print(f"Plot saved to {full_path}")


def store_weights(st_w, tr_w, timestamp, T, weights_dir=path_exp_weights):
    student_weights_path = os.path.join(weights_dir, f"final_student_weights_{timestamp}_T={T}.pth")
    teacher_weights_path = os.path.join(weights_dir, f"final_teacher_weights_{timestamp}_T={T}.pth")

    print(f"Saving final student weights to: {student_weights_path}")
    torch.save(st_w, student_weights_path)
    
    print(f"Saving final teacher weights to: {teacher_weights_path}")
    torch.save(tr_w, teacher_weights_path)