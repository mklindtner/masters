from experiment.experiment1_models import distil_MNIST
from experiment.experiment1_data_MNIST import setup_distillation_experiment, EXP_MNIST_TAU, EXP_MNIST_TR_LR
from datetime import datetime
from experiment.experiment1_stat_plot import csv_results_MNIST, plot_results_MNIST, store_weights
import numpy as np  
import random       
import torch
import argparse


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)



def main(args):    
    tr_items, msc_items, T_total_from_setup = setup_distillation_experiment(args.batch_size)
    T_total = args.iterations if args.iterations is not None else T_total_from_setup

    results, _ = distil_MNIST(
        tr_items=tr_items,
        st_items=None,
        msc_list=msc_items,
        T_total=T_total
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hp_dict = vars(args) 
    hp_dict['tau'] = EXP_MNIST_TAU
    hp_dict['tr_lr'] = EXP_MNIST_TR_LR
    hp_dict['iterations'] = T_total

    plot_results_MNIST(results, timestamp=timestamp, hp=hp_dict)
    csv_results_MNIST(results, timestamp=timestamp, hp=hp_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Original MNIST Teacher Training Experiment')

    # The only command-line argument is now the batch size
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size (M)')
    parser.add_argument('--iterations', type=int, default=int(2e3), help='(Optional) Override total iterations.')
    parser.add_argument('--output_dir', type=str, default="experiment/MNIST_distillation/single_run", help='Directory to save run artifacts')

    args = parser.parse_args()
    
    print("--- Running experiment with the following configuration ---")
    print(f"Iterations: {args.iterations},  Batch Size (M): {args.batch_size}")
    print("  All other hyperparameters are fixed as per the GPED paper.")
    print("---------------------------------------------------------")
    
    main(args)
