from experiment.experiment1_models import bayesian_distillation
from experiment.experiment1_data_bayers import (
    setup_experiment,
    DEFAULT_T,
    DEFAULT_H,
    DEFAULT_B,
    DEFAULT_TAU,
    DEFAULT_BATCH_SIZE,
    DEFAULT_N,
    DEFAULT_TR_POLY_INIT_LR,
    DEFAULT_TR_POLY_DECAY_GAMMA, 
    DEFAULT_TR_POLY_LR_B
)
import matplotlib.pyplot as plt 
from datetime import datetime
from experiment.experiment1_stat_plot import save_results_to_csv_bayers, plot_results_bayers, store_weights
import argparse


def main(args):
    tr_bayers, tr_model, train_loader, test_loader, val_criterion, device = setup_experiment(
        batch_size=args.batch_size,
        tau=args.tau,
        N=args.N
    )

    tr_items_bayers = [tr_bayers, tr_model, train_loader, test_loader]
    msc_items = [args.B, args.H, val_criterion, device]
    tr_hyp_par = [args.tr_poly_a, args.tr_poly_gamma, args.tr_poly_b]

    results, tr_w = bayesian_distillation(
        tr_items=tr_items_bayers,
        msc_items=msc_items,
        tr_hyp_par=tr_hyp_par,
        T_total=args.iterations
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hp_dict = vars(args)

    
    plot_results_bayers(results, timestamp=timestamp ,hp=hp_dict)
    save_results_to_csv_bayers(results, timestamp=timestamp, hp=hp_dict)
    # store_weights(tr_w, "teacher_weights", timestamp=timestamp) # Optional: save weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MNIST distillation experiment')

    parser.add_argument('--iterations', type=int, default=DEFAULT_T, help='Total training iterations')
    parser.add_argument('--H', type=int, default=DEFAULT_H, help='Distillation frequency')
    parser.add_argument('--B', type=int, default=DEFAULT_B, help='Burn-in period')

    parser.add_argument('--tau', type=float, default=DEFAULT_TAU, help='Precision of the prior (tau)')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size (M)')
    parser.add_argument('--N',type=int, default=DEFAULT_N, help='DATA SIZE (N)')

    parser.add_argument('--tr_poly_a', type=float, default=DEFAULT_TR_POLY_INIT_LR, help='(Polynomial decay): Initial Teacher learning rate')
    parser.add_argument('--tr_poly_gamma',  type=float, default=DEFAULT_TR_POLY_DECAY_GAMMA, help='(Polynomial decay): gamma decay')
    parser.add_argument('--tr_poly_b', type=float, default=DEFAULT_TR_POLY_LR_B, help='(Polynomial decay): b decay')


    args = parser.parse_args()

    print("--- Setting up experiment with the following parameters: ---")
    print(f"  Data size (N): {args.N}, Batch Size (M): {args.batch_size}, Prior Precision (tau): {args.tau}")
    print(f"  Training: {args.iterations} iterations, Burn-in: {args.B}, Logging Freq (H): {args.H}")
    print(f"  Poly Decay LR: a={args.tr_poly_a}, b={args.tr_poly_b}, gamma={args.tr_poly_gamma}")
    print("---------------------------------------------------------")
    main(args)