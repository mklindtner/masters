import argparse
from datetime import datetime
from parkinsons_telemonitoring.parkinsons_data import DEFAULT_TR_POLY_A, DEFAULT_TR_POLY_DECAY_GAMMA, DEFAULT_TR_POLY_LR_B, DEFAULT_OUTPUT_FOLDER, setup_experiment, DEFAULT_BATCH_SIZE, DEFAULT_TR_LR, DEFAULT_TAU, DEFAULT_ST_DROPOUT, DEFAULT_ST_LR_INIT, DEFAULT_BURNIN, DEFAULT_H, DEFAULT_T, DEFAULT_VAL_STEP
from parkinsons_telemonitoring.parkinsons_model import bayesian_distillation_parkin
from parkinsons_telemonitoring.parkinsons_stat_plot import create_and_save_plots, bayes_uncertainty_analysis

def main(args):
    tr_items, st_items, tr_hyp_param, msc_items = setup_experiment(
        batch_size=args.batch_size,
        tau=args.tau,
        st_dropout=args.st_dropout,
        st_lr_init=args.st_lr_init,
        B=args.B,
        H=args.H,
        T=args.iterations,
        poly_a=args.tr_poly_a,
        poly_gamma=args.tr_poly_gamma,
        poly_b=args.tr_poly_b,
        student_mode=args.student_mode,
        val_step=args.val_step
    )

       
    results, tr_samples, _ = bayesian_distillation_parkin(tr_items, st_items=st_items, msc_items=msc_items, val_step=args.val_step, tr_hyp_par=tr_hyp_param, T_total=args.iterations)
    hp_dict = vars(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    create_and_save_plots(results, hp_dict, args.output_dir, timestamp)
    bayes_uncertainty_analysis(tr_items, msc_items, tr_samples, args.output_dir, timestamp)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Parkinsons Telemonitoring distillation experiment.')

    parser.add_argument('--iterations', type=int, default=DEFAULT_T, help='Total training iterations')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size (M)')
    parser.add_argument('--tau', type=float, default=DEFAULT_TAU, help='Precision of the prior (tau)')
    parser.add_argument('--st_dropout', type=float, default=DEFAULT_ST_DROPOUT, help='Student dropout rate')
    parser.add_argument('--st_lr_init', type=float, default=DEFAULT_ST_LR_INIT, help='Student initial learning rate')
    parser.add_argument('--B', type=int, default=DEFAULT_BURNIN, help='Burn-in period')
    parser.add_argument('--H', type=int, default=DEFAULT_H, help='Distillation frequency')
    parser.add_argument('--student_mode', type=str, required=True, choices=['mean_and_variance', 'variance_only', 'mean_only'])
        
    parser.add_argument('--tr_poly_a', type=float, default=DEFAULT_TR_POLY_A, help='(Polynomial decay): Initial Teacher learning rate')
    parser.add_argument('--tr_poly_gamma',  type=float, default=DEFAULT_TR_POLY_DECAY_GAMMA, help='(Polynomial decay): gamma decay')
    parser.add_argument('--tr_poly_b', type=float, default=DEFAULT_TR_POLY_LR_B, help='(Polynomial decay): b decay')
    parser.add_argument('--val_step', type=int, default=DEFAULT_VAL_STEP, help='(Polynomial decay): b decay')


    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_FOLDER, help='Directory to save all run artifacts')     

    args = parser.parse_args()
    main(args)