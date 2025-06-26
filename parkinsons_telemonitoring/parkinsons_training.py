import argparse
from datetime import datetime
from parkinsons_telemonitoring.parkinsons_data import DEFAULT_TR_POLY_A, DEFAULT_TR_POLY_DECAY_GAMMA, DEFAULT_TR_POLY_LR_B, DEFAULT_OUTPUT_FOLDER, setup_experiment, DEFAULT_BATCH_SIZE, DEFAULT_TR_LR, DEFAULT_TAU, DEFAULT_ST_DROPOUT, DEFAULT_ST_LR_INIT, DEFAULT_BURNIN, DEFAULT_H, DEFAULT_T
from parkinsons_telemonitoring.parkinsons_model import distillation_posterior_parkinsons, bayesian_distillation_parkin
from parkinsons_telemonitoring.parkinsons_stat_plot import plot_results_bayers, save_results_to_csv_bayers

def main(args):
    
    tr_list, tr_hyp_param, msc_list = setup_experiment(
        batch_size=args.batch_size,
        tau=args.tau,
        st_dropout=args.st_dropout,
        st_lr_init=args.st_lr_init,
        B=args.B,
        H=args.H,
        T=args.iterations,
        poly_a=args.tr_poly_a,
        poly_gamma=args.tr_poly_gamma,
        poly_b=args.tr_poly_b
    )
    if args.output_dir == None:
        print("no output dir given, exiting")
        return
    #new2
    results, _ = bayesian_distillation_parkin(tr_list, msc_items=msc_list, tr_hyp_par=tr_hyp_param, T_total=args.iterations)
    hp_dict = vars(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_results_bayers(results_data=results, timestamp=timestamp, hp=hp_dict, output_dir=args.output_dir)
    save_results_to_csv_bayers(results_data=results, hp=hp_dict, timestamp=timestamp, output_dir=args.output_dir)

    #old
    # log_results, final_tr_w, final_st_w = distillation_posterior_parkinsons(
    #     tr_list, st_list, msc_list, T_steps=args.T
    # )
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # plot_tr_results_distillation(log_results, timestamp=timestamp, T=T)
    # save_results_to_csv(log_results, timestamp=timestamp, T=T)
    # store_weights(final_tr_w, final_st_w, timestamp=timestamp, T=T) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Parkinsons Telemonitoring distillation experiment.')

    parser.add_argument('--iterations', type=int, default=DEFAULT_T, help='Total training iterations')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size (M)')
    parser.add_argument('--tau', type=float, default=DEFAULT_TAU, help='Precision of the prior (tau)')
    parser.add_argument('--st_dropout', type=float, default=DEFAULT_ST_DROPOUT, help='Student dropout rate')
    parser.add_argument('--st_lr_init', type=float, default=DEFAULT_ST_LR_INIT, help='Student initial learning rate')
    parser.add_argument('--B', type=int, default=DEFAULT_BURNIN, help='Burn-in period')
    parser.add_argument('--H', type=int, default=DEFAULT_H, help='Distillation frequency')
    
    
    # parser.add_argument('--tr_lr', type=float, default=DEFAULT_TR_LR, help='Teacher learning rate')
    parser.add_argument('--tr_poly_a', type=float, default=DEFAULT_TR_POLY_A, help='(Polynomial decay): Initial Teacher learning rate')
    parser.add_argument('--tr_poly_gamma',  type=float, default=DEFAULT_TR_POLY_DECAY_GAMMA, help='(Polynomial decay): gamma decay')
    parser.add_argument('--tr_poly_b', type=float, default=DEFAULT_TR_POLY_LR_B, help='(Polynomial decay): b decay')

    parser.add_argument('--output_dir', type=str, default= DEFAULT_OUTPUT_FOLDER, help='Directory to save all run artifacts')



    args = parser.parse_args()
    main(args)