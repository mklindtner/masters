# from parkinsons_telemonitoring.parkinsons_model import distillation_posterior_parkinsons
# from parkinsons_telemonitoring.parkinsons_data import tr_list, st_list, msc_list
# from parkinsons_telemonitoring.parkinsons_stat_plot import plot_tr_results_distillation, save_results_to_csv, store_weights
# from datetime import datetime

# T_test = 3050

# results, tr_w, st_w = distillation_posterior_parkinsons(tr_list, st_list, msc_list, T_steps=T_test)
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# plot_tr_results_distillation(results, timestamp=timestamp, T=T_test)
# save_results_to_csv(results, timestamp=timestamp, T=T_test)
# store_weights(tr_w,st_w, timestamp=timestamp, T=T_test)


### NEW TRIAL ###

# parkinsons_training.py

import argparse
from datetime import datetime
# Import the new setup function and the default values
from parkinsons_telemonitoring.parkinsons_data import setup_experiment, DEFAULT_BATCH_SIZE, DEFAULT_TR_LR, DEFAULT_TAU, DEFAULT_ST_DROPOUT, DEFAULT_ST_LR_INIT, DEFAULT_BURNIN, DEFAULT_H, DEFAULT_T
from parkinsons_telemonitoring.parkinsons_model import distillation_posterior_parkinsons
from parkinsons_telemonitoring.parkinsons_stat_plot import plot_tr_results_distillation, save_results_to_csv, store_weights

def main(args):
    
    tr_list, st_list, msc_list,T = setup_experiment(
        batch_size=args.batch_size,
        tr_lr=args.tr_lr,
        tau=args.tau,
        st_dropout=args.st_dropout,
        st_lr_init=args.st_lr_init,
        B=args.B,
        H=args.H,
    )

    # 2. Run the distillation process
    log_results, final_tr_w, final_st_w = distillation_posterior_parkinsons(
        tr_list, st_list, msc_list, T_steps=args.T
    )
    
    # 3. Save artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_tr_results_distillation(log_results, timestamp=timestamp, T=T)
    save_results_to_csv(log_results, timestamp=timestamp, T=T)
    store_weights(final_tr_w, final_st_w, timestamp=timestamp, T=T) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Parkinsons Telemonitoring distillation experiment.')

    # Add arguments for all your hyperparameters, using the imported constants as defaults
    parser.add_argument('--iterations', type=int, default=DEFAULT_T, help='Total training iterations')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size (M)')
    parser.add_argument('--tr_lr', type=float, default=DEFAULT_TR_LR, help='Teacher learning rate')
    parser.add_argument('--tau', type=float, default=DEFAULT_TAU, help='Precision of the prior (tau)')
    parser.add_argument('--st_dropout', type=float, default=DEFAULT_ST_DROPOUT, help='Student dropout rate')
    parser.add_argument('--st_lr_init', type=float, default=DEFAULT_ST_LR_INIT, help='Student initial learning rate')
    parser.add_argument('--B', type=int, default=DEFAULT_BURNIN, help='Burn-in period')
    parser.add_argument('--H', type=int, default=DEFAULT_H, help='Distillation frequency')


    args = parser.parse_args()
    main(args)