import argparse
from datetime import datetime
from parkinsons_telemonitoring.parkinsons_data import DEFAULT_TR_POLY_A, DEFAULT_TR_POLY_DECAY_GAMMA, DEFAULT_TR_POLY_LR_B, DEFAULT_OUTPUT_FOLDER, setup_experiment, DEFAULT_BATCH_SIZE, DEFAULT_TR_LR, DEFAULT_TAU, DEFAULT_ST_DROPOUT, DEFAULT_ST_LR_INIT, DEFAULT_BURNIN, DEFAULT_H, DEFAULT_T
from parkinsons_telemonitoring.parkinsons_model import bayesian_distillation_parkin
from parkinsons_telemonitoring.parkinsons_stat_plot import plot_results_bayers, save_results_to_csv_bayers


def bayesian_distillation_parkin(tr_items, msc_items, tr_hyp_par, T_total=1e6, verbose=True):
    tr_bayers, tr_network, tr_loader_train, tr_loader_valid = tr_items
    B, H, eval_criterion, device = msc_items
    lr_init, decay_gamma, lr_b = tr_hyp_par
    train_iterator = itertools.cycle(tr_loader_train)
    results = []


    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

    print(f"--- Starting Distillation Process for {T_total} steps ---")
    
    T = tqdm(range(T_total), desc="Total Steps", disable=not verbose, bar_format=bar_format)

    # tr_criterion_nll = nn.CrossEntropyLoss(reduction='mean')

    for t in T:
        lr = lr_init * (lr_b + t)**(-decay_gamma)
        T.set_postfix(LR=f"{lr:.2e}") 
        inputs, labels = next(train_iterator)
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(inputs.size(0), -1)

        #The tr_network should be inside here
        tr_bayers.sgld_step(inputs, labels, lr)

        if t >= B and (t % H == 0):
            with torch.no_grad():            
                mean,log_var = tr_network(inputs)
                tr_nll_train = eval_criterion(input=mean, target=labels, var=torch.exp(log_var))
                tr_nll_avg = 1/len(labels) * tr_nll_train.item()

                tr_network.eval()  
                teacher_nll = validate_network(tr_network, tr_loader_valid, eval_criterion, device, verbose=False)
                
            results.append({
                't': t + 1,
                'tr_nll_val': teacher_nll,
                'tr_nll_train': tr_nll_avg,
            })        

            tr_network.train()
    return results, tr_network.state_dict()


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

    parser.add_argument('--output_dir', type=str, default="parkinsons_telemonitoring/var_const/singlevar_runs", help='Directory to save all run artifacts')



    args = parser.parse_args()
    main(args)
