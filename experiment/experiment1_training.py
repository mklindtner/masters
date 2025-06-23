from experiment.experiment1_models import distillation_posterior_MNIST, bayesian_distillation
from experiment.experiment1_data import tr_items, st_items, msc_items, T, B, H, tr_optim, trainloader, testloader, tr_criterion, tr_model, device
import matplotlib.pyplot as plt 
from datetime import datetime
from experiment.experiment1_stat_plot import save_results_to_csv, plot_results, store_weights
from experiment.experiment1_data_bayers import  tr_items_bayers, tr_hyp_par, tr_hyp_par_all

T_test = T

# results, st_w, tr_w = distillation_posterior_MNIST(tr_items=tr_items, st_items=st_items, msc_list=msc_items, T_total=T_test)
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# plot_results(results, T=T_test, timestamp=timestamp)
# save_results_to_csv(results, T=T_test, timestamp=timestamp)



results, tr_w = bayesian_distillation(tr_items=tr_items_bayers, msc_items=msc_items, tr_hyp_par=tr_hyp_par, T_total = T_test)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_results(results, timestamp=timestamp, hp = tr_hyp_par_all, distil_type="MNIST_bayesian")
save_results_to_csv(results, T=T_test, timestamp=timestamp, distill_type="MNIST_bayesian")