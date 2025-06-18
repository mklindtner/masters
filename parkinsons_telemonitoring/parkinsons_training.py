from parkinsons_telemonitoring.parkinsons_model import train_teacher_network, distillation_posterior_parkinsons
# from parkinsons_model import train_teacher_network, distillation_posterior_parkinsons
# from parkinsons_data import tr_optim, device, trainloader, testloader, tr_model, tr_criterion, tr_eval
from parkinsons_telemonitoring.parkinsons_data import tr_list, st_list, msc_list
from parkinsons_telemonitoring.parkinsons_stat_plot import plot_tr_results_distillation, save_results_to_csv, store_weights
from datetime import datetime

T_test = 3050
# results  = train_teacher_network(tr_optim=tr_optim, tr_network=tr_model, T_steps=T_test, 
#                                                     tr_loader_train=trainloader, tr_loader_valid=testloader, criterion=tr_criterion,
#                                                     device=device, tr_eval=tr_eval, verbose=True
#                                                     ) 

# plot_tr_results_teacher(results, label= "teacher_nll_plot")

results, tr_w, st_w = distillation_posterior_parkinsons(tr_list, st_list, msc_list, T_steps=T_test)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

plot_tr_results_distillation(results, timestamp=timestamp, T=T_test)
save_results_to_csv(results, timestamp=timestamp, T=T_test)
store_weights(tr_w,st_w, timestamp=timestamp, T=T_test)

