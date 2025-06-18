from parkinsons_model import train_teacher_network, distillation_posterior_parkinsons
# from parkinsons_data import tr_optim, device, trainloader, testloader, tr_model, tr_criterion, tr_eval
from parkinsons_data import tr_list, st_list, msc_list
from parkinsons_stat_plot import plot_tr_results_teacher, plot_tr_results_distillation

T_test = 20000
# results  = train_teacher_network(tr_optim=tr_optim, tr_network=tr_model, T_steps=T_test, 
#                                                     tr_loader_train=trainloader, tr_loader_valid=testloader, criterion=tr_criterion,
#                                                     device=device, tr_eval=tr_eval, verbose=True
#                                                     ) 

# plot_tr_results_teacher(results, label= "teacher_nll_plot")

results = distillation_posterior_parkinsons(tr_list, st_list, msc_list, T_steps=T_test)

plot_tr_results_distillation(results)


