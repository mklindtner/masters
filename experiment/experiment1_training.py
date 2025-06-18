from experiment1_models import train_teacher_network, distillation_posterior_MNIST
from experiment1_data import tr_items, st_items, msc_items, T, B, H
import matplotlib.pyplot as plt 
from datetime import datetime
from experiment1_stat_plot import save_results_to_csv, plot_results


T_test = 1000
#I needed to make sure teacher network worked as expected
# tr_nll, st_nll = train_teacher_network(tr_optim=tr_optim, tr_network=tr_model, T_epochs=T_test, 
#                                            tr_loader_train=trainloader, tr_loader_valid=testloader, criterion=criterion,
#                                            device=device, verbose=True
#                                            ) 

results = distillation_posterior_MNIST(tr_items=tr_items, st_items=st_items, msc_list=msc_items, T_total=T_test)
plot_results(results)
save_results_to_csv(results)



