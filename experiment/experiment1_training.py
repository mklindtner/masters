from experiment.experiment1_models import train_teacher_network, distillation_posterior_MNIST
from experiment.experiment1_data import tr_items, st_items, msc_items, T, B, H, tr_optim, trainloader, testloader, criterion, tr_model, device
import matplotlib.pyplot as plt 
from datetime import datetime
from experiment.experiment1_stat_plot import save_results_to_csv, plot_results, store_weights

T_test = 50000

results, st_w, tr_w = distillation_posterior_MNIST(tr_items=tr_items, st_items=st_items, msc_list=msc_items, T_total=T_test)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_results(results, T=T_test, timestamp=timestamp)
save_results_to_csv(results, T=T_test, timestamp=timestamp)
# store_weights(st_w, tr_w, T=T_test, timestamp=timestamp)



#For testing sgld
# theta, tr = train_teacher_network(tr_optim=tr_optim, tr_network=tr_model, T_epochs=T_test, 
#                                            tr_loader_train=trainloader, tr_loader_valid=testloader, criterion=criterion,
#                                            device=device, verbose=True
#                                            ) 

# plot_results(results, T=T_test, timestamp=timestamp)

