from models import mcmc_ULA, mcmc_MALA, mcmc_SGLD
import torch
import matplotlib.pyplot as plt
from plotter import plot_samplers_2D
from toydata import algo2D, w_MAP, target


#Sampling
T = 1000
w_MALA = mcmc_MALA(algo2D, theta_init=torch.tensor([0.0,0.0]), T=T, h_sq=1e-2/2)
w_ULA = mcmc_ULA(algo2D, theta_init=w_MAP, T=T, h_sq = 1e-2/2)
w_SGLD = mcmc_SGLD(algo2D, theta_init=w_MAP, T=T, eps=1e-2/2)


# Plots for samplers
_, axes = plt.subplots(1,1, figsize=(12,9))
plot_samplers_2D(axes, w_MALA, w_ULA, w_SGLD, target)
# plt.tight_layout()
plt.show(block=False)
print("kek")






#General posterior distillation

# adam = optim.Adam([torch.tensor([[0.5,0.5]], requires_grad=True)])
# loss = torch.nn.MSELoss(reduction='sum')

# def algo_student_reg(w, x):
#     inputs = torch.column_stack((torch.ones(len(x)), x))
#     return inputs @ w[:,None]

# w_teacher, w_student = posterior_expectation_distillation(algo_teacher=algo2D, algo_student=algo2D_student_simple, theta_init=w_MAP, phi_init=w_MAP, f=algo_student_reg , reg=None,criterion=loss, opt=adam, T=T)

# _, axes = plt.subplots(1,2, figsize=(12,9))

# row_labels = [r'SGLD Teacher $\mu_0$, $\sigma_0$', r'SGLD Teacher $\mu_1$, $\sigma_1$']
# plot_sample_2D_solo(axes, w_teacher, target, [], row_labels, label = "Teacher")
# plot_sample_2D_solo(axes, w_SGLD, target, None, row_labels, color='red', label = "SGLD")
# plot_sample_2D_solo(axes, w_student, target, None, row_labels, color='purple', label = "Student")


# plt.tight_layout
# plt.show()