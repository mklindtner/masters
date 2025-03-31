from models import gped2DNormal, design_matrix, plot_distribution, analytical_gradient, mcmc_ULA, mcmc_MALA, mcmc_SGLD, posterior_expectation_distillation
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
import matplotlib.pyplot as plt
import math
from plotter import plotter,  plot_mcmc, plot_all_MCMC, plot_simple_student, plot_kl_divergence, plot_actual
import torch.optim as optim
from statistics import weight_kl, E_weights, quantiles, row_statistic, row_statistic_actual


#Example week3
alpha = 5
beta = 3/4
xtrain = torch.tensor([1.764, 0.4, 0.979, 2.241, 1.868, -0.977,  0.95, -0.151, -0.103, 0.411, 0.144, 1.454, 0.761, 0.122,
              0.444, 0.334, 1.494, -0.205,  0.313, -0.854])[:,None]
ytrain = torch.tensor([-0.464, 2.024, 3.191, 2.812, 6.512, -3.022, 1.99, 0.009, 2.513, 3.194, 0.935, 3.216, 0.386, -2.118,
               0.674, 1.222, 4.481, 1.893, 0.422,  -1.209])[:,None]

N = 20
sz = 2
T = 1000
prior_mean = torch.tensor([0,0])
theta_init = torch.tensor([0.0,0.0], requires_grad=True)



algo2D = gped2DNormal(xtrain,ytrain, batch_sz=len(xtrain), alpha=alpha, beta=beta, prior_mean=prior_mean, D=2)

#MLE/MAP & distribution
Phi_train = design_matrix(algo2D.x)
w_MLE = np.linalg.solve(Phi_train.T@Phi_train, Phi_train.T@algo2D.y).ravel()
w_MAP = (beta*torch.linalg.solve(alpha*torch.eye(2) + beta*(Phi_train.T@Phi_train), Phi_train.T)@algo2D.y).ravel()

#S,M
S = torch.inverse(torch.eye(2) + beta * Phi_train.T @ Phi_train)
M = beta*S@Phi_train.T @ algo2D.y

#General posterior distillation

adam = optim.Adam([torch.tensor([[0.5,0.5]], requires_grad=True)])
loss = torch.nn.MSELoss(reduction='sum')

def algo_student_reg(w, x):
    inputs = torch.column_stack((torch.ones(len(x)), x))
    return inputs @ w[:,None]



w_MALA = mcmc_MALA(algo2D, theta_init=w_MAP, T=1000)
w_ULA = mcmc_ULA(algo2D, theta_init=w_MAP, T=1000, lr = 1e-2)
w_SGLD = mcmc_SGLD(algo2D, theta_init=w_MAP, T=1000)


target = MultivariateNormal(loc=M.T.squeeze(), covariance_matrix=S)

#Get statistics
divergences_MALA, kl_min_MALA, kl_max_MALA = weight_kl(w_MALA, target)
divergences_ULA, kl_min_ULA, kl_max_ULA = weight_kl(w_ULA, target)
divergences_SGLD, kl_min_SGLD, kl_max_SGLD = weight_kl(w_SGLD, target)



#Plot statistics
_, axes = plt.subplots(4,3, figsize=(12,9))


id = 0
#w_MALA 
row_statistic(axes[id][0], w_MALA)
plot_kl_divergence(axes[id][id+1], divergences_MALA, kl_min_MALA,kl_max_MALA)
plot_mcmc(axes[id][id+2],'MALA', w_MALA, algo2D, w_MAP)

#w_ULA
row_statistic(axes[id+1][0], w_ULA)
plot_kl_divergence(axes[id+1][id+1], divergences_ULA, kl_min_ULA, kl_max_ULA)
plot_mcmc(axes[id+1][id+2],'ULA', w_ULA, algo2D, w_MAP)


#SGLD
row_statistic(axes[id+2][0], w_SGLD)
plot_kl_divergence(axes[id+2][id+1], divergences_SGLD, kl_min_SGLD, kl_max_SGLD)
plot_mcmc(axes[id+2][id+2],'SGLD', w_SGLD, algo2D, w_MAP)



#Actual
row_statistic_actual(axes[id+3][0], M.T)
axes[id+3][1].axis("off")
plot_distribution(axes[id+3][2],density_fun=algo2D.log_joint, color='g', label='Posterior', title='Posterior', visibility=0.25)

# axes[id+3][2].axis("off")

# plot_actual(axes[id+4], algo2D, w_MAP, w_MLE)




#etc.
plt.tight_layout()
plt.show()





# torch.nn.functional(guess.log_prob(algo2D.y), target.log_prob(algo2D.y))


#Plot expectation for simple student model
# w_teacher, w_gen = posterior_expectation_distillation(algo_teacher=algo2D, algo_student=algo_student_reg, theta_init=w_MAP, phi_init=w_MAP, reg=None, alphas = None, criterion=loss, opt=adam, T=10000)

#Plot Posterior expectation distillation
# plot_simple_student(w_teacher, w_gen, algo2D, w_MAP, w_MLE)


#plot all samplers
#plot_all_MCMC(algo2D, w_MAP)


#Plot single mcmc
# w_ULA = mcmc_ULA(algo2D, theta_init=w_MAP, T=1000, lr = 1e-2)
# w_MALA = mcmc_MALA(algo2D, theta_init=w_MAP, T=1000)
# # w_MALA = mcmc_MALA(algo2D, theta_init=torch.tensor([0.0,0.0], requires_grad=True), T=1000)
# w_SGLD = mcmc_SGLD(algo2D, theta_init=w_MAP, T=1000)
# algo2D.sim = False
# plotter(w_SGLD, algo2D, w_MAP, w_MLE)
# plotter(w_MALA, algo2D, w_MAP, w_MLE)
# plotter(w_ULA)