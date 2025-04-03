from models import gped2DNormal, gped2DNormal_student, design_matrix, mcmc_ULA, mcmc_MALA, mcmc_SGLD, posterior_expectation_distillation
import numpy as np
import torch
import matplotlib.pyplot as plt
from plotter import plot_all_samplers_2D, plot_simple_student_2D
import torch.optim as optim
# from torch.distributions.multivariate_normal import MultivariateNormal
# import math
# from torch.distributions.kl import kl_divergence
# from plotter import plotter,  plot_mcmc, plot_all_MCMC, plot_simple_student, plot_kl_divergence, plot_actual
# from statistics import weight_kl, E_weights, quantiles, row_statistic, row_statistic_actual



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


def algo_student_reg(w, x):
    inputs = torch.column_stack((torch.ones(len(x)), x))
    return inputs @ w[:,None]


algo2D = gped2DNormal(xtrain,ytrain, batch_sz=len(xtrain), alpha=alpha, beta=beta, prior_mean=prior_mean, D=2)
algo2D_student_simple = gped2DNormal_student(xtrain, ytrain, alpha=alpha, beta=beta, batch_sz=len(xtrain), prior_mean=prior_mean, D=2)

#MLE/MAP & distribution
Phi_train = design_matrix(algo2D.x)
w_MLE = np.linalg.solve(Phi_train.T@Phi_train, Phi_train.T@algo2D.y).ravel()
w_MAP = (beta*torch.linalg.solve(alpha*torch.eye(2) + beta*(Phi_train.T@Phi_train), Phi_train.T)@algo2D.y).ravel()

#Analytical mean and covariance for S,M for 2D example
S = torch.inverse(torch.eye(2) + beta * Phi_train.T @ Phi_train)
M = beta*S@Phi_train.T @ algo2D.y


w_MALA = mcmc_MALA(algo2D, theta_init=w_MAP, T=1000)
w_ULA = mcmc_ULA(algo2D, theta_init=w_MAP, T=1000, lr = 1e-2)
w_SGLD = mcmc_SGLD(algo2D, theta_init=w_MAP, T=1000)

#3 rows for samplers, 1 for actual values
#2 rows for student and teacher weights
_, axes = plt.subplots(6,3, figsize=(12,9))



#General posterior distillation

adam = optim.Adam([torch.tensor([[0.5,0.5]], requires_grad=True)])
loss = torch.nn.MSELoss(reduction='sum')

def algo_student_reg(w, x):
    inputs = torch.column_stack((torch.ones(len(x)), x))
    return inputs @ w[:,None]



#Plot expectation for simple student model
w_teacher, w_student = posterior_expectation_distillation(algo_teacher=algo2D, algo_student=algo2D_student_simple, theta_init=w_MAP, phi_init=w_MAP, f=algo_student_reg , reg=None, alphas = None, criterion=loss, opt=adam, T=1000)

#Plots
plot_all_samplers_2D(axes, w_MALA=w_MALA, w_ULA=w_ULA, w_SGLD=w_SGLD, M=M, S=S, algo2D=algo2D, w_MAP=w_MAP)
plot_simple_student_2D(axes, w_teacher, w_student, algo2D, w_MAP, w_MLE)


plt.show()