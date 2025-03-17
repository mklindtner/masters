from models import gped2DNormal, design_matrix, plot_distribution, analytical_gradient, mcmc_ULA, mcmc_MALA, mcmc_SGLD
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import math
from plotter import plotter


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

#MLE/MAP
Phi_train = design_matrix(algo2D.x)
w_MLE = np.linalg.solve(Phi_train.T@Phi_train, Phi_train.T@algo2D.y).ravel()
w_MAP = (beta*torch.linalg.solve(alpha*torch.eye(2) + beta*(Phi_train.T@Phi_train), Phi_train.T)@algo2D.y).ravel()

# w_ULA = mcmc_ULA(algo2D, theta_init=w_MAP, T=1000, lr = 2e-2)
# w_MALA = mcmc_MALA(algo2D, theta_init=torch.tensor([0.0,0.0], requires_grad=True), T=1000)
w_SGLD = mcmc_SGLD(algo2D, theta_init=w_MAP, T=1000)

algo2D.sim = False
plotter(w_SGLD, algo2D, w_MAP, w_MLE)
# plotter(w_MALA, algo2D, w_MAP, w_MLE)
# plotter(w_ULA)