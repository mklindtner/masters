from models import gped2DNormal, design_matrix,plot_weights, plot_distribution, analytical_gradient, mcmc_ULA
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import math


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



algo2D = gped2DNormal(xtrain,ytrain, alpha=alpha, beta=beta, prior_mean=prior_mean, D=2)

# #MLE/MAP
Phi_train = design_matrix(algo2D.x)
w_MLE = np.linalg.solve(Phi_train.T@Phi_train, Phi_train.T@algo2D.y).ravel()
w_MAP = (beta*torch.linalg.solve(alpha*torch.eye(2) + beta*(Phi_train.T@Phi_train), Phi_train.T)@algo2D.y).ravel()

w = mcmc_ULA(algo2D, theta_init=w_MAP, T=10000, lr = 1e-2/10)
# anal_w, m = analytical_gradient(theta_init, Phi, algo2D.x, beta)
# print(f"Analytical Gradient\n {anal_w}")
# print(f"log_joint_gradient-value\n {w}")
# print(f"difference\n {anal_w - w}")
# print("breakpoint")

fig, axes = plt.subplots(1, 3, figsize=(18,6))
axes[2].plot(w[:,1],w[:,0], "ro")
axes[2].plot(w_MAP[1],w_MAP[0], "bo")
#plot_weights(axes[1],algo=algo2D, thetas=w, color='g', title='Posterior', visibility=0.75)

algo2D.sim = False
plot_distribution(axes[0],density_fun=algo2D.log_prior, color='b', label='Prior', title='Prior', visibility=0.25)
plot_distribution(axes[1],density_fun=algo2D.log_likelihood, color='r', label='likelihood', title='Likelihood', visibility=0.25)
plot_distribution(axes[2],density_fun=algo2D.log_joint, color='g', label='Posterior', title='Posterior', visibility=0.25)




axes[1].plot(w_MLE[1], w_MLE[0], 'mo', label='MLE estimate')
axes[1].legend(loc='lower right')
# axes[2].plot(w_MLE[0], w_MLE[1], 'mo', label='MLE estimate')
# axes[2].plot(w_MAP[0], w_MAP[1], 'bo', label='MAP/Posterior mean')
# axes[2].legend(loc='lower right')
plt.show()
print("breakpoint")