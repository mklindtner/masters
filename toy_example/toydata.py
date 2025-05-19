import torch
from models import gped2DNormal, gped2DNormal_student, design_matrix
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np



alpha = 5
beta = 3/4
xtrain = torch.tensor([1.764, 0.4, 0.979, 2.241, 1.868, -0.977,  0.95, -0.151, -0.103, 0.411, 0.144, 1.454, 0.761, 0.122,
              0.444, 0.334, 1.494, -0.205,  0.313, -0.854])[:,None]
ytrain = torch.tensor([-0.464, 2.024, 3.191, 2.812, 6.512, -3.022, 1.99, 0.009, 2.513, 3.194, 0.935, 3.216, 0.386, -2.118,
               0.674, 1.222, 4.481, 1.893, 0.422,  -1.209])[:,None]

N = 20
sz = 2
# T = 30000
prior_mean = torch.tensor([0,0])
theta_init = torch.tensor([0.0,0.0], requires_grad=True)
phi_init = torch.tensor([0.0,0.0], requires_grad=True)

algo2D = gped2DNormal(xtrain,ytrain, batch_sz=len(xtrain), alpha=alpha, beta=beta, prior_mean=prior_mean, D=2)
algo2D_student_simple = gped2DNormal_student(xtrain, ytrain, alpha=alpha, beta=beta, batch_sz=len(xtrain), prior_mean=prior_mean, D=2)



#MLE/MAP & distribution
Phi_train = design_matrix(algo2D.x)
w_MLE = np.linalg.solve(Phi_train.T@Phi_train, Phi_train.T@algo2D.y).ravel()
w_MAP = (beta*torch.linalg.solve(alpha*torch.eye(2) + beta*(Phi_train.T@Phi_train), Phi_train.T)@algo2D.y).ravel()

#Analytical mean and covariance for S,M for 2D example
S = torch.inverse(alpha*torch.eye(2) + beta * Phi_train.T @ Phi_train)
M = beta*S@Phi_train.T @ algo2D.y
target = MultivariateNormal(loc=M.T.squeeze(), covariance_matrix=S)