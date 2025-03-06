#SImpler example
# xtrain = torch.tensor([1.0,2.0,3.0,4.0])
# ytrain = torch.tensor([2.0,3.0,4.0,5.0])
# beta = 1
# alpha = 1   
# T = 100
# prior_mean = torch.tensor([0,0])
# theta_init = torch.tensor([0.0,0.0], requires_grad=True)


#Test mcmc
# w = mcmc(algo2D,theta_init=theta_init, lr=1e-3, T=T)
# # Analytical gradient for comparison
# Phi = torch.column_stack((torch.ones_like(xtrain), xtrain))
# theta = theta_init.view(-1, 1)  # Reshape theta to 2x1
# Phi_theta = torch.matmul(Phi, theta)  # Result is 4x1
# analytical_gradient = torch.matmul(Phi.t(), (ytrain - Phi_theta.squeeze())) - theta.squeeze()