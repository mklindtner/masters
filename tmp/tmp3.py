import torch
import torch.distributions as dist

# Define the design matrix Phi (N x D)
N = 20  # Number of observations
D = 2   # Number of features (including bias)
Phi = torch.randn(N, D)  # Shape [N, D]

# Define the response variables y (N observations)
y = torch.randn(N)  # Shape [N]

# Define the weight vector w (D-dimensional)
w = torch.randn(D)  # Shape [D]

# Define the noise variance (sigma^2)
sigma = 1.0
variance = sigma**2

# Compute the mean (Phi * w)
mean = torch.mv(Phi, w)  # Shape [N]

# Define the covariance matrix (sigma^2 * I)
covariance = variance * torch.eye(N)  # Shape [N, N]

# Create the MultivariateNormal distribution for the likelihood
likelihood = dist.MultivariateNormal(mean, covariance)

# Compute the log probability of the response variables y
log_prob = likelihood.log_prob(y)  # Shape [] (scalar)
print("Log probability of y given w:", log_prob.item())