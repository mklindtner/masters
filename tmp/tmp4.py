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

# Create the Normal distribution for the likelihood
# Each observation y_i is modeled as a univariate Gaussian
likelihood = dist.Normal(mean, torch.sqrt(torch.tensor(variance)))

# Compute the log probability of the response variables y
log_prob = likelihood.log_prob(y)  # Shape [N]
print("Log probabilities:", log_prob)

# Total log likelihood (sum of individual log probabilities)
total_log_likelihood = log_prob.sum()
print("Total log likelihood:", total_log_likelihood.item())