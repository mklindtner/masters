import torch
import torch.distributions as dist

# Define the input observations x (20 observations, 1 feature each)
x = torch.randn(20, 1)  # Shape [20, 1]

# Define the response variables y (20 observations, 1 response each)
y = torch.randn(20, 1)  # Shape [20, 1]

# Define the weight vector w (2D vector, including a bias term)
w = torch.tensor([0.5, 1.0])  # Shape [2]

# Define the variance of the noise
sigma = 1.0
variance = sigma**2

# Construct the design matrix Phi (20x2)
# For a linear model with a bias term, Phi = [1, x]
Phi = torch.cat([torch.ones(20, 1), x], dim=1)  # Shape [20, 2]

# Compute the mean (Phi * w)
mean = torch.mm(Phi, w.unsqueeze(1)).squeeze()  # Shape [20]

# Define the covariance matrix (sigma^2 * I)
# Since y is scalar, the covariance is a scalar variance
covariance = variance * torch.ones(20)  # Shape [20]

# Create the Normal distribution (scalar responses)
normal_dist = dist.Normal(mean, covariance.sqrt())

# Check batch shape and event shape
print("Batch shape:", normal_dist.batch_shape)  # Output: torch.Size([20])
print("Event shape:", normal_dist.event_shape)  # Output: torch.Size([])

# Compute the log probability of the response variables y
log_prob = normal_dist.log_prob(y.squeeze())  # Shape [20]
print("Log probabilities:", log_prob)