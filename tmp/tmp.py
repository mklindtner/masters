import torch
import torch.distributions as dist

# Set random seed for reproducibility
torch.manual_seed(42)

# Define dimensions
num_samples = 20  # Number of observations
num_features = 3  # Number of features
output_dim = 2    # Response variable is 2D

# Generate random data
Phi = torch.randn(num_samples, num_features)  # Design matrix (20x3)
w = torch.randn(num_features, output_dim)    # Weight matrix (3x2)
y = torch.randn(num_samples, output_dim)     # Response variables (20x2)

# Define variance (sigma^2)
sigma = 1.0
variance = sigma**2

# Compute mean (Phi * w)
mean = torch.mm(Phi, w)  # Mean of the Gaussian (20x2)

# Define covariance matrix (sigma^2 * I)
# For a 2D Gaussian, the covariance matrix is 2x2
covariance = variance * torch.eye(output_dim)  # Covariance matrix (2x2)

# Create a MultivariateNormal distribution for each sample
log_probs = []
for i in range(num_samples):
    gaussian_dist = dist.MultivariateNormal(loc=mean[i], covariance_matrix=covariance)
    log_prob = gaussian_dist.log_prob(y[i])  # Log likelihood for each response
    log_probs.append(log_prob)

# Convert log_probs to a tensor
log_probs = torch.tensor(log_probs)

# Print the log probabilities
print("Log probabilities of the response variables:")
print(log_probs)

# Total log likelihood (sum of individual log probabilities)
total_log_likelihood = log_probs.sum()
print("\nTotal log likelihood:", total_log_likelihood.item())