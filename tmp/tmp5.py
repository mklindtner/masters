import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

# Generate synthetic data
torch.manual_seed(42)
true_w = torch.tensor([1.0, 0.5])  # True parameters: w = [a, b]
true_cov = torch.tensor([[1.0, 0.5], [0.5, 1.0]])  # True covariance matrix

# Generate predictor variable x and observed data y
x = torch.linspace(-2, 2, 20)  # 20 points between -2 and 2
Phi = torch.stack([torch.ones_like(x), x], dim=1)  # Design matrix: [1, x]
mu = Phi @ true_w  # True mean: μ = a + b * x
data = dist.MultivariateNormal(mu, true_cov).sample()  # Observed data (20 x 1)

# Define the negative log-posterior (up to a constant)
def negative_log_posterior(w, Phi, data, prior_mean, prior_cov):
    """
    Compute the negative log-posterior for the parameters w.
    
    Args:
        w: Parameters (2D tensor: [a, b]).
        Phi: Design matrix (N x 2 tensor).
        data: Observed data (N x 1 tensor).
        prior_mean: Prior mean for w (2D tensor).
        prior_cov: Prior covariance for w (2x2 tensor).
    """
    # Compute the mean: μ = Φw
    mu = Phi @ w

    # Likelihood term: log p(data | w)
    likelihood = dist.MultivariateNormal(mu, true_cov).log_prob(data).sum()

    # Prior term: log p(w)
    prior = dist.MultivariateNormal(prior_mean, prior_cov).log_prob(w)

    # Negative log-posterior
    return -(likelihood + prior)

# SGLD Sampling for the parameters w
def sgld_sample_joint_prior(Phi, data, prior_mean, prior_cov, lr=1e-2, noise_scale=1e-2, num_steps=1000):
    """
    Perform SGLD sampling for the parameters w with a joint prior.
    
    Args:
        Phi: Design matrix (N x 2 tensor).
        data: Observed data (N x 1 tensor).
        prior_mean: Prior mean for w (2D tensor).
        prior_cov: Prior covariance for w (2x2 tensor).
        lr: Learning rate (step size).
        noise_scale: Scale of the Gaussian noise added to gradients.
        num_steps: Number of SGLD steps to perform.
    """
    # Initialize the parameters
    w = torch.tensor([0.0, 0.0], requires_grad=True)  # Parameters: [a, b]

    # Store the sampled parameters
    samples_w = []

    for step in range(num_steps):
        # Compute the negative log-posterior
        loss = negative_log_posterior(w, Phi, data, prior_mean, prior_cov)

        # Compute gradients
        loss.backward()

        # SGLD update
        with torch.no_grad():
            # Add Gaussian noise to the gradient
            noise = torch.randn_like(w.grad) * noise_scale
            w.grad.add_(noise)

            # Update parameters: θ = θ - lr * ∇θ + noise
            w.data.add_(-lr, w.grad)

            # Store the current parameters
            samples_w.append(w.detach().clone())

        # Zero gradients for the next step
        w.grad.zero_()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    return torch.stack(samples_w)

# Prior parameters (joint prior over w = [a, b])
prior_mean = torch.tensor([0.0, 0.0])  # Prior mean
prior_cov = torch.eye(2) * 10.0  # Weak prior covariance

# Perform SGLD sampling
samples_w = sgld_sample_joint_prior(Phi, data, prior_mean, prior_cov, lr=1e-2, noise_scale=1e-2, num_steps=1000)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(samples_w[:, 0].numpy(), samples_w[:, 1].numpy(), alpha=0.5, label="SGLD Samples")
plt.scatter(true_w[0], true_w[1], color="red", marker="x", s=100, label="True w")
plt.xlabel("a (Intercept)")
plt.ylabel("b (Slope)")
plt.title("SGLD Samples for w = [a, b]")
plt.legend()
plt.grid()
plt.show()