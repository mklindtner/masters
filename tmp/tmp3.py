import torch

# Create input data
x = torch.linspace(0, 1, steps=10).unsqueeze(1)  # shape: (10, 1)

# True parameters (theta)
theta_0 = torch.tensor([2.0])
theta_1 = torch.tensor([3.0])

# Generate y using true parameters (no noise for simplicity)
y_true = theta_0 + theta_1 * x


# Initialize phi parameters with requires_grad=True to track gradients
phi_0 = torch.randn(1, requires_grad=True)
phi_1 = torch.randn(1, requires_grad=True)

learning_rate = 0.1

for i in range(100):
    # Compute prediction
    y_pred = phi_0 + phi_1 * x

    # Compute mean squared error loss
    loss = torch.mean((y_pred - y_true) ** 2)

    # Backpropagation
    loss.backward()

    # Gradient descent update (manually, without optimizer)
    with torch.no_grad():
        phi_0 -= learning_rate * phi_0.grad
        phi_1 -= learning_rate * phi_1.grad

        # Zero the gradients after updating
        phi_0.grad.zero_()
        phi_1.grad.zero_()

    if i % 10 == 0:
        print(f"Step {i}: Loss = {loss.item():.4f}, phi_0 = {phi_0.item():.4f}, phi_1 = {phi_1.item():.4f}")
