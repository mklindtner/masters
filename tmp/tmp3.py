import torch

# Input vector with 2 elements, requires_grad=True so we can compute Jacobian
x = torch.tensor([1.0, 2.0], requires_grad=True)

# Define a function that outputs a vector of 2 elements
y = torch.stack([
    x[0] + 2 * x[1],   # y1
    3 * x[0] - x[1]    # y2
])

# We'll collect rows of the Jacobian
jacobian = []

# Loop over each output and compute gradient w.r.t. input
for i in range(len(y)):
    grad_outputs = torch.zeros_like(y)
    grad_outputs[i] = 1.0  # We select y[i]
    
    # Compute dy[i]/dx
    grads = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=grad_outputs, retain_graph=True)[0]
    jacobian.append(grads)

# Stack into 2D matrix
J = torch.stack(jacobian)

print("Jacobian:\n", J)
