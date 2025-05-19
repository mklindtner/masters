import torch
from torch.distributions.multivariate_normal import MultivariateNormal

def SGLD_step(theta, eta_t, grad):
    D = theta.shape[0]
    z_t = MultivariateNormal(torch.tensor([0.0,0.0]), eta_t*torch.eye(D)).rsample()
    theta = (theta + eta_t/2 * grad + z_t).detach().clone().requires_grad_(True)
    return theta



def distillation_expectation(algo2D, theta_init, phi_init, sgld_params, distil_params, T=10000):

    D = theta_init.shape[0]


    #Initiale SGLD weights
    a,b,gamma,eta_sq = sgld_params
    theta = theta_init.detach().clone().requires_grad_(True)
    samples_theta = torch.empty((T,D), dtype=theta.dtype, device=theta.device)

    #Inititalize distillation weights
    H, burn_in = distil_params
    T_phi = int((T-burn_in) / H)
    phi = phi_init.detach().clone().requires_grad_(True)
    samples_phi = torch.empty((T_phi,D), dtype=phi.dtype, device=phi.device)

    assert type(T_phi) == int, f"{T-burn_in / H} is not an integer, distillation epochs must be integers."


    for t in range(T):
        grad = algo2D.log_joint_gradient(theta)
        theta = SGLD_step(theta, eta_sq, grad)
        eta_sq = max(a/(b+t)**gamma,1e-7)         

        samples_theta[t] = theta


    return (samples_theta, samples_phi)