import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from toydata import StudentToyData
import torch.optim as optim

def SGLD_step(theta, eta_t, grad):
    D = theta.shape[0]
    z_t = MultivariateNormal(torch.tensor([0.0,0.0]), eta_t*torch.eye(D)).rsample()
    theta = (theta + eta_t/2 * grad + z_t).detach().clone().requires_grad_(True)
    return theta


#Use the entire dataset for |S| and |S'|
#Use alpha_s = 1 for all s
def distillation_expectation(algo2D, theta_init, phi_init, sgld_params, distil_params, f, loss, T=10000):
    D = theta_init.shape[0]

    #Initiale SGLD weights
    a,b,gamma,eta_sq = sgld_params
    theta = theta_init.detach().clone().requires_grad_(True)
    samples_theta = torch.empty((T,D), dtype=theta.dtype, device=theta.device)

    #Inititalize distillation weights
    H, burn_in = distil_params
    T_phi = int((T-burn_in) / H)
    t_phi = 0
    phi = phi_init.detach().clone().requires_grad_(True)
    samples_phi = torch.empty((T_phi,D), dtype=phi.dtype, device=phi.device)

    assert type(T_phi) == int, f"{T-burn_in / H} is not an integer, distillation epochs must be integers."

    #Inititalize optimizer etc.    
    ghat = 0
    optimizer = optim.SGD(f.parameters(), lr=0.1)

    for t in range(T):
        grad = algo2D.log_joint_gradient(theta)
        theta = SGLD_step(theta, eta_sq, grad)
        eta_sq = max(a/(b+t)**gamma,1e-7)         

        if t > burn_in and t % H == 0:
            #choose g(y,x,theta_t) = theta_t
            ghat = theta    
            gpred = torch.log(f(algo2D.x.flatten()))

            f.zero_grad()
            output = loss(ghat, gpred)
            output.backward()
            optimizer.step()

            #add phi weights here?
            samples_phi[t_phi] = gpred
            t_phi += 1
            

        samples_theta[t] = theta


    return (samples_theta, samples_phi)