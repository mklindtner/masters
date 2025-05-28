import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from toydata import StudentToyDataSimple1
import torch.optim as optim

def SGLD_step(theta, eta_t, grad):
    D = theta.shape[0]
    z_t = MultivariateNormal(torch.tensor([0.0,0.0]), eta_t*torch.eye(D)).rsample()
    theta = (theta + eta_t/2 * grad + z_t).detach().clone().requires_grad_(True)
    return theta


#Use the entire dataset for |S| and |S'|
#Use alpha_s = 1 for all s
def distillation_expectation(algo2D, theta_init, phi_init, sgld_params, distil_params, f, g, loss, T=10000, logger=None):
    D = theta_init.shape[0]
    obs_shape = algo2D.x.T.shape[1]

    #Initiale SGLD weights
    a,b,gamma,eta_sq = sgld_params
    theta = theta_init.detach().clone().requires_grad_(True)
    samples_theta = torch.empty((T,D), dtype=theta.dtype, device=theta.device)

    #Inititalize distillation weights
    H,alpha_s = distil_params
    burn_in = int(0.10*T)

    T_phi = int((T-burn_in) / H)
    t_phi = 0
    phi = phi_init.detach().clone().requires_grad_(True)

    samples_phi = torch.empty((T_phi,obs_shape), dtype=phi.dtype, device=phi.device)


    #Initialize "student weight captures for various iterations"
    samples_phi_iter = torch.empty((1, 2), dtype=phi.dtype, device=phi.device)
    phi_iter_cnt = 0

    assert type(T_phi) == int, f"{T-burn_in / H} is not an integer, distillation epochs must be integers."

    #Inititalize optimizer etc.    
    ghat = 0
    optimizer = optim.Adam(f.parameters(), lr=alpha_s)

    for t in range(T):
        grad = algo2D.log_joint_gradient(theta)

        if torch.isnan(grad).any():
            raise(f"NaN in SGLD gradient at t={t}. Stopping.")

        theta = SGLD_step(theta, eta_sq, grad)
        eta_sq = max(a/(b+t)**gamma,1e-7)         

        if t > burn_in and t % H == 0:
            gfoo = g(algo2D.x, samples_theta[:t])            
            ghat = torch.mean(gfoo, dim=1)

            #Keep all samples in memory            
            gpred = f(algo2D.x)

            f.zero_grad()
            output = loss(ghat.view_as(gpred), gpred)
            output.backward()

            if logger:
                    w0_val = f.fc1.bias.item()
                    w1_val = f.fc1.weight.item() 
                    
                    grad_w0 = f.fc1.bias.grad.item() 
                    grad_w1 = f.fc1.weight.grad.item()
                    
                    logger.log_step(
                        t_phi + 1, t, output.item(),
                        w0_val, grad_w0, w1_val, grad_w1
                    )
            
            # print(logger.get_dataframe())

            optimizer.step()

            #add phi weights here?
            samples_phi[t_phi] = gpred.detach().clone().squeeze(1)
            t_phi += 1

            if phi_iter_cnt == 5:
                samples_phi_iter[0,0] = f.fc1.bias.detach().clone()
                samples_phi_iter[0,1] = f.fc1.weight.detach().clone()
            phi_iter_cnt += 1
            
            

        samples_theta[t] = theta.detach().clone()


    return (samples_theta, samples_phi, samples_phi_iter)