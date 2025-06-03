import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from toydata import StudentToyDataReqLin
import torch.optim as optim
from debugging import student_step5, student_step50, student_step1000, student_step2500, student_step5000

def SGLD_step(theta, eta_t, grad):
    D = theta.shape[0]
    z_t = MultivariateNormal(torch.tensor([0.0,0.0]), eta_t*torch.eye(D)).rsample()
    theta = (theta + eta_t/2 * grad + z_t).detach().clone().requires_grad_(True)
    return theta


#Use the entire dataset for |S| and |S'|
#Use alpha_s = 1 for all s
def distillation_expectation(algo2D, theta_init, phi_init, sgld_params, distil_params, f,g, loss, T=10000, logger=None):
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
    # samples_phi_iter = torch.empty((1, 2), dtype=phi.dtype, device=phi.device)
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

            if logger:
                if phi_iter_cnt == 5:
                    # samples_phi_iter[0,0] = f.fc1.bias.detach().clone()
                    # samples_phi_iter[0,1] = f.fc1.weight.detach().clone()
                    logger.add_student_weight(student_step5, f.fc1.bias.detach().clone(), f.fc1.weight.detach().clone())
                if phi_iter_cnt == 50:
                    logger.add_student_weight(student_step50, f.fc1.bias.detach().clone(), f.fc1.weight.detach().clone())
                if phi_iter_cnt == 1000:
                    logger.add_student_weight(student_step1000, f.fc1.bias.detach().clone(), f.fc1.weight.detach().clone())
                if phi_iter_cnt == 2500:
                    logger.add_student_weight(student_step2500, f.fc1.bias.detach().clone(), f.fc1.weight.detach().clone())
                if phi_iter_cnt == 5000:
                    logger.add_student_weight(student_step5000, f.fc1.bias.detach().clone(), f.fc1.weight.detach().clone())


            phi_iter_cnt += 1
            
            

        samples_theta[t] = theta.detach().clone()


    return (samples_theta, samples_phi)



def distillation_expectation_scalable(
    algo2D,
    theta_init,
    sgld_params,
    st_params,
    st_list, # List of (student_network, g_function, loss_function)
    T=10000,
    logger=None
    ):

    D_theta = theta_init.shape[0]
    N_data = algo2D.x.shape[0]

    # Initialize SGLD
    a, b, gamma, eta_sq_init = sgld_params
    theta = theta_init.detach().clone()
    samples_theta = torch.empty((T, D_theta), dtype=theta.dtype, device=theta.device)
    eta_sq = eta_sq_init

    # Initialize distillation parameters
    H, student_lr = st_params
    burn_in = int(0.10 * T)
    T_phi = int((T - burn_in) / H) if (T > burn_in) else 0

    # Generalize initialization for any number of students
    num_students = len(st_list)
    optimizers = [optim.Adam(f.parameters(), lr=student_lr) for f, g, loss in st_list]
    list_of_phi_samples = [torch.empty((T_phi, N_data), dtype=torch.float32, device=theta.device) for _ in range(num_students)]

    t_phi = 0
    phi_iter_cnt = 0

    for t in range(T):
        theta_detach = theta.detach().clone().requires_grad_(True)
        grad = algo2D.log_joint_gradient(theta_detach)
        if torch.isnan(grad).any(): raise ValueError(f"NaN in SGLD gradient at t={t}.")
        theta = SGLD_step(theta, eta_sq, grad)
        eta_sq = max(a / (b + t)**gamma, 1e-7)
        samples_theta[t] = theta.detach().clone()

        if t >= burn_in and (t - burn_in) % H == 0:
            current_theta_samples = samples_theta[:t+1]


            for i, (f, g, loss_fn) in enumerate(st_list):
                gfoo = g(algo2D.x, current_theta_samples)
                ghat = torch.mean(gfoo, dim=1)

                f.train()
                optimizers[i].zero_grad()
                gpred = f(algo2D.x).squeeze(-1)
                output = loss_fn(ghat.view_as(gpred), gpred)
                output.backward()
                optimizers[i].step()

                list_of_phi_samples[i][t_phi] = gpred.detach().clone()

                # Log results for the i-th student
                if logger and hasattr(f, 'fc1'):
                    w0_val = f.fc1.bias.item()
                    w1_val = f.fc1.weight[0,0].item()
                    w0_grad = f.fc1.bias.grad.item() 
                    w1_grad = f.fc1.weight.grad[0,0].item()

                    logger.logger_step(i, t, phi_iter_cnt, output, w0_val, w0_grad, w1_val, w1_grad)
                    # logger.log_step(
                    #     f"s{i+1}_tphi_{t_phi + 1}", t, output.item(),
                    #     w0_val, grad_w0, w1_val, grad_w1
                    # )
                    # if phi_iter_cnt in [5, 50, 100, 500, 1000, 2500, 5000]:
                    #      logger.add_student_weight(f"student{i+1}_step{phi_iter_cnt}", f.fc1.bias.detach().clone(), f.fc1.weight.detach().clone())

            t_phi += 1
            phi_iter_cnt += 1
    
    # Set all models to evaluation mode
    for f, g, loss in st_list:
        f.eval()

    return samples_theta, list_of_phi_samples
