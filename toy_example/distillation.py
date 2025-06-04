import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.distributions
from toydata import f_SCALAR, f_DIST
import torch.optim as optim
from debugging import student_step5, student_step50, student_step1000, student_step2500, student_step5000
from toydata import beta

def SGLD_step(theta, eta_t, grad):
    D = theta.shape[0]
    z_t = MultivariateNormal(torch.tensor([0.0,0.0]), eta_t*torch.eye(D)).rsample()
    theta = (theta + eta_t/2 * grad + z_t).detach().clone().requires_grad_(True)
    return theta


#Use the entire dataset for |S| and |S'|
#Use alpha_s = 1 for all s
#We assume f is scalar type
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
    T_phi = (T - burn_in) // H if ((T - burn_in) % H == 0) else (T - burn_in) // H + 1

    # Generalize initialization for any number of students
    # num_students = len(st_list)
    optimizers = [optim.Adam(f.parameters(), lr=student_lr) for f, _, _, _ in st_list]
    # list_of_phi_samples = [torch.empty((T_phi, N_data), dtype=torch.float32, device=theta.device) for _ in range(num_students)]
    list_of_phi_samples = []

    
    for _, _, _, st_type_from_list in st_list: 
        if st_type_from_list == f_SCALAR:
            list_of_phi_samples.append({
                'type': f_SCALAR,
                'predictions': torch.empty((T_phi, N_data), dtype=torch.float32, device=theta.device)
            })
        else:
            list_of_phi_samples.append({
                'type': f_DIST,
                'mean': torch.empty((T_phi, N_data), dtype=torch.float32, device=theta.device),
                'log_variance': torch.empty((T_phi, N_data), dtype=torch.float32, device=theta.device),
                'NLL': torch.empty((T_phi), dtype=torch.float32, device=theta.device),
                'kl_div': torch.empty((T_phi), dtype=torch.float32, device=theta.device)
            })

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


            for i, (f, g, loss_fn, st_type) in enumerate(st_list):
                gfoo = g(algo2D.x, current_theta_samples)

                if st_type == f_SCALAR:
                    ghat = torch.mean(gfoo, dim=1)
                else:
                    ghat = gfoo

                f.train()
                optimizers[i].zero_grad()
                if st_type == f_SCALAR:
                    gpred = f(algo2D.x).squeeze(-1)
                    output = loss_fn(gpred, ghat.view_as(gpred))
                    list_of_phi_samples[i]['predictions'][t_phi] = gpred.detach().clone()

                elif st_type == f_DIST: #PPD : Posterior Predictive Distribution
                    gpred_mean, gpred_log_var = f(algo2D.x)
                    gpred_var = torch.exp(gpred_log_var)

                    #NLLNOrmalGaus
                    output = loss_fn(gpred_mean.squeeze(), ghat.squeeze(), gpred_var.squeeze())

                    # output = loss_fn(ghat.squeeze(), gpred_mean.squeeze(), gpred_log_var.squeeze())  #assummes NLL metric
                    list_of_phi_samples[i]['mean'][t_phi] = gpred_mean.detach().clone().squeeze()
                    list_of_phi_samples[i]['log_variance'][t_phi] = gpred_log_var.detach().clone().squeeze()
                    list_of_phi_samples[i]['NLL'][t_phi] = output.item()


                    # --- KL(st_pdd || teacher_pdd)---
                    Phi_train = torch.column_stack((algo2D.x, torch.ones(len(algo2D.x))))
                    ll_mu = Phi_train @ current_theta_samples.T
                    teacher_mu = torch.mean(ll_mu,dim=1)
                    teacher_var = 1/beta + torch.mean(ll_mu**2,dim=1) - teacher_mu**2 
                    teacher_var = torch.clamp(teacher_var, 1e-6)
                    teacher_pdd = torch.distributions.Normal(teacher_mu.squeeze(), torch.sqrt(teacher_var).squeeze())

                    st_sqrt = torch.sqrt(gpred_var + 1e-8).squeeze()
                    st_ppd = torch.distributions.Normal(gpred_mean.squeeze(), st_sqrt)
                    kl_div_per_point =  torch.distributions.kl.kl_divergence(st_ppd, teacher_pdd)
                    list_of_phi_samples[i]['kl_div'][t_phi] =  torch.mean(kl_div_per_point).item()

                output.backward()
                optimizers[i].step()


                # Log results for the i-th student
                if logger and hasattr(f, 'fc1'):
                    w0_val = f.fc1.bias.item()
                    w1_val = f.fc1.weight[0,0].item()
                    w0_grad = f.fc1.bias.grad.item() 
                    w1_grad = f.fc1.weight.grad[0,0].item()
                    logger.logger_step(i, t, phi_iter_cnt, output, w0_val, w0_grad, w1_val, w1_grad)                  

            t_phi += 1
            phi_iter_cnt += 1
    
    # Set all models to evaluation mode
    for f, _, loss,_ in st_list:
        f.eval()

    return samples_theta, list_of_phi_samples
