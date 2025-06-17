# def distillation_posterior_MNIST(tr_items, st_items, msc_list, verbose=True):

#     tr_optim, tr_network, tr_loader_train, tr_loader_test = tr_items
#     st_network, st_optim, U = st_items
#     B, T_epochs, H, criterion, device = msc_list
    
#     # --- 2. Setup for the Main Loop ---
#     print(f"--- Starting Distillation Process for {T_epochs} epochs ---")
#     print(f"Burn-in period B set to {B} epochs.")
    
#     epoch_loop = tqdm(range(T_epochs), desc="Total Distillation Progress", disable=not verbose)

#     # --- 3. Main Training Loop ---
#     for t in epoch_loop:
#         tr_network.train()
        
#         # --- Teacher Training Step ---
#         batch_loop = tqdm(tr_loader_train, desc=f"Epoch {t+1}/{T_epochs} (Teacher Step)", leave=False, disable=not verbose)
#         for inputs, labels in batch_loop:
#             inputs, labels = inputs.to(device), labels.to(device)
#             inputs = inputs.view(inputs.size(0), -1)

#             tr_optim.zero_grad()
#             outputs = tr_network(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             tr_optim.step()

#         # --- Distillation Step (after burn-in) ---
#         if t >= B and (t % H == 0): 
#             if verbose:
#                 epoch_loop.set_postfix(Status="Distilling to Student")

#             distill_loop = tqdm(tr_loader_train, desc=f"Epoch {t+1} (Distilling)", leave=False, disable=not verbose)

#             for inputs, labels in distill_loop:
#                 inputs = inputs.to(device)
#                 inputs = inputs.view(inputs.size(0), -1)

#                 # Use our U_s function to get the teacher's latest prediction.
#                 # This is the target for the student.
#                 teacher_targets = U(tr_network, inputs)

#                 # TODO: Implement the student update step here.
#                 # The student (st_network) would now be trained to match
#                 # these teacher_targets using a distillation loss.
#                 pass
#         elif t < B:
#             epoch_loop.set_postfix(Status="Teacher Burn-in")
#         else:
#             epoch_loop.set_postfix(Status="Teacher Step")
    
#     print("--- Finished Distillation Process ---")
#     return None


# wip
# def distillation_expectation_mnist(
#     theta_init,
#     tr_loader,
#     st_loader,
#     st_params,
#     st_network,
#     tr_optim,    
#     tr_network,
#     T=10000,
#     logger=None
#     ):

#     # D_theta = theta_init.shape[0]

#     # N_data = algo2D.x.shape[0]

#     theta = theta_init.detach().clone()
#     samples_theta = torch.empty((T, tr_network.parameters()), dtype=theta.dtype, device=theta.device)

#     # Initialize Teacher
#     # tr_lr = tr_params
#     # optim_tr = optim.SGD(tr_network, lr=tr_lr)
#     tr_nll = []

#     # Initialize distillation parameters
#     H, st_lr, B = st_params
#     T_phi = (T - B) // H if ((T - B) % H == 0) else (T - B) // H + 1

#     # Initialize student
#     optim_st = optim.Adam(st_network.parameters(), lr=st_lr)
#     optim.lr_scheduler(optim_st, step_size=200, gamma=0.5)
#     samples_phi = torch.empty((T_phi, st_network.parameters()))
#     st_nll = [] 

    

#     t_phi = 0   

#     for t in range(T):
#         theta_detach = theta.detach().clone().requires_grad_(True)
        

#         # grad = algo2D.log_joint_gradient(theta_detach)
#         theta_next = 


#         if torch.isnan(grad).any(): raise ValueError(f"NaN in SGLD gradient at t={t}.")
#         # theta = SGLD_step(theta, eta_sq, grad)
#         # eta_sq = max(a / (b + t)**gamma, 1e-7)
#         samples_theta[t] = theta.detach().clone()

#         if t >= B and (t - B) % H == 0:
#             current_theta_samples = samples_theta[:t+1]


#             for i, (f, g, loss_fn, st_type) in enumerate(st_list):
#                 gfoo = g(algo2D.x, current_theta_samples)

#                 if st_type == f_SCALAR:
#                     ghat = torch.mean(gfoo, dim=1)
#                 else:
#                     ghat = gfoo

#                 f.train()
#                 optimizers[i].zero_grad()
#                 if st_type == f_SCALAR:
#                     gpred = f(algo2D.x).squeeze(-1)
#                     output = loss_fn(gpred, ghat.view_as(gpred))
#                     list_of_phi_samples[i]['predictions'][t_phi] = gpred.detach().clone()
#                     list_of_phi_samples[i]['st_w0'][t_phi] = f.fc1.bias.detach().clone()
#                     list_of_phi_samples[i]['st_w'][t_phi] = f.fc1.weight.detach().clone()
#                     list_of_phi_samples[i]['teacher_W'][t_phi] = torch.mean(current_theta_samples,axis=0)

#                 elif st_type == f_DIST: #PPD : Posterior Predictive Distribution
#                     gpred_mean, gpred_log_var = f(algo2D.x)
#                     gpred_var = torch.exp(gpred_log_var)

#                     #NLLNormalGauss is the loss function
#                     output = loss_fn(gpred_mean.squeeze(), ghat.squeeze(), gpred_var.squeeze())

#                     list_of_phi_samples[i]['mean'][t_phi] = gpred_mean.detach().clone().squeeze()
#                     list_of_phi_samples[i]['log_variance'][t_phi] = gpred_log_var.detach().clone().squeeze()
#                     list_of_phi_samples[i]['nll_loss'][t_phi] = output.item()


#                     # --- KL(st_pdd || teacher_pdd)---
#                     Phi_train = torch.cat([torch.ones_like(algo2D.x), algo2D.x], dim=1)
#                     l_mu = Phi_train @ current_theta_samples.T
#                     teacher_mu = torch.mean(l_mu,dim=1)
#                     teacher_var = 1.0/algo2D.beta + torch.mean(l_mu**2,dim=1) - teacher_mu**2 
#                     teacher_var = torch.clamp(teacher_var, 1e-6)
#                     teacher_pdd = torch.distributions.Normal(teacher_mu.squeeze(), torch.sqrt(teacher_var).squeeze())

#                     st_sqrt = torch.sqrt(gpred_var + 1e-8).squeeze()
#                     st_ppd = torch.distributions.Normal(gpred_mean.squeeze(), st_sqrt)
#                     kl_div_points_st_teacher =  torch.distributions.kl.kl_divergence(st_ppd, teacher_pdd)
#                     list_of_phi_samples[i]['kl_div_st_teacher'][t_phi] =  torch.mean(kl_div_points_st_teacher).item()

#                     kl_div_points_teacher_anal = torch.distributions.kl.kl_divergence(teacher_pdd, target_PDD)
#                     list_of_phi_samples[i]['kl_div_teacher_anal'][t_phi] = torch.mean(kl_div_points_teacher_anal).item()

#                 output.backward()
#                 optimizers[i].step()


#                 # Log results for the i-th student, not updated for kl / NLL
#                 if logger and hasattr(f, 'fc1'):
#                     w0_val = f.fc1.bias.item()
#                     w1_val = f.fc1.weight[0,0].item()
#                     w0_grad = f.fc1.bias.grad.item() 
#                     w1_grad = f.fc1.weight.grad[0,0].item()
#                     logger.logger_step(i, t, t_phi, output, w0_val, w0_grad, w1_val, w1_grad)                  

#             t_phi += 1
    
#     # Set all models to evaluation mode
#     for f, _, loss,_ in st_list:
#         f.eval()

#     return samples_theta, list_of_phi_samples

