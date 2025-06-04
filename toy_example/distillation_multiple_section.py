from toydata import theta_init, algo2D, SGLD_params, distil_params, T, st_list, xtrain, ytrain, colors_gradient, H

from debugging import StudentLogger, log_filename
from distillation import distillation_expectation_scalable
import matplotlib.pyplot as plt
import numpy as np
import torch


with StudentLogger(log_filepath=log_filename) as student_logger:
    teacher_samples, student_samples = distillation_expectation_scalable(
            algo2D, 
            theta_init=theta_init, 
            sgld_params=SGLD_params, 
            st_params=distil_params, 
            st_list=st_list,
            T=T, 
            logger=student_logger
        )
    
weight_teacher_final = teacher_samples[-1, :].cpu().numpy()
preds_student1_final = student_samples[0]['predictions'][-1, :].cpu().numpy()
preds_student2_final = student_samples[1]['predictions'][-1, :].cpu().numpy()

xtrain_np = xtrain.squeeze().cpu().numpy()
ytrain_np = ytrain.squeeze().cpu().numpy()
sort_indices = np.argsort(xtrain_np)
xtrain_sorted = xtrain_np[sort_indices]
ytrain_sorted = ytrain_np[sort_indices]
preds_st_linear_sorted = preds_student1_final[sort_indices]
preds_st_sq_sorted = preds_student2_final[sort_indices]
preds_teacher_sorted = weight_teacher_final[0] + weight_teacher_final[1]*xtrain_sorted


#Variance
epistemic_var_g = np.clip(preds_st_sq_sorted - preds_st_linear_sorted**2, min=0.0, max=None) 
epistemic_stdev_g = np.sqrt(epistemic_var_g)

#confidence interval bounds are +/- 2 standard deviations
upper_bound_g = preds_st_linear_sorted + 2 * epistemic_stdev_g
lower_bound_g = preds_st_linear_sorted - 2 * epistemic_stdev_g

# plt.figure(figsize=(12, 7)) 

# plot for mean, epistemic uncertainity and standard deviation of mean
# plt.plot(xtrain_sorted,  ytrain_sorted, 'k.', label='Data', markersize=12)
# plt.plot(xtrain_sorted, preds_teacher_sorted, linestyle='--', color='black', label=f'Teacher Bayesian Linear Fit:')
# plt.plot(xtrain_sorted, preds_st_linear_sorted, linestyle='-', color=colors_gradient["step_final"], label='Student 1 (g(x,y,w))', linewidth=2)
# plt.plot(xtrain_sorted, preds_st_sq_sorted, linestyle='-', color=colors_gradient["st_sq"], label='Student 2 (Quadratic g(x,y,w)^2)', linewidth=2)
# plt.fill_between(xtrain_sorted,
#                  lower_bound_g,
#                  upper_bound_g,
#                  color='blue',
#                  alpha=0.2, # Use alpha for transparency
#                  label='Epistemic Uncertainty (±2 std. dev.)')

# plt.plot(xtrain_sorted, upper_bound_g, color='blue', linestyle='-', linewidth=1.5)
# plt.plot(xtrain_sorted, lower_bound_g, color='blue', linestyle='-', linewidth=1.5)


# plt.xlabel("Input Feature (xtrain)")
# plt.ylabel("Output Value")
# plt.title(f"Student Network Predictions vs. Training Data: {T}-iterations")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()



ppd_st_mean = student_samples[2]['mean'][-1, :].cpu().numpy()
ppd_st_log_var = student_samples[2]['log_variance'][-1,:].cpu().numpy()
ppd_st_nll = student_samples[2]['NLL'].cpu().numpy()
ppd_st_kl = student_samples[2]['kl_div'].cpu().numpy()

burn_in = int(0.10 * T)
st_total_steps = (T - burn_in) // H if ((T - burn_in) % H == 0) else (T - burn_in) // H + 1
st_steps = np.arange(st_total_steps)
# --- Plot NLL Loss ---
plt.figure(figsize=(10, 6))
plt.plot(st_steps, ppd_st_nll, marker='.', linestyle='-', color='purple')
plt.title(f"NLL Loss for Distributional Student {st_total_steps}-steps")
plt.xlabel("Distillation Step Index (t_phi)")
plt.ylabel("Negative Log-Likelihood (NLL)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# --- Plot KL Divergence ---
plt.figure(figsize=(10, 6))
plt.plot(st_steps, ppd_st_kl, marker='.', linestyle='-', color='green')
plt.title(f"KL Divergence (Student PDD || Teacher Est. PDD) for {st_total_steps}-steps")
plt.xlabel("Distillation Step Index (t_phi)")
plt.ylabel("KL Divergence")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

