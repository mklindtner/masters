from toydata import theta_init, algo2D, SGLD_params, distil_params, T, st_list, xtrain, ytrain, colors_gradient

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
preds_student1_final = student_samples[0][-1, :].cpu().numpy()
preds_student2_final = student_samples[1][-1, :].cpu().numpy()

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


plt.figure(figsize=(12, 7)) 

#Fixed
plt.plot(xtrain_sorted,  ytrain_sorted, 'k.', label='Data', markersize=12)
plt.plot(xtrain_sorted, preds_teacher_sorted, linestyle='--', color='black', label=f'Teacher Bayesian Linear Fit:')
plt.plot(xtrain_sorted, preds_st_linear_sorted, linestyle='-', color=colors_gradient["step_final"], label='Student 1 (g(x,y,w))', linewidth=2)
plt.plot(xtrain_sorted, preds_st_sq_sorted, linestyle='-', color=colors_gradient["st_sq"], label='Student 2 (Quadratic g(x,y,w)^2)', linewidth=2)
# Plot the shaded variance region using fill_between
plt.fill_between(xtrain_sorted,
                 lower_bound_g,
                 upper_bound_g,
                 color='blue',
                 alpha=0.2, # Use alpha for transparency
                 label='Epistemic Uncertainty (±2 std. dev.)')

plt.plot(xtrain_sorted, upper_bound_g, color='blue', linestyle='-', linewidth=1.5)
plt.plot(xtrain_sorted, lower_bound_g, color='blue', linestyle='-', linewidth=1.5)


plt.xlabel("Input Feature (xtrain)")
plt.ylabel("Output Value")
plt.title(f"Student Network Predictions vs. Training Data: {T}-iterations")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()