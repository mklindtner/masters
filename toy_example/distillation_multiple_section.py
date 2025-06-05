from toydata import theta_init, algo2D, SGLD_params, distil_params, T, st_list, xtrain, ytrain, colors_gradient, H, M,S
from debugging import StudentLogger, log_filename
from distillation import distillation_expectation_scalable
import matplotlib.pyplot as plt
import numpy as np
from mkl_statistics import weight_kl, metric_mahalanobis_sq
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
preds_student1_final = student_samples[0]['predictions'][-1, :].cpu().numpy() #g1 predictions
preds_student2_final = student_samples[1]['predictions'][-1, :].cpu().numpy() #g2 predictions



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


#Student Steps 
burn_in = int(0.10 * T)
st_total_steps = (T - burn_in) // H if ((T - burn_in) % H == 0) else (T - burn_in) // H + 1
st_steps = np.arange(st_total_steps)


# # plot for mean, epistemic uncertainity and standard deviation of mean
# plt.figure(figsize=(12, 7)) 
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


#Plot squared Mahalanobis distance

#For linear weights and teacher vs student
st_W_lin = []
for bias, w in zip(student_samples[0]['st_w0'], student_samples[0]['st_w']):
    st_W_lin.append(torch.tensor([bias.item(),w.item()]))


st_W_lin = torch.stack(st_W_lin)
teacher_M = torch.stack(student_samples[0]['teacher_W'])
teacher_S = torch.cov(teacher_M.T) + torch.eye(teacher_M.T.shape[0])*1e-8
mahal_teacher_st = metric_mahalanobis_sq(teacher_M,st_W_lin, teacher_S)


#For analytical vs teacher
MM = M.T.repeat(teacher_M.shape[0], 1)
mahal_anal_teacher = metric_mahalanobis_sq(MM, teacher_M, S)


plt.plot(st_steps, mahal_anal_teacher, label="analytical/teacher", color="green")
plt.plot(st_steps, mahal_teacher_st, label="teacher/student", color="red")
plt.title(f"Mahalanobis Sq Distance for Posterior mean\n {st_total_steps}-steps")
plt.xlabel("student iterations")
plt.ylabel("Mahalnobis Sq Distance")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# print("ok")


# # --- Plot NLL Loss ---
# ppd_st_nll = student_samples[2]['nll_loss'].cpu().numpy()


# plt.figure(figsize=(10, 6))
# plt.plot(st_steps, ppd_st_nll, linestyle='-', color=colors_gradient["st_sq"], linewidth=1)
# plt.title(f"NLL Loss between Teacher and Student Distribution\n {st_total_steps}-steps")
# plt.xlabel("Distillation Step Index (t_phi)")
# plt.ylabel("Negative Log-Likelihood (NLL)")
# plt.ylim(0,3)
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()


# # --- Plot KL Divergence ---
# ppd_st_teacher_kl = student_samples[2]['kl_div_st_teacher'].cpu().numpy()
# ppd_teacher_anal_kl = student_samples[2]['kl_div_teacher_anal'].cpu().numpy()

# plt.figure(figsize=(10, 6))
# plt.plot(st_steps, ppd_st_teacher_kl, marker='.', linestyle='-', color='green', label="KL-Divergence for Teacher and Student",  linewidth=2)
# plt.plot(st_steps, ppd_teacher_anal_kl, linestyle='--', color='black', label="(Baseline) KL-Divergence for Teacher and analytical PPD ", linewidth=1.5)
# plt.title(f"KL( Student P|| Teacher PPD) and KL(Teacher_PPD || analytical_PPD) for {st_total_steps}-steps")
# plt.xlabel("Distillation Step Index (t_phi)")
# plt.ylabel("KL Divergence")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

