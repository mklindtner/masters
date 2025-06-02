from toydata import theta_init, algo2D, SGLD_params, distil_params, T, st_list, xtrain, ytrain

from debugging import StudentLogger, log_filename
from distillation import distillation_expectation_scalable
import matplotlib.pyplot as plt
import numpy as np


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
    

preds_student1_final = student_samples[0][-1, :].cpu().numpy()
preds_student2_final = student_samples[1][-1, :].cpu().numpy()

xtrain_np = xtrain.squeeze().cpu().numpy()
ytrain_np = ytrain.squeeze().cpu().numpy()
sort_indices = np.argsort(xtrain_np)
xtrain_sorted = xtrain_np[sort_indices]
preds_student1_sorted = preds_student1_final[sort_indices]
preds_student2_sorted = preds_student2_final[sort_indices]


# plt.subplots(figsize=(10, 6))
plt.figure(figsize=(12, 7)) # Adjusted figsize slightly

plt.plot(xtrain_np, ytrain_np, 'k.', label='Data', markersize=12)
plt.plot(xtrain_sorted, preds_student1_sorted, 'b-', label='Student 1 (Linear g)', linewidth=2)
plt.plot(xtrain_sorted, preds_student2_sorted, 'g-', label='Student 2 (Quadratic g)', linewidth=2)

plt.xlabel("Input Feature (xtrain)")
plt.ylabel("Output Value")
plt.title("Student Network Predictions vs. Training Data")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()