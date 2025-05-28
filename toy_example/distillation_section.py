from distillation import distillation_expectation
from toydata import theta_init, phi_init, algo2D, MSEloss, f_student, SGLD_params, distil_params, T, g_bayesian_linear_reg, xtrain,ytrain, M
import torch.linalg as LA
import numpy as np # For plotting, if needed later
import matplotlib.pyplot as plt # For plotting, if needed later
import torch
from debugging import StudentLogger, log_filename

#student samples should be (samples, 20,2)??


with StudentLogger(log_filepath=log_filename) as student_logger:
    teacher_samples, student_samples, student_samples_few = distillation_expectation(
            algo2D, theta_init=theta_init, phi_init=phi_init, 
            sgld_params=SGLD_params, distil_params=distil_params, 
            f=f_student,g=g_bayesian_linear_reg,
            loss=MSEloss, T=T, 
            logger=student_logger
        )
    # print(student_logger.get_dataframe())


#Analytical and last sample for regression line
yhat_anal = M[0,0] + M[1,0]*algo2D.x
# yhat_student = student_samples[-1][0,0] + student_samples[-1][0,1]*algo2D.x


# yhat_st_weights = torch.tensor([f_student.fc1.bias.data, f_student.fc1.weight.data])
# yhat_teacher_final = torch.mean(g_bayesian_linear_reg(algo2D.x, teacher_samples), axis=1)   
yhat_teacher_final = torch.mean(teacher_samples,axis=0)[0] + torch.mean(teacher_samples,axis=0)[1]*algo2D.x
yhat_st_final = f_student.fc1.bias.data + f_student.fc1.weight.data*algo2D.x
yhat_st_five = student_samples_few[0,0] + student_samples_few[0,1]*algo2D.x
#Metric


#Plot over time w. norms values
# plt.plot(np.arange(len(l2_norms_over_time)), l2_norms_over_time.numpy(), marker='o', linestyle='-')









#Plot over regression
plt.subplots(figsize=(10, 6))
plt.plot(xtrain, ytrain, 'k.', label='Data', markersize=12)
plt.plot(xtrain, yhat_anal, color='green', linestyle='--', label=f'Analytical Bayesian Fit: y = {M[0,0]:.2f} + {M[1,0]:.2f}x')
plt.plot(xtrain, yhat_st_final, color = "red",  linestyle='--', label=f'Student Bayesian Fit m_{(T)}: y = {f_student.fc1.bias.data.item():.2f} + {f_student.fc1.weight.data.item():.2f}x')
plt.plot(xtrain, yhat_st_five, color="purple", linestyle="--", label= f'Student Bayesian Fit m_{(5)}: y = {student_samples_few[0,0].item():.2f} + {student_samples_few[0,1].item():.2f}x')
plt.plot(xtrain, yhat_teacher_final, color="blue", linestyle="--", label= f'Teaceher Bayesian Fit m_{(T)}: y = {teacher_samples[-1,0].item():.2f} + { teacher_samples[-1,1].item():.2f}x')


# plt.plot(xtrain, yhat_student_last, 'k.', label='Data', markersize=12)
# plt.plot(xtrain, yhat_student_last, color = "red",  linestyle='--', label=f'Student Bayesian Fit (m_N): y = {M[0,0]:.2f} + {M[1,0]:.2f}x')
plt.xlabel("Input x")
plt.ylabel("Response y")
plt.title(f"Bayesian Regression with: {T}-steps")
plt.legend()




# other plot about norms
#plt.plot(np.arange(len(l2_norm_time), l2_norm_time.numpy(), marker='o', linestyle='-'))
# plt.xlabel("Student Update Step")
# plt.ylabel("L2 Norm (|| Student Weights - Analytical Weights ||)")
# plt.title("L2 Norm of Weight Difference Over Distillation Steps")




plt.grid(True)
plt.tight_layout()
plt.show()




print(student_samples.shape)