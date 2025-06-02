from distillation import distillation_expectation
from toydata import theta_init, phi_init, algo2D, MSEloss, f_student, SGLD_params, distil_params, T, g_bayesian_linear_reg, g_meansq, xtrain,ytrain, M, f_student_sq
import torch.linalg as LA
import numpy as np # For plotting, if needed later
import matplotlib.pyplot as plt # For plotting, if needed later
import torch
from debugging import StudentLogger, log_filename, student_step5, student_step50, student_step1000, student_step2500, student_step5000



with StudentLogger(log_filepath=log_filename) as student_logger:
    teacher_samples, student_samples = distillation_expectation(
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


#student linear regression
yhat_teacher_final = torch.mean(teacher_samples,axis=0)[0] + torch.mean(teacher_samples,axis=0)[1]*algo2D.x
st_yhat_final = f_student.fc1.bias.data + f_student.fc1.weight.data*algo2D.x
st_yhat5 = student_step5[0,0] + student_step5[0,1]*algo2D.x
st_yhat50 = student_step50[0,0] + student_step50[0,1]*algo2D.x
st_yhat1000 = student_step1000[0,0] + student_step1000[0,1]*algo2D.x
st_yhat2500 = student_step2500[0,0] + student_step2500[0,1]*algo2D.x


#student non linear regression
st_yhat_final_nonlinear =  f_student_sq.fc1.bias.data + f_student_sq.fc1.weight.data*algo2D.x


colors_gradient = {
    'teacher': 'k',             # Black for ground truth
    'student_final': 'red',   # Green for the final converged model
    'step_5': '#4682B4',        # Lightest blue
    'step_50': 'yellow',       # Medium-light blue
    'step_1000': 'blue',      # Medium-dark blue
    'step_2500': '#4169E1',       # Darkest blue (approaching final)
    'step_final': '#3F51B5'
}


#Plot over time w. norms values
# plt.plot(np.arange(len(l2_norms_over_time)), l2_norms_over_time.numpy(), marker='o', linestyle='-')



plt.subplots(figsize=(10, 6))
plt.plot(xtrain, ytrain, 'k.', label='Data', markersize=12)
#Plot over linear regression
plt.plot(xtrain, yhat_anal, color='black', linestyle='--', label=f'Analytical Bayesian Fit: y = {M[0,0]:.2f} + {M[1,0]:.2f}x')
plt.plot(xtrain, yhat_teacher_final, color="grey", linestyle="--", label= f'Teacher Bayesian Fit m_{(T)}: y = {teacher_samples[-1,0].item():.2f} + { teacher_samples[-1,1].item():.2f}x')
# plt.plot(xtrain, st_yhat_final, color = colors_gradient['step_final'],  linestyle='--', label=f'Student Bayesian Fit m_{(T)}: y = {f_student.fc1.bias.data.item():.2f} + {f_student.fc1.weight.data.item():.2f}x')
# plt.plot(xtrain, st_yhat5, color=colors_gradient["step_5"], linestyle="--", label= f'Student Bayesian Fit m_{(5)}: y = {student_step5[0,0].item():.2f} + {student_step5[0,1].item():.2f}x')
# plt.plot(xtrain, st_yhat50, color=colors_gradient["step_50"], linestyle="--", label= f'Student Bayesian Fit m_{(50)}: y = {student_step50[0,0].item():.2f} + {student_step50[0,1].item():.2f}x')
# plt.plot(xtrain, st_yhat1000, color=colors_gradient["step_1000"], linestyle="--", label= f'Student Bayesian Fit m_{(1000)}: y = {student_step1000[0,0].item():.2f} + {student_step1000[0,1].item():.2f}x')
# plt.plot(xtrain, st_yhat2500, color=colors_gradient["step_2500"], linestyle="--", label= f'Student Bayesian Fit m_{(2500)}: y = {student_step2500[0,0].item():.2f} + {student_step2500[0,1].item():.2f}x')


#Plot over Non linear regression
plt.plot(xtrain, st_yhat_final_nonlinear, color="red",  label= f'Student Bayesian Fit sq m_{(T)}: y = { f_student_sq.fc1.bias.data.item():.2f} + { f_student_sq.fc1.weight.data.item():.2f}x')


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