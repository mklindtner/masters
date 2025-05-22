from distillation import distillation_expectation
from toydata import theta_init, phi_init, target, algo2D, MSEloss, f_student, SGLD_params, distil_params, T, g_mean

teacher_samples, student_samples = distillation_expectation(algo2D, theta_init=theta_init, phi_init=phi_init, sgld_params=SGLD_params, distil_params=distil_params, f=f_student,g=g_mean ,loss=MSEloss, T=T)


print(student_samples.shape)