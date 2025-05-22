from toydata import theta_init, phi_init, target, algo2D, MSEloss, f_student, SGLD_params, distil_params, T
from statistics import weight_kl
import torch
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
from models import mcmc_MALA, mcmc_ULA, mcmc_SGLD
from distillation import distillation_expectation

def MHRW(algo2D, theta_init, T=10000):
    theta = theta_init.detach().clone().requires_grad_(True)
    chain = [None]*T
    eta = 1
    for t in range(T):
        thetaprime = theta.detach() + eta*torch.randn_like(theta)
        #algo2D.log_joint(theta+g).item() would yield the same
        log_proposal = (algo2D.log_likelihood(thetaprime) + algo2D.log_prior(thetaprime))
        log_current = (algo2D.log_likelihood(theta) + algo2D.log_prior(theta))
        
        A = torch.min(torch.tensor([0]), log_proposal - log_current)
        u = torch.log(torch.rand(1))
        if u <= A:
            theta = thetaprime
            # print("chose propsal")
        chain[t] = theta.detach().clone()
        

    return torch.stack(chain)

def kl_sample(W_samples, target,offset):
    D = W_samples[0].shape[0]
    x = torch.arange(offset,len(W_samples))
    kl_samples = torch.zeros(len(W_samples)-offset)    
    jitter_eps = 1e-6
    for i, id in enumerate(x):
        w_s = W_samples[0:id]
        W_est = torch.mean(w_s, axis=0)        
        W_shat = torch.cov(w_s.T) + torch.eye(D) * jitter_eps
        W_guess = MultivariateNormal(W_est, W_shat)
        kl_samples[i] = kl_divergence(W_guess, target).item()   
    return kl_samples

#sample baseline
def kl_baseline(target, T):
    target_samples = target.rsample((T,))
    mhat_true = torch.mean(target_samples,axis=0)
    shat_true = torch.cov(target_samples.T)
    sample_true = MultivariateNormal(mhat_true, shat_true)
    KL_true = kl_divergence(sample_true, target)
    return KL_true






# SGLD_samples = mcmc_SGLD(algo=algo2D, theta_init=theta_init, T=T)
# MALA_samples = mcmc_MALA(algo=algo2D, theta_init=theta_init, T=T)
# ULA_samples = mcmc_ULA(algo=algo2D, theta_init=theta_init, T=T)
# RW_samples = MHRW(algo2D=algo2D, theta_init=theta_init, T=T)
teacher_samples, student_samples = distillation_expectation(algo2D, theta_init=theta_init, phi_init=phi_init, sgld_params=SGLD_params, distil_params=distil_params, f=f_student,loss=MSEloss, T=T)


offset = 5
sample_sz = T
# kl_rw = kl_sample(RW_samples, target, offset)
# kl_mala = kl_sample(MALA_samples, target, offset)
# kl_ula = kl_sample(ULA_samples, target, offset)
# kl_sgld = kl_sample(SGLD_samples, target, offset)
kl_distillation_teacher = kl_sample(teacher_samples, target, offset)
kl_distillation_student = kl_sample(student_samples, target, offset)
kl_target = kl_baseline(target, sample_sz)

t = torch.arange(5,sample_sz)
burn_in, H = distil_params
t_student = torch.arange(burn_in, T, H)[offset:]
plt.axhline(kl_target, label="Random Sample from Analytical posterior", linestyle="--")
# plt.plot(t, kl_rw, label="MHRW KL", color = "green")
# plt.plot(t, kl_mala, label="MALA KL", color= "brown")
# plt.plot(t, kl_ula, label="ULA KL", color="blue")
# plt.plot(t, kl_sgld, label="SGLD KL", color="red")
plt.plot(t, kl_distillation_teacher, label="distillation_teacher_samples", color="purple")
plt.plot(t_student, kl_distillation_student, label="distillation_student_sampels", color="pink")
# plt.ylim(0,0.2)       
plt.legend()
plt.tight_layout()
plt.show()