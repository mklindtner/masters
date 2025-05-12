from toydata import theta_init, target, algo2D
from statistics import weight_kl
import torch
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence

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


T = 1060
RW_samples = MHRW(algo2D=algo2D, theta_init=theta_init, T=T)


#kl_divergences at various time T
def kl_RW(RW_samples, target):
    offset = 5
    x = torch.arange(offset,len(RW_samples))
    kl_rw = torch.zeros(len(RW_samples)-offset)    
    for i, id in enumerate(x):
        w_s = RW_samples[0:id]
        W_est = torch.mean(w_s, axis=0)        
        W_shat = torch.cov(w_s.T) + torch.eye(2) * 1e-6
        W_guess = MultivariateNormal(W_est, W_shat)
        kl_rw[i] = kl_divergence(W_guess, target).item()   
    return kl_rw


#sample baseline
def kl_baseline(target):
    target_samples = target.rsample((RW_samples.shape[0],))
    mhat_true = torch.mean(target_samples,axis=0)
    shat_true = torch.cov(target_samples.T)
    sample_true = MultivariateNormal(mhat_true, shat_true)
    KL_true = kl_divergence(sample_true, target)
    return KL_true

kl_rw = kl_RW(RW_samples, target)
kl_target = kl_baseline(target)

t = torch.arange(5,len(RW_samples))
plt.axhline(kl_target, label="Random Sample from Analytical posterior", linestyle="--")
plt.plot(t, kl_rw, label="MHRW KL", color = "green")
plt.tight_layout()
# plt.show(block=False)
print("kek")