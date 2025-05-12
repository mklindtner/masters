from GPED import alpha, beta
from toydata import theta_init, target
from GPED import algo2D
from statistics import weight_kl
import torch
import matplotlib.pyplot as plt

def MHRW(algo2D, theta_init, target, T=100):
    theta = theta_init.detach().clone().requires_grad_(True)
    accepted_proposals = [None]*T
    for t in T:
        g = torch.rand()
        proposal = algo2D.log_likelihood(theta+g) + algo2D.log_prior(theta+g)
        current = target.log_prob(theta) + algo2D.log_prior(theta)
        
        A = torch.min(1, proposal - current)
        u = torch.rand()
        if torch.min(1,A) <= u:
            accepted_proposals[t] = u
            print("chose u")
        else:
            theta = proposal
            accepted_proposals[t] = proposal
            print("chose propsal")
    return torch.stack(accepted_proposals)



RW_samples = MHRW(algo2D=algo2D, theta_init=theta_init, T=1060)



    
# sample_sz = w_MALA.shape[0]
KL_RW, KL_baseline, sample_true = weight_kl(W_samples=RW_samples, target=target)


t = torch.arange(0,len(RW_samples))
plt.axhline(KL_baseline, label="Random Sample from Analytical posterior", linestyle="--")
plt.plot(t, KL_RW, label="MALA KL", color = "green")
plt.tight_layout()
plt.show(block=False)
