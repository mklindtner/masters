import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
import matplotlib.pyplot as plt
import torch

def metric_mahalanobis_sq(mu_target, muhat, S_inv):  
    num_steps = muhat.shape[0]
    distances_sq = torch.empty(num_steps, dtype=mu_target.dtype, device=mu_target.device)

    for k in range(num_steps):
        # current_mu_estimate = muhat[k]
        diff = mu_target[k] - muhat[k] # Shape [D_theta]
        dist_sq = diff @ S_inv @ diff
        distances_sq[k] = dist_sq.item() 
    return distances_sq


def weight_kl(W_samples, target):
    x = torch.arange(5,len(W_samples))
    W_stat = torch.zeros(len(W_samples)-5)    
    sample_sz = W_samples.shape[0]
    muhat = None; shat = None
    
    for i, id in enumerate(x):
        w_s = W_samples[0:id]
        W_est = torch.mean(w_s, axis=0)        
        W_shat = torch.cov(w_s.T)
        W_guess = MultivariateNormal(W_est, W_shat)
        W_stat[i] = kl_divergence(W_guess, target).item()   
       
    
    target_samples = target.rsample((sample_sz,))
    mhat_true = torch.mean(target_samples,axis=0)
    shat_true = torch.cov(target_samples.T)
    sample_true = MultivariateNormal(mhat_true, shat_true)
    KL_true = kl_divergence(sample_true, target)

    return W_stat, KL_true, target_samples



def sampler_row_statistic(ax, W_samples, cell_txt, row_labels):

    quantiles = torch.tensor([0.025, 0.975])
    ci_mu1 = torch.quantile(W_samples[:,0], quantiles)
    ci_mu2 = torch.quantile(W_samples[:,1], quantiles)
    sigmahat_ = torch.cov(W_samples.T)


    muhat0_str = f"({ci_mu1[0]:.3f}, {ci_mu1[1]:.3f})"
    muhat1_str = f"({ci_mu2[0]:.3f}, {ci_mu2[1]:.3f})"

    cell_txt.append([muhat0_str, f"{sigmahat_[0,0]:.3f}"])
    cell_txt.append([muhat1_str, f"{sigmahat_[1,1]:.3f}"])


    col_labels = [r'$\mu$ 95% CI', r'Variance Point estimate ']

    ax.axis('off') # Turn off axis lines and ticks for the table plot
    table = ax.table(cellText=cell_txt,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     loc='center',       # Position table in the center
                     cellLoc='center')   # Center text within cells
    # Adjust formatting
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5) # Adjust width and height scaling factors
    # ax.set_title("Statistics")
    ax.set_title("Marginal statistics")
    return cell_txt

def row_statistic_actual(ax, W):
    b0_mean_str = f"{W[:,0].item():.3f}"
    b1_mean_str = f"{W[:,1].item():.3f}"
    cell_txt = [
        [b0_mean_str],
        [b1_mean_str]
    ]
    row_labels = [r'$\beta_0$', r'$\beta_1$']
    col_labels = ["True parameters"]
    ax.axis('off') # Turn off axis lines and ticks for the table plot
    table = ax.table(cellText=cell_txt,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     loc='center',       # Position table in the center
                     cellLoc='center')   # Center text within cells
    # Adjust formatting
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5) # Adjust width and height scaling factors