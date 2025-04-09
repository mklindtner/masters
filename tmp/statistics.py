import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
import matplotlib.pyplot as plt



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


    # pest_shat = torch.tensor([pest_shat[0,0], pest_shat[1,1]])
       
    
    target_samples = target.rsample((sample_sz,))
    mhat_true = torch.mean(target_samples,axis=0)
    shat_true = torch.cov(target_samples.T)
    sample_true = MultivariateNormal(mhat_true, shat_true)
    KL_true = kl_divergence(sample_true, target)

    return W_stat, KL_true, target_samples

# def E_weights(W_samples):
#     # return 1/len(W_samples) * torch.sum(W_samples,dim=0)
#     muhat = torch.mean(W_samples, axis=0)
#     shat = torch.cov(W_samples.T)
#     foo = MultivariateNormal(muhat, shat).samples((len(W_samples),))
#     return torch.mean(foo), 


# def quantiles_mu(statistic):
#     # beta0 = statistic[:,0]
#     # beta1 = statistic[:,1]
#     quantiles = torch.tensor([0.025, 0.975], dtype=statistic.dtype)
#     ci_beta0 = torch.quantile(statistic[0,0], quantiles)
#     ci_beta1 = torch.quantile(statistic[0,1], quantiles)
#     return ci_beta0, ci_beta1

# def quantiles_var(statistic):
#     quantiles = torch.tensor([0.025, 0.975], dtype=statistic.dtype)
#     return torch.quantile(statistic, )

# def quantile_M_S(): 

# def table_statistics(ax, MALA_samples, ULA_samples, SGLD_samples, target):



def sampler_row_statistic(ax, W_samples, cell_txt, row_labels):

    quantiles = torch.tensor([0.025, 0.975])
    ci_mu1 = torch.quantile(W_samples[:,0], quantiles)
    ci_mu2 = torch.quantile(W_samples[:,1], quantiles)
    sigmahat_ = torch.cov(W_samples.T)


    muhat0_str = f"({ci_mu1[0]:.3f}, {ci_mu1[1]:.3f})"
    muhat1_str = f"({ci_mu2[0]:.3f}, {ci_mu2[1]:.3f})"

    cell_txt.append([muhat0_str, f"{sigmahat_[0,0]:.3f}"])
    cell_txt.append([muhat1_str, f"{sigmahat_[1,1]:.3f}"])


    col_labels = [r'95% CI', r'Variance Point estimate ']

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
    ax.set_title("Statistics")
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