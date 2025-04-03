import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
import matplotlib.pyplot as plt



def posterior_params_samplers(w_MALA, w_ULA, w_SGLD, target):
    w_MALA = torch.mean()



def weight_kl(MALA_samples, ULA_samples, SGLD_samples, target):
    x = torch.arange(5,len(MALA_samples))
    KL_MALA = [None]*len(x)
    KL_ULA = [None]*len(x)
    KL_SGLD = [None]*len(x)

    for i, id in enumerate(x):
        MALA_w_s = MALA_samples[0:id]
        MALA_mhat = torch.mean(MALA_w_s, axis=0)
        MALA_shat = torch.cov(MALA_w_s.T)
        MALA_guess = MultivariateNormal(MALA_mhat, MALA_shat)
        KL_MALA[i] = kl_divergence(MALA_guess, target).item()

        ULA_w_s = ULA_samples[0:id]
        ULA_mhat = torch.mean(ULA_w_s, axis=0)
        ULA_shat = torch.cov(ULA_w_s.T)
        ULA_guess = MultivariateNormal(ULA_mhat, ULA_shat)
        KL_ULA[i] = kl_divergence(ULA_guess,target).item()

        SGLD_w_s = SGLD_samples[0:id]
        SGLD_mhat = torch.mean(SGLD_samples,axis=0)
        SGLD_shat = torch.cov(SGLD_w_s.T)
        SGLD_guess = MultivariateNormal(SGLD_mhat, SGLD_shat)
        KL_SGLD[i] = kl_divergence(SGLD_guess, target).item()
    
    sample_sz = MALA_samples.shape[0]
    target_samples = target.rsample((sample_sz,))
    mhat_true = torch.mean(target_samples,axis=0)
    shat_true = torch.cov(target_samples.T)
    sample_true = MultivariateNormal(mhat_true, shat_true)
    KL_true = kl_divergence(sample_true, target)

    return KL_MALA, KL_ULA, KL_SGLD, KL_true

def E_weights(W_samples):
    return 1/len(W_samples) * torch.sum(W_samples,dim=0)


def quantiles(W_samples):
    beta0 = W_samples[:,0]
    beta1 = W_samples[:,1]
    quantiles = torch.tensor([0.025, 0.975], dtype=W_samples.dtype)
    ci_beta0 = torch.quantile(beta0, quantiles)
    ci_beta1 = torch.quantile(beta1, quantiles)
    return ci_beta0, ci_beta1


def row_statistic(ax, W_samples):
    expectation_weights = E_weights(W_samples)
    ci_beta0, ci_beta1 = quantiles(W_samples)        
    cell_txt = []

    b0_mean_str = f"{expectation_weights[0]:.3f}"
    ci_beta0_str = f"({ci_beta0[0]:.3f},{ci_beta0[1]:.3f})"
    b1_mean_str = f"{expectation_weights[1]:.3f}"
    ci_beta1_str =  f"({ci_beta1[0]:.3f},{ci_beta1[1]:.3f})"
    # b0_actual_str = f"{w_actual[:,0][0]:.3f}"
    # b1_actual_str=  f"{w_actual[:,1][0]:.3f}"

    cell_txt = [
        [b0_mean_str, ci_beta0_str],
        [b1_mean_str, ci_beta1_str]
        # [b0_actual_str, "-"], 
        # [b1_actual_str, "-"]
    ]

    # row_labels = [r'$\beta_0$', r'$\beta_1$', r'actual $\beta_0$', r'actual $\beta_1$']
    row_labels = [r'$\beta_0$', r'$\beta_1$']
    col_labels = ["Expectation", "95% Credible Interval"]

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
    # ax.set_title("Sample Statistics")

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