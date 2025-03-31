import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
import matplotlib.pyplot as plt



def weight_kl(W_samples, target_dst):
    """
        calculate the KL-divergence between all samples and a target distribution
        Assumes the samples to be normal
        
    """
    kl_divs = [None]*W_samples.shape[0]

    for id, W in enumerate(W_samples):
        guess = MultivariateNormal(W, covariance_matrix=torch.eye(2))
        kl_divs[id] = kl_divergence(guess, target_dst).item()

    return kl_divs, min(kl_divs), max(kl_divs)


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