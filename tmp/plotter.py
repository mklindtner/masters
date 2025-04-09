from models import plot_distribution, mcmc_MALA, mcmc_SGLD, mcmc_ULA
import matplotlib.pyplot as plt
# from torch.distributions import kl_divergence
import torch
import numpy as np
from statistics import weight_kl, sampler_row_statistic
from torch.distributions.multivariate_normal import MultivariateNormal


def plot_kl_divergence(ax, kl_divergences, kl_min, kl_max):
    x_axis = range(1,len(kl_divergences) + 1)
    ax.plot(x_axis, kl_divergences)
    title = f"Time Vs kl_div\n min/max: {kl_min:.0f}:{kl_max:.0f}"
    ax.set(xlabel='iteration', ylabel='KL-Divergence', title=title)



def plot_actual(axes, algo2D, w_MAP, w_MLE):
    plot_distribution(axes[1],density_fun=algo2D.log_likelihood, color='r', label='likelihood', title='Likelihood', visibility=0.25)
    plot_distribution(axes[2],density_fun=algo2D.log_joint, color='g', label='Posterior', title='Posterior', visibility=0.25)
    plot_distribution(axes[0],density_fun=algo2D.log_prior, color='b', label='Prior', title='Prior', visibility=0.25)
    axes[1].plot(w_MLE[1], w_MLE[0], 'mo', label='MLE estimate')
    axes[1].legend(loc='lower right')
    
    axes[2].plot(w_MAP[1], w_MAP[0], 'bo', label='MAP/Posterior mean')
    axes[2].legend(loc='lower right')
    


def plotter(w, algo2D, w_MAP, w_MLE):
    _, axes = plt.subplots(1, 3, figsize=(18,6))
    axes[2].plot(w[:,1],w[:,0], "ro", label="estimated weights")
    algo2D.sim = False
    plot_distribution(axes[1],density_fun=algo2D.log_likelihood, color='r', label='likelihood', title='Likelihood', visibility=0.25)
    plot_distribution(axes[2],density_fun=algo2D.log_joint, color='g', label='Posterior', title='Posterior', visibility=0.25)
    plot_distribution(axes[0],density_fun=algo2D.log_prior, color='b', label='Prior', title='Prior', visibility=0.25)
    axes[1].plot(w_MLE[1], w_MLE[0], 'mo', label='MLE estimate')
    axes[1].legend(loc='lower right')
    
    axes[2].plot(w_MAP[1], w_MAP[0], 'bo', label='MAP/Posterior mean')
    axes[2].legend(loc='lower right')
    plt.show()



def plot_mcmc(axis,name, w, algo2D, w_MAP):
    axis.plot(w[:,1],w[:,0], "ro", label="samples")
    algo2D.sim = False
    axis.plot(w_MAP[1], w_MAP[0], 'bo', label='MAP')
    axis.legend(loc='lower right',fontsize='small', markerscale=0.3)
    plot_distribution(axis,density_fun=algo2D.log_joint, color='g', label='Posterior', title=name, visibility=0.25)


def plot_all_MCMC(algo2D, w_MAP):
    w_ULA = mcmc_ULA(algo2D, theta_init=w_MAP, T=1000, eps = 1e-2)
    w_MALA = mcmc_MALA(algo2D, theta_init=w_MAP, T=1000)
    w_SGLD = mcmc_SGLD(algo2D, theta_init=w_MAP, T=1000)

    # #Plot all mcmc
    _, axes = plt.subplots(1, 3, figsize=(18,6))

    plot_mcmc(w_ULA, algo2D, w_MAP, axes[0], 'ULA')
    plot_mcmc(w_MALA, algo2D, w_MAP, axes[1], 'MALA')
    plot_mcmc(w_SGLD, algo2D, w_MAP, axes[2], 'SGLD')
    plt.show()


def plot_samplers_2D(axis, w_MALA, w_ULA, w_SGLD, target):
    
    sample_sz = w_MALA.shape[0]
    KL_MALA, KL_baseline, sample_true = weight_kl(W_samples=w_MALA, target=target)
    KL_ULA, _,_ =  weight_kl(W_samples=w_ULA, target=target)
    KL_SGLD, _,_= weight_kl(W_samples=w_SGLD, target=target) 
    

    t = torch.arange(0,sample_sz-5)
    t_sgld = torch.arange(0,sample_sz-200-5)
    axis[1].axhline(KL_baseline, linestyle="--")
    axis[1].plot(t, KL_MALA, label="MALA KL", color = "green")
    axis[1].plot(t, KL_ULA, label="ULA KL", color ="red")
    axis[1].plot(t_sgld, KL_SGLD, label="SGLD KL", color="purple")
    axis[1].legend(loc="upper right")
    axis[1].grid(True)
    axis[1].set(title="Sampler KL", xlabel="iterations", ylabel="KL_Divergence", ylim=(0,1))

    row_labels = [r'MALA $\mu_0$, $\sigma_0$', r'MALA $\mu_1$, $\sigma_1$']
    cell_txt = sampler_row_statistic(axis[0], w_MALA, [], row_labels)
    row_labels += [r'ULA $\mu_0$, $\sigma_0$', r'ULA $\mu_1$, $\sigma_1$']
    cell_txt = sampler_row_statistic(axis[0], w_ULA, cell_txt, row_labels)
    row_labels += [r'SGLD $\mu_0$, $\sigma_0$', r'SGLD $\mu_1$, $\sigma_1$']
    cell_txt = sampler_row_statistic(axis[0], w_SGLD, cell_txt, row_labels)
    row_labels += [r'actual $\mu_0$, $\sigma_0$', r'actual $\mu_1$, $\sigma_1$']
    sampler_row_statistic(axis[0],sample_true, cell_txt, row_labels)


    plt.tight_layout
    plt.show()


def plot_sample_2D_solo(axes, w_sample, target, cell_txt, row_label, color='green'):
    KL_Sample, KL_baseline, _ = weight_kl(W_samples=w_sample, target=target)
    t = torch.arange(0,len(w_sample)-5)
    axes[1].axhline(KL_baseline, linestyle="--")
    axes[1].plot(t, KL_Sample, label="Teacher KL", color = color)
    # row_labels = [r'Teacher $\mu_0$, $\sigma_0$', r'Teacher $\mu_1$, $\sigma_1$']
    # cell_txt = sampler_row_statistic(axes[0], w_sample, cell_txt, row_label)
    # return  cell_txt



#Depricated
def plot_simple_student_2D(axes, w_teacher, w_student, algo2D, w_MAP, w_MLE):
    algo2D.sim = False

    print(f'MLE: {w_MLE}\t MAP:{w_MAP}')
    x = torch.arange(1, w_student.shape[0]+1)
    axes[4][0].plot(x,w_student[:,1])
    axes[4][0].set(xlabel='T', ylabel='intercept', title='Student phi_0 weights')
    axes[4][1].plot(x,w_student[:,0])
    axes[4][1].set(xlabel='T', ylabel='slope', title='Student phi_1 weights')
    axes[4][2].axis("off")

    x_teacher = torch.arange(1, w_teacher.shape[0]+1)

    axes[5][0].plot(x_teacher,w_teacher[:,1])
    axes[5][0].set(xlabel='T', ylabel='intercept', title='Teacher phi_0 weights')
    axes[5][1].plot(x_teacher,w_teacher[:,0])
    axes[5][1].set(xlabel='T', ylabel='slope', title='Teacher phi_1 weights')
    axes[5][2].axis("off")
    plt.tight_layout()



    