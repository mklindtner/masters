from models import plot_distribution, mcmc_MALA, mcmc_SGLD, mcmc_ULA
import matplotlib.pyplot as plt
from torch.distributions import kl_divergence
import torch
import numpy as np


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
    # axes[2].plot(w[:,0],w[:,1], "yo", label="estimated weights opposite")

    # axes[2].plot(w_MAP[1],w_MAP[0], "bo")
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
    w_ULA = mcmc_ULA(algo2D, theta_init=w_MAP, T=1000, lr = 1e-2)
    w_MALA = mcmc_MALA(algo2D, theta_init=w_MAP, T=1000)
    # w_MALA = mcmc_MALA(algo2D, theta_init=torch.tensor([0.0,0.0], requires_grad=True), T=1000)
    w_SGLD = mcmc_SGLD(algo2D, theta_init=w_MAP, T=1000)

    # #Plot all mcmc
    _, axes = plt.subplots(1, 3, figsize=(18,6))

    plot_mcmc(w_ULA, algo2D, w_MAP, axes[0], 'ULA')
    plot_mcmc(w_MALA, algo2D, w_MAP, axes[1], 'MALA')
    plot_mcmc(w_SGLD, algo2D, w_MAP, axes[2], 'SGLD')
    plt.show()



def plot_simple_student(w_teacher, w_gen, algo2D, w_MAP, w_MLE):
    algo2D.sim = False
    plotter(w_gen, algo2D, w_MAP, w_MLE)
    _, axes = plt.subplots(1, 4, figsize=(18,6))

    plot_mcmc(w_teacher, algo2D, w_MAP, axes[0], 'teacher weights')
    plot_mcmc(w_gen, algo2D, w_MAP, axes[1], 'student weights')
    print(f'MLE: {w_MLE}\t MAP:{w_MAP}')
    x = torch.arange(1, w_gen.shape[0]+1)
    axes[2].plot(x,w_gen[:,0])
    axes[2].set(xlabel='T', ylabel='intercept', title='phi_0 weights')
    axes[3].plot(x,w_gen[:,1])
    axes[3].set(xlabel='T', ylabel='slope', title='phi_1 weights')
    plt.show()
