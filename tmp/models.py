import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import math

class BNN:
    def forward():
        pass

    def log_joint(x):
        pass

    def gradient_log_joint():
        pass



class gped2DNormal(BNN):
    def __init__(self,x,y,batch_sz, alpha, beta, prior_mean,D, sim=True):
        self.alpha = alpha
        self.beta = beta
        self.prior = MultivariateNormal(prior_mean,covariance_matrix=1/self.alpha * torch.eye(D))
        self.x = x
        self.y = y
        self.D = D
        self.sim = sim
        # self.batch_sz = batch_sz
        self.N = len(x)
        self.M = batch_sz
        # self._standardize() x
        
    def _standardize(self):
        x_mean = torch.mean(self.x)
        x_std = torch.std(self.x)
        self.x = (self.x - x_mean) / x_std

        y_mean = torch.mean(self.y)
        y_std = torch.std(self.y)
        self.y = (self.y - y_mean) / y_std

    def log_prior(self,theta):
        return self.prior.log_prob(theta)

    def log_likelihood(self,theta):
        if self.x.dim() == 1:
            self.x = self.x[:,None] 

        phi = torch.cat([torch.ones_like(self.x), self.x],dim=1)
        sigma = 1/self.beta * torch.eye(len(self.x))
        # sigma = 1/self.beta * torch.eye(self.D)

        #if matrix of weights and not estiamte single value
        if not self.sim:
            mu2 = torch.mm(theta, phi.T)
            likelihood = MultivariateNormal(mu2, covariance_matrix=sigma)
            log_prob = likelihood.log_prob(self.y.squeeze())
            return (len(self.x) / self.batch_sz)  *log_prob
        
        #predict mu
        mu = torch.mv(phi, theta)
        likelihood = MultivariateNormal(mu,covariance_matrix=sigma)
        if self.y.shape[0] == 1:
            return likelihood.log_prob(self.y)
        return likelihood.log_prob(self.y.squeeze())

    def log_joint(self, theta):
        return (self.log_prior(theta) + self.log_likelihood(theta))
    
    def log_joint_gradient(self, theta): 
        # Ensure theta requires gradients
        if not theta.requires_grad:
            theta = theta.requires_grad_(True)
    
        # Zero the gradient before computing the new gradient
        if theta.grad is not None:
            theta.grad.zero_()

        self.log_joint(theta).backward()
        return theta.grad.detach()
        




def plot_weights(ax, algo, thetas, color=None, visibility=1, label=None, title=None):
    A, B = torch.meshgrid(thetas[:,0], thetas[:,1])
    algo.sim = False
    AB = torch.column_stack((A.ravel(), B.ravel()))
    posterior = algo.log_likelihood(AB)
    posterior = posterior.reshape(thetas.shape[0],thetas.shape[0])


    contour = ax.contour(thetas[:,0],thetas[:,1], torch.exp(posterior).detach().numpy(), color='g', title='ok bro',alpha=visibility)
    ax.plot([-1000], [-1000], color=color, label=label)
    ax.set(xlabel='slope', ylabel='intercept',   xlim=(-4, 4), ylim=(-4, 4), title=title)


def plot_distribution(ax, density_fun, color=None, visibility=1, label=None, title=None, num_points = 1000):
    
    # create grid for parameters (a,b)
    a_array = torch.linspace(-4, 4, num_points)
    b_array = torch.linspace(-4, 4, num_points)
        
    A_array, B_array = torch.meshgrid(a_array, b_array)   
        
    AB = torch.column_stack((A_array.ravel(), B_array.ravel()))
    
    Z = density_fun(AB)
    Z = Z.reshape(a_array.shape[0], b_array.shape[0])
    
    # plot contour  
    ax.contour(a_array, b_array, torch.exp(Z).numpy(), colors=color, alpha=visibility)
    ax.plot([-1000], [-1000], color=color, label=label)
    ax.set(xlabel='slope', ylabel='intercept', xlim=(-4, 4), ylim=(-4, 4), title=title)


def analytical_gradient(theta, Phi, ytrain, beta):

    prior_cov = torch.eye(2)

    S0_inv = torch.inverse(prior_cov)
    Phi_T_Phi = torch.matmul(Phi.t(), Phi)
    S = torch.inverse(S0_inv + beta * Phi_T_Phi)

    Phi_T_y = torch.matmul(Phi.t(), ytrain.squeeze())
    m = beta * torch.matmul(S, Phi_T_y)

    # gradient
    grad = -torch.matmul(torch.inverse(S), (theta - m))
    return grad, m


#minder mere om ULA
def mcmc_ULA(algo, theta_init,lr=1e-2, T=100):
    theta = theta_init.detach().clone().requires_grad_(True)
    zt = math.sqrt(2*lr)
    samples_theta = [None]*(T)

    for t in range(T):
        theta_grad = algo.log_joint_gradient(theta)       
        with torch.no_grad():
            #update teacher
            noise = torch.randn_like(theta_grad)*zt
            theta_grad_update = (lr/2) * theta_grad + noise
            # theta_grad_update = (lr/2)*theta_grad
            theta.add_(theta_grad_update)                        
            samples_theta[t] = theta.detach().clone()

            theta.grad.zero_()
    return torch.stack(samples_theta)



def design_matrix(x):
    return torch.column_stack((torch.ones(len(x)), x))



# Debugging: Print values at each iteration
# if T == 0 or T-t < 2:
#     print(f"Iteration {t}:")
#     print("Theta:", theta)
#     print("Theta Gradient:", theta_grad)
#     print("Noise:", noise)
#     print("Theta Grad Update:", theta_grad_update)
#     print("Updated Theta:", theta)
#     print("----------------------------")



# print("Log_prob shape:", log_prob.shape)
# print("Log_prob min:", torch.min(log_prob).item())
# print("Log_prob max:", torch.max(log_prob).item())

# Debugging: Print shapes and values
# print("Phi shape:", phi.shape)
# print("Phi values:", phi)
# print("Sigma shape:", sigma.shape)
# print("Sigma values:", sigma)
# print("Y values:", self.y)