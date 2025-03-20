import torch
from torch.distributions.multivariate_normal import MultivariateNormal


#SGLD
def mcmc_SGLD(algo, theta_init, eps=1e-2, T=100):
    theta = theta_init.detach().clone().requires_grad_(True)

    samples_theta = [None]*T

    for t in range(T):        
        theta_grad = algo.log_joint_gradient(theta)
        with torch.no_grad():            
            eta_t = torch.normal(mean=0.0, std=eps, size=theta.shape, dtype=theta.dtype, device=theta.device)
            delta_theta = eps/2 * theta_grad + eta_t
            theta.add_(delta_theta)
            samples_theta[t] = theta.detach().clone()
            theta.grad.zero_()

            print(t)
            eps = 4/algo.N * (t+1)**(-0.55)
            eps = eps**0.5 #normal uses std and not variance

    return torch.stack(samples_theta)


#ULA
def mcmc_ULA(algo, theta_init,lr=1e-2, T=100):
    theta = theta_init.detach().clone().requires_grad_(True)
    # zt = math.sqrt(2*lr)
    zt = (2*lr)**0.5
    samples_theta = [None]*(T)

    for t in range(T):
        theta_grad = algo.log_joint_gradient(theta)       
        with torch.no_grad():
            #update teacher
            noise = torch.randn_like(theta_grad)*zt
            theta_grad_update = (lr/2) * theta_grad + noise
            theta.add_(theta_grad_update)                        
            samples_theta[t] = theta.detach().clone()

            theta.grad.zero_()
    return torch.stack(samples_theta)


#Mala
def mcmc_MALA(algo, theta_init, T=100):
    theta = theta_init.detach().clone().requires_grad_(True)
    D = 2
    samples_theta = [None]*T
     
    h = 1e-2
    cov = h*torch.eye(D)

    for t in range(T):
        grad = algo.get_log_joint_gradient(theta) 
        mu =  theta + h/2 * grad    
        proposal_dist = MultivariateNormal(mu,cov)

        theta = metropolis_step(theta, proposal_dist=proposal_dist, pi_dist=algo.log_joint)
        samples_theta[t] = theta.detach().clone()

    return torch.stack(samples_theta)


#Assume proposal distriubtion is symmetric
#assumes they are in log space
def metropolis_step(x,proposal_dist, pi_dist):
    xprime = proposal_dist.sample()
    # print(f"{xprime[0].item()}, {xprime[1].item()}", pi_dist(x).item())
    proposal = pi_dist(xprime) - pi_dist(x)

    #log domian check
    log_proposal = torch.min(torch.tensor(0.0),proposal)


    u = torch.rand(1)
    #print(u.item(), torch.exp(log_proposal).item(), sep=",")
    if u < torch.exp(log_proposal):
        return xprime
    return x


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
        self.N = len(x)
        self.M = batch_sz
        self._standardize()

        self.log_npdf = lambda x, m, v: -0.5*np.log(2*np.pi*v) -0.5*(x-m)**2/v 
        self.predict = lambda x, a, b: a + b*x
        
    def _standardize(self):
        x_mean = torch.mean(self.x)
        x_std = torch.std(self.x)
        self.x = (self.x - x_mean) / x_std

        y_mean = torch.mean(self.y)
        y_std = torch.std(self.y)
        self.y = (self.y - y_mean) / y_std

    def log_prior(self,theta):
        return self.prior.log_prob(theta)
        

    def log_likelihood(self, theta):    
        #for single sample            
        if theta.dim() == 1:
            theta = theta[None,:]        

        #get batch size
        idx = torch.randperm(len(self.x))[:self.M]
        batch_x = self.x[idx]
        batch_y = self.y[idx]

        if self.sim == False:
            batch_x = self.x
            batch_y = self.y
            self.M = len(self.x)
        
        theta_pred = self.predict(batch_x, theta[:,0], theta[:,1])   
            
        likelihood = self.log_npdf(batch_y, theta_pred, 1/self.beta) 
        ll = torch.sum(likelihood, 0)
        ll_batch = (self.N / self.M ) * ll 
        return ll_batch
        

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

    #e.g. for MALA bcs it only needs the gradient of the target distribution and not change the gradient
    def get_log_joint_gradient(self, theta):
        # Ensure theta requires gradients
        if not theta.requires_grad:
            theta = theta.requires_grad_(True)
    
        # Zero the gradient before computing the new gradient
        if theta.grad is not None:
            theta.grad.zero_()
        
        # return self.log_joint(theta).grad.detach()
        return torch.autograd.grad(self.log_joint(theta), theta,create_graph=False)[0]
