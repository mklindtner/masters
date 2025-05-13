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
    def __init__(self,x,y,batch_sz, alpha, beta, prior_mean,D, sim=True, should_std = True):
        self.alpha = alpha
        self.beta = beta
        self.prior = MultivariateNormal(prior_mean,covariance_matrix=1/self.alpha * torch.eye(D))
        self.x = x
        self.y = y
        self.D = D
        self.sim = sim
        self.N = len(x)
        self.M = batch_sz
        # if should_std:
        #     self._standardize()

        self.log_npdf = lambda x, m, v: -0.5*torch.log(2*torch.pi*v) -0.5*(x-m)**2/v
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

        variance = torch.tensor([1/self.beta])
        likelihood = self.log_npdf(batch_y, theta_pred, variance)
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

        if theta.grad is None:
            raise RuntimeError(
                "theta_val.grad is None after backward(). "
                "This means self.log_joint(theta_val) does not depend on theta_val "
                "in a way that autograd can track, or theta_val was not a leaf node "
                "with requires_grad=True for the log_joint computation graph, "
                "or the graph was broken."
            )            
        return theta.grad.detach()

    #e.g. for MALA bcs it only needs the gradient of the target distribution and not change the gradient
    def get_log_joint_gradient(self, theta):
        if not theta.requires_grad:
            theta = theta.requires_grad_(True)

        if theta.grad is not None:
            theta.grad.zero_()
        else:
            self.log_joint(theta).backward()
            
        return torch.autograd.grad(self.log_joint(theta), theta,create_graph=False)[0]


        


class gped2DNormal_student(BNN):

    def __init__(self, x,y, batch_sz, prior_mean, alpha, beta, D=2, should_std = True):
        self.x = x
        self.y = y
        self.batch_sz = batch_sz
        self.alpha = alpha
        self.log_npdf = lambda x, m, v: -0.5*np.log(2*np.pi*v) -0.5*(x-m)**2/v
        self.prior = MultivariateNormal(prior_mean,covariance_matrix=1/self.alpha * torch.eye(D))

        self.predict = lambda x, a, b: a + b*x
        # if should_std:
        #     self._standardize()

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

    def gradient_log_joint():
        pass

    def Us(self, gyis, theta_t, mis):
        return self.g(gyis, theta_t, mis)

    def U_simple_2D(self, theta_t):
        return 1/len(theta_t) * torch.sum(theta_t, axis=0)

    def U_simple_predictive(self, theta_t):
        ll =  self.log_likelihood(theta_t)
        return 1/len(theta_t) * ll



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


def analytical_gradient(theta, Phi, ytrain, beta, alpha):
    prior_cov = 1/alpha*torch.eye(2)

    S0_inv = torch.inverse(prior_cov)
    Phi_T_Phi = torch.matmul(Phi.t(), Phi)
    S = torch.inverse(S0_inv + beta * Phi_T_Phi)

    Phi_T_y = torch.matmul(Phi.t(), ytrain.squeeze())
    m = beta * torch.matmul(S, Phi_T_y)

    # gradient
    grad = -torch.matmul(torch.inverse(S), (theta - m))
    return grad, m


def SGLD_step(theta_init, algo2D, t):
    theta = theta_init.detach().clone().requires_grad_(True)

    eps = 1/(t**0.75+1)
    if eps > 1e-2:
        eps = 1e-2

    theta_grad = algo2D.log_joint_gradient(theta) 
    with torch.no_grad():
        eta_t = torch.normal(mean=0.0, std=math.sqrt(eps), size=theta.shape)
        delta_theta = eps / 2 * theta_grad + eta_t
        theta += delta_theta

        if theta.grad is not None:
            theta.grad.zero_()

    return theta


def MALA_step(theta, algo2D):
    # theta = theta0.detach().clone().requires_grad_(True)
    D = 2
    h = 1e-2
    cov = h*torch.eye(D)


    grad = algo2D.get_log_joint_gradient(theta) 
    mu =  theta + h/2 * grad
    proposal_dist = MultivariateNormal(mu,cov)

    # print(f"Step {t}: Gradient = {grad}, Proposal Mean (mu) = {mu}")

    theta = metropolis_step(theta, proposal_dist=proposal_dist, pi_dist=algo2D.log_joint)
    return theta


def posterior_expectation_distillation(algo_teacher, algo_student, theta_init, phi_init, f, criterion, reg, opt, eps=1e-2, T=100, H=10, burn_in=100):
    
    #Initialize for teacher
    theta = theta_init.detach().clone().requires_grad_(True)
    samples_theta_teacher = [None]*T

    #Initialize for student
    student_sampling = math.ceil(int(((T-burn_in)/H))) - 1
    phi = phi_init.detach().clone().requires_grad_(True)
    samples_phi_student = [None]*student_sampling
    s = 0
    alphas = [1e-2]*student_sampling


    for t in range(T):
        # theta = MALA_step(theta, algo2D=algo_teacher)
        theta = SGLD_step(theta, algo2D=algo_teacher, t=t)

        samples_theta_teacher[t] = theta.detach().clone().requires_grad_(True)

        if t > burn_in and t % H == 0:  
            post_burn = t-burn_in+1                             
            teacher_samples = torch.stack(samples_theta_teacher[:post_burn])

            gyis = algo_student.U_simple_2D(teacher_samples)
            # gyis = algo_student.U_simple_predictive(teacher_samples)

            opt.zero_grad()

            pred = f(phi, algo_student.x)

            loss = criterion(gyis, pred)
            loss.backward()
            
            with torch.no_grad():
                phi -= alphas[s] * phi.grad

            samples_phi_student[s] = phi.detach().clone()
            phi.grad.zero_()
            opt.step()

            # print(s)
            s += 1

            if s % 100 == 0:
                print(f"Student Epoch {s}, Loss: {loss.item()}, Theta: {theta}, Phi: {phi}")

    samples_theta_teacher = [theta.detach().clone() for theta in samples_theta_teacher]
    samples_phi_student = [phi.detach().clone() for phi in samples_phi_student]
    return torch.stack(samples_theta_teacher), torch.stack(samples_phi_student)



def mcmc_SGLD(algo, theta_init, h_sq=1e-1, T=1000):
    D = theta_init.shape[0]
    theta = theta_init.detach().clone().requires_grad_(True)
    samples_theta = torch.empty((T,D), dtype=theta.dtype, device=theta.device)
    a = 1e-6; b=1; gamma = 0.75
    h_t_sq = h_sq
    for t in range(T):
        grad = algo.log_joint_gradient(theta)
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            print(f"Warning: NaN/inf gradient at SGLD iteration {t}. Grad: {grad}")
        
        mean_proposal = theta + h_t_sq/2 *grad
        if torch.isnan(mean_proposal).any() or torch.isinf(mean_proposal).any():
            print(f"Warning: NaN/inf mean_proposal at SGLD iteration {t}. Mean: {mean_proposal}")
        
        theta = MultivariateNormal(mean_proposal, h_t_sq*torch.eye(D)).rsample().clone().detach().requires_grad_(True)
        # if t < 200:
        samples_theta[t] = theta.detach().clone()
        h_t_sq = max(a/(b+t)**gamma,1e-12)

        
    return samples_theta


#works
# #SGLD
# def mcmc_SGLD(algo, theta_init, eps=1e-2, T=1000):
#     theta = theta_init.detach().clone().requires_grad_(True)

#     burn_in = 200
#     samples_theta = [None]*(T-burn_in)
#     id = 0
#     for t in range(T):
#         theta_grad = algo.log_joint_gradient(theta)
#         #suggestions for eps
#         eps = 1/(t**0.75+1)
#         if eps > 1e-2:
#             eps = 1e-2
#         if t < burn_in:
#             continue

#         with torch.no_grad():
#             eta_t = torch.normal(mean=0.0, std=math.sqrt(eps), size=theta.shape, dtype=theta.dtype, device=theta.device)
#             theta += (eps / 2) * theta_grad + eta_t
            
#             samples_theta[id] = theta.detach().clone()
#             id += 1
#             if theta.grad is not None:
#                 theta.grad.zero_()

#             # print(t)
#             # eps = 12/algo.N * (t+1)**(-0.55)
#             # eps = 12/algo.N * (t+1)**(-0.90)


#             # eps = eps**1/2 #normal uses std and not variance

#     return torch.stack(samples_theta)


#ULA
def mcmc_ULA(algo, theta_init, h_sq=1e-2, T=100):
    D = theta_init.shape[0]
    theta = theta_init.detach().clone().requires_grad_(True)
    samples_theta = torch.empty((T,D), dtype=theta.dtype, device=theta.device)

    for t in range(T):
        theta_grad = algo.log_joint_gradient(theta)
        theta = MultivariateNormal(theta + h_sq/2 * theta_grad, h_sq*torch.eye(D)).rsample().clone().detach().requires_grad_(True)
        samples_theta[t] = theta.detach().clone()
            
    return samples_theta


#Mala
def mcmc_MALA(algo, theta_init, T=100, h_sq=1e-1):
    D = theta_init.shape[0]
    
    theta = theta_init.detach().clone().requires_grad_(True)
    samples_theta = torch.empty((T,D), dtype=theta.dtype, device=theta.device)

    cov = h_sq*torch.eye(theta.shape[0])
    
    for t in range(T):
        grad = h_sq/2 * algo.log_joint_gradient(theta)


        # proposal (numerator)
        thetaprime = MultivariateNormal(theta + grad, cov).rsample().clone().detach().requires_grad_(True)
        

        gradprime = h_sq/2 * algo.log_joint_gradient(thetaprime)
        g_theta = MultivariateNormal(thetaprime + gradprime, cov)
        log_proposal = algo.log_joint(thetaprime) + g_theta.log_prob(theta)

        #current (denominator)
        g_thetaprime = MultivariateNormal(theta + grad, cov)
        log_current = algo.log_joint(theta) + g_thetaprime.log_prob(thetaprime)

        #Metropolis choice
        A = torch.min(torch.tensor(0), log_proposal - log_current)
        u = torch.log(torch.rand(1))

        if u <= A:
            theta = thetaprime
        samples_theta[t] = theta.detach().clone()

    # return torch.stack(samples_theta)
    return samples_theta



#assumes they are in log space
def metropolis_step(x,proposal_dist, pi_dist):
    xprime = proposal_dist.sample()
    proposal = pi_dist(xprime) - pi_dist(x)

    #domian check
    log_proposal = torch.min(torch.tensor(0.0),proposal)


    u = torch.rand(1)
    if u < torch.exp(log_proposal):
        return xprime
    return x


#pMALA



#Etc.
def algo_student_reg(w, x):
    inputs = torch.column_stack((torch.ones(len(x)), x))
    return inputs @ w[:,None]



def design_matrix(x):
    return torch.column_stack((torch.ones(len(x)), x))