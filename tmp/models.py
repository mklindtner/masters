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
    
    #def U():


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
        else:
            self.log_joint(theta).backward()
            
        # return self.log_joint(theta).grad.detach()
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
        if should_std:
            self._standardize()

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


def SGLD_step(theta, theta_grad, eps, N, t):
    eta_t = torch.normal(mean=0.0, std=eps, size=theta.shape, dtype=theta.dtype, device=theta.device)
    delta_theta = eps/2 * theta_grad + eta_t
    theta.add_(delta_theta)
    theta.grad.zero_()

    eps = 12/N * (t+1)**(-0.55)
    eps = eps**0.5 #normal uses std and not variance
    return theta, eps


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
        # theta_grad = algo_teacher.log_joint_gradient(theta)
        theta = MALA_step(theta, algo2D=algo_teacher)

        # theta, eps = SGLD_step(theta, theta_grad, eps, algo_teacher.N, t)

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





#SGLD
def mcmc_SGLD(algo, theta_init, eps=1e-2, T=100):
    theta = theta_init.detach().clone().requires_grad_(True)

    samples_theta = [None]*T

    for t in range(T):
        theta_grad = algo.log_joint_gradient(theta)

        with torch.no_grad():
            eta_t = torch.normal(mean=0.0, std=eps, size=theta.shape, dtype=theta.dtype, device=theta.device)
            theta = theta + eps**2 / 2 * theta_grad + eta_t
            
            samples_theta[t] = theta.detach().clone()
            if theta.grad is not None:
                theta.grad.zero_()

            # print(t)
            # eps = 12/algo.N * (t+1)**(-0.55)
            # eps = 12/algo.N * (t+1)**(-0.90)


            # eps = eps**1/2 #normal uses std and not variance

    return torch.stack(samples_theta)


#ULA
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

            theta.add_(theta_grad_update)
            samples_theta[t] = theta.detach().clone()

            theta.grad.zero_()
    return torch.stack(samples_theta)

#Mala
def mcmc_MALA(algo, theta_init, T=100):
    theta = theta_init.detach().clone().requires_grad_(True)
    D = 2
    samples_theta = [None]*T

    #Roberts & Rosenthal (1998) optimal scaling parameter but did not work
        # h = 2.38**2 / D
    h = 1e-2
    cov = h*torch.eye(D)

    for t in range(T):
        grad = algo.get_log_joint_gradient(theta)
        mu =  theta + h/2 * grad
        proposal_dist = MultivariateNormal(mu,cov)

        # print(f"Step {t}: Gradient = {grad}, Proposal Mean (mu) = {mu}")

        theta = metropolis_step(theta, proposal_dist=proposal_dist, pi_dist=algo.log_joint)
        samples_theta[t] = theta.detach().clone()

    return torch.stack(samples_theta)


#Assume proposal distriubtion is symmetric
#assumes they are in log space
def metropolis_step(x,proposal_dist, pi_dist):
    xprime = proposal_dist.sample()
    # print(f"{xprime[0].item()}, {xprime[1].item()}", pi_dist(x).item())
    proposal = pi_dist(xprime) - pi_dist(x)

    #domian check
    log_proposal = torch.min(torch.tensor(0.0),proposal)


    u = torch.rand(1)
    #print(u.item(), torch.exp(log_proposal).item(), sep=",")
    if u < torch.exp(log_proposal):
        return xprime
    return x


#pMALA






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


# anal_w, m = analytical_gradient(theta_init, Phi, algo2D.x, beta)
# print(f"Analytical Gradient\n {anal_w}")
# print(f"log_joint_gradient-value\n {w}")
# print(f"difference\n {anal_w - w}")
# print("breakpoint")