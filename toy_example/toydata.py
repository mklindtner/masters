import torch
from models import gped2DNormal, gped2DNormal_student, design_matrix
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import torch.nn as nn
import torch.nn.functional as F




alpha = 5
beta = 3/4
xtrain = torch.tensor([1.764, 0.4, 0.979, 2.241, 1.868, -0.977,  0.95, -0.151, -0.103, 0.411, 0.144, 1.454, 0.761, 0.122,
              0.444, 0.334, 1.494, -0.205,  0.313, -0.854])[:,None]
ytrain = torch.tensor([-0.464, 2.024, 3.191, 2.812, 6.512, -3.022, 1.99, 0.009, 2.513, 3.194, 0.935, 3.216, 0.386, -2.118,
               0.674, 1.222, 4.481, 1.893, 0.422,  -1.209])[:,None]

N = 20
sz = 2
T = 100
prior_mean = torch.tensor([0,0])
theta_init = torch.tensor([0.0,0.0], requires_grad=True)


algo2D = gped2DNormal(xtrain,ytrain, batch_sz=len(xtrain), alpha=alpha, beta=beta, prior_mean=prior_mean, D=2)
algo2D_student_simple = gped2DNormal_student(xtrain, ytrain, alpha=alpha, beta=beta, batch_sz=len(xtrain), prior_mean=prior_mean, D=2)



#MLE/MAP & distribution
Phi_train = torch.column_stack((torch.ones(len(algo2D.x)), algo2D.x))
w_MLE = np.linalg.solve(Phi_train.T@Phi_train, Phi_train.T@algo2D.y).ravel()
w_MAP = (beta*torch.linalg.solve(alpha*torch.eye(2) + beta*(Phi_train.T@Phi_train), Phi_train.T)@algo2D.y).ravel()

#Analytical mean and covariance for S,M for 2D example
S = torch.inverse(alpha*torch.eye(2) + beta * Phi_train.T @ Phi_train)
M = beta*S@Phi_train.T @ algo2D.y
target = MultivariateNormal(loc=M.T.squeeze(), covariance_matrix=S)

#PDD Mean and covariance
PDD_M = (Phi_train @ M).squeeze()
PDD_S = (Phi_train @ S @ Phi_train.T) + torch.eye(algo2D.x.shape[0]) * 1/beta
PDD_S = PDD_S + torch.eye(PDD_S.shape[0]) * 1e-6

#For quicker calculations we're going to ignore the covariance between samples
PDD_sigma_sq = torch.clamp(torch.diag(PDD_S), min=1e-8)
PDD_sigma = torch.sqrt(PDD_sigma_sq)
target_PDD = torch.distributions.Normal(PDD_M, PDD_sigma)   

# Phi_current_x = design_matrix_torch(algo2D.x)
# mu_analytical_ppd = (Phi_current_x @ M_analytical).squeeze().to(torch.float32) # Ensure float32 for consistency
# var_epistemic_analytical = torch.diag(Phi_current_x @ S_analytical @ Phi_current_x.T).to(torch.float32)
# # Ensure beta_teacher is float32 for division
# var_analytical_ppd = (1.0/beta_teacher.to(torch.float32)) + var_epistemic_analytical
# var_analytical_ppd = torch.clamp(var_analytical_ppd, min=1e-6) # Ensure variance is positive

#Single g regressors
g_bayesian_linear_reg = lambda x, w: w[:,0] + w[:,1]*x
g_meansq = lambda x,w: (w[:,0] + w[:,1]*x)**2

def g_pred_likelihood(x_points, w):
    #Mean
    phi_x = torch.cat([torch.ones_like(x_points), x_points], dim=1)
    likelihood_weights = phi_x @ w.T    # Shape [N_data, num_teacher_samples]    
    mu = torch.mean(likelihood_weights, dim=1)      # Shape [N_data]
    var = torch.mean(likelihood_weights**2, dim=1) - mu**2
    
    #Var
    aleatoric_var = 1.0 / beta  # Beta^(-1)   
    sigma_sq = aleatoric_var + var  # Shape [N_data]
    sigma_sq = torch.clamp(sigma_sq, min=1e-6)     #approximations implies the variance could be negative so just gonna clamp.

    #Sample
    epsilon = torch.randn_like(mu) 
    y_true_samples = mu + torch.sqrt(sigma_sq) * epsilon

    return y_true_samples


#New regressors
def g1_blinear(x, w): return torch.cat([torch.ones_like(x), x], dim=1) @ w.T
def g2_bsq(x, w): return g1_blinear(x, w)**2
def g3_ppd(x, w): return g_pred_likelihood(x,w)

class StudentToyDataReqLin(nn.Module):
    def __init__(self):
        super(StudentToyDataReqLin, self).__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x
    

class StudentToyDataRegSq(nn.Module):
    def __init__(self):
        super(StudentToyDataRegSq, self).__init__()
        self.fc1 = nn.Linear(3, 1)

    def forward(self, x):
        features = torch.cat([x**2, x, torch.ones_like(x)], dim=1)
        x = self.fc1(features)
        return x
    

class StudentToyDataPredictivePosterior(nn.Module):
    def __init__(self):
        super(StudentToyDataPredictivePosterior, self).__init__()
        self.mean = nn.Linear(2,1)
        self.log_variance = nn.Linear(3,1)

    def forward(self, x):
        mean_features = torch.cat([x, torch.ones_like(x)], dim=1)
        mean = self.mean(mean_features)
        log_variance_features = torch.cat([x**2, x, torch.ones_like(x)], dim=1)
        log_variance = self.log_variance(log_variance_features)
        return (mean, log_variance)
    

MSEloss = nn.MSELoss()
NLLloss = nn.NLLLoss()
H = 20;alpha_s = 1e-2
#  burn_in = 1000; 
distil_params = [H,alpha_s]
f_student = StudentToyDataReqLin()
f_student_sq = StudentToyDataRegSq()
f_student_pred_post = StudentToyDataPredictivePosterior()
f_SCALAR = 'scalar'; f_DIST = 'dist'

SGLD_params = (2.1*1e-1,1.65, 0.556, 1e-2)
phi_init = torch.tensor([0.0,0.0], requires_grad=True)

st_list = [(f_student, g1_blinear, nn.MSELoss(), 'scalar'), (f_student_sq,g2_bsq, nn.MSELoss(), 'scalar'), (f_student_pred_post, g3_ppd, nn.GaussianNLLLoss(reduction='mean', eps=1e-6), 'dist')]



colors_gradient = {
    'teacher': 'k',             # Black for ground truth
    'student_final': 'red',   # Green for the final converged model
    'step_5': '#4682B4',        # Lightest blue
    'step_50': 'yellow',       # Medium-light blue
    'step_1000': 'blue',      # Medium-dark blue
    'step_2500': '#4169E1',       # Darkest blue (approaching final)
    'step_final': '#3F51B5',
    'st_sq': 'cornflowerblue'
}

