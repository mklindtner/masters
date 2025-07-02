from torch import optim, nn
import torch
import math
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.nn.utils import parameters_to_vector
import itertools
import torch.distributions as D



class FCC_Regression_Parkinsons_Mean(nn.Module):
    def __init__(self, input_size, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 400)
        self.fc2 = nn.Linear(400, 400)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc3_mean = nn.Linear(400, 1)

#For stort netværk?
class FFC_Regression_Parkinsons(nn.Module):#

    def __init__(self, input_size, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 400)
        self.fc2 = nn.Linear(400, 400)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc3_mean = nn.Linear(400, 1)
        self.fc3_log_var = nn.Linear(400,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))      
        x = self.dropout(x)  
        x = F.relu(self.fc2(x))        
        x = self.dropout(x)
        
        mean = self.fc3_mean(x)

        #brug softplus til log var
        log_var = self.fc3_log_var(x)
        # mean = torch.clamp(mean, min=-30, max=30)
        # log_var = 5.0 * torch.tanh(self.fc3_log_var(x))
        return mean, log_var


def validate_network(model, validation_loader, criterion, device, verbose=True):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    val_loop = tqdm(validation_loader, desc="Validating", leave=False, disable=not verbose)
    
    with torch.no_grad():
        for inputs, labels in val_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            
            mean, log_var = model(inputs)
            loss = criterion(input=mean, target=labels, var=torch.exp(log_var))
            
            total_loss += loss.item()
            total_samples += len(labels)
            
    return total_loss / total_samples


def U_s(teacher_model, inputs):
    teacher_model.eval()
    with torch.no_grad():
        targets = teacher_model(inputs)
    return targets




class BayesianRegressionParkin():
    
    def __init__(self, f, n,m, likelihood_criterion, tau=10):
        self.tau = tau
        self.lik_criterion = likelihood_criterion
        self.f = f
        self.N = n
        self.M = m
        

    #We are not using this because if we then do .backward() we'll find the autodiff grad which is much more inefficient than the analytical expression
    def log_prior(self):
        W = torch.cat([w.view(-1) for w in self.f.parameters()])
        W_sq = torch.dot(W, W)
        return -1/2 * self.tau * W_sq


    def log_likelihood(self, x,y):
        mean, log_var = self.f(x)
        var = torch.exp(log_var)
        return -self.lik_criterion(input=mean, target=y, var=var)
        # return -self.lik_criterion(outputs, y)

    def log_joint(self, x,y):
        return self.log_prior() + self.log_likelihood(x,y)
    
    def log_joint_gradient(self, x,y):        

        #prior: analytical grad
        w_grad_prior_list = [-self.tau* w.data.view(-1) for w in self.f.parameters()]
        w_grad_prior = torch.cat([w.view(-1) for w in w_grad_prior_list])

        #likelihood: autodiff grad
        self.f.zero_grad()
        w_grad_likelihood_loss = self.log_likelihood(x,y)
        w_grad_likelihood_loss.backward()
        w_grad_likelihood = torch.cat([w.grad.view(-1) for w in self.f.parameters()])

        return w_grad_prior + self.N/self.M * w_grad_likelihood

    
    def sgld_step(self, x, y, lr):
        grad = self.log_joint_gradient(x,y)

        with torch.no_grad():
            noise = torch.randn_like(grad) * math.sqrt(lr)
            w_deltas = lr/2 * grad + noise

            offset = 0
            for w in self.f.parameters():
                wslice = w.view(-1).shape[0]
                w_delta = w_deltas[offset : offset+wslice].view(w.shape)  
                w.add_(w_delta)
                offset += wslice




def bayesian_distillation_parkin(tr_items, msc_items, tr_hyp_par, T_total=1e6, verbose=True):
    tr_bayers, tr_network, tr_loader_train, tr_loader_valid = tr_items
    B, H, eval_criterion, device = msc_items
    lr_init, decay_gamma, lr_b = tr_hyp_par
    train_iterator = itertools.cycle(tr_loader_train)
    results = []


    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

    print(f"--- Starting Distillation Process for {T_total} steps ---")
    
    T = tqdm(range(T_total), desc="Total Steps", disable=not verbose, bar_format=bar_format)

    # tr_criterion_nll = nn.CrossEntropyLoss(reduction='mean')

    for t in T:
        lr = lr_init * (lr_b + t)**(-decay_gamma)
        T.set_postfix(LR=f"{lr:.2e}") 
        inputs, labels = next(train_iterator)
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(inputs.size(0), -1)

        #The tr_network should be inside here
        tr_bayers.sgld_step(inputs, labels, lr)

        if t >= B and (t % H == 0):
            with torch.no_grad():            
                mean,log_var = tr_network(inputs)
                # tr_nll_train = tr_criterion_nll(outputs, labels)
                tr_nll_train = eval_criterion(input=mean, target=labels, var=torch.exp(log_var))
                tr_nll_avg = 1/len(labels) * tr_nll_train.item()

                tr_network.eval()  
                teacher_nll = validate_network(tr_network, tr_loader_valid, eval_criterion, device, verbose=False)
                
            results.append({
                't': t + 1,
                'tr_nll_val': teacher_nll,
                'tr_nll_train': tr_nll_avg,
            })        

            tr_network.train()
    return results, tr_network.state_dict()