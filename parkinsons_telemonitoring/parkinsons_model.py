from torch import optim, nn
import torch
import math
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.nn.utils import parameters_to_vector
import itertools
import torch.distributions as D


#NB! I have chosen to use NLLGaus('sum') for the formula to best fit GPED algo1. But this defiend the constanti nfront of hte likelihood and weight_decay in the step
class SGLD(optim.Optimizer):
    """
    SGLD Optimizer for teacher
    """
    def __init__(self, params, lr=1e-3, weight_decay=0, N=1, M=1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, weight_decay=weight_decay, N=N, M=M)
        super(SGLD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            N = group['N']
            M = group['M']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                #l2 regularization = prior w. N(0,1/tau) as explained in theory section
                prior_grad = -weight_decay * p.data

                #p.grad =  ∇(-log p(y|x,θ)) = - ∇(log p(y|x,θ))
                #hence we need -p.grad to get the positive ll_grad
                ll_grad =  -N/M * p.grad
                

                gradient_step = 0.5 * lr * (prior_grad + ll_grad)

                #White noise
                noise = torch.randn_like(p.data) * math.sqrt(lr)

                w_update = gradient_step + noise
                #correct update
                p.data.add_(w_update)

                #"cheating"  for testing the gradient
                # p.data.add_(gradient_step)
                


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
        mean = torch.clamp(mean, min=-30, max=30)
        log_var = 5.0 * torch.tanh(self.fc3_log_var(x))
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

def train_teacher_network(tr_optim, tr_network, T_steps, tr_loader_train, tr_loader_valid, criterion, device, tr_eval, verbose=True):        
    train_iterator = itertools.cycle(tr_loader_train)

    T = tqdm(range(T_steps), desc="SGLD Sampling", disable=not verbose)
    results = []

    for t in T:
        tr_network.train()
        inputs, labels = next(train_iterator)
        inputs, labels = inputs.to(device), labels.to(device)        
        tr_optim.zero_grad()

        mean, log_var = tr_network(inputs)

        loss = criterion(input=mean, target=labels, var=torch.exp(log_var))

        loss.backward()
        tr_optim.step()

        # current_params = parameters_to_vector(tr_network.parameters()).clone().detach().cpu()
        # samples_w[t] = current_params
        tr_gauss_nll = loss.item() / len(labels)

        if (t + 1 ) % tr_eval == 0:
            validation_nll = validate_network(tr_network, tr_loader_valid, criterion, device, verbose=verbose)
            # tr_gaussnll_val.append(validation_nll)

            if verbose:
                T.set_postfix(Validation_NLL=f"{validation_nll:.4f}")
            results.append({
                't_val': t+1,
                'tr_gauss_nll_loss_train': tr_gauss_nll,
                'tr_gauss_nll_loss_val': validation_nll
            })
        else:
            results.append({
                't': t,
                'tr_gauss_nll_loss_train': tr_gauss_nll
            })

    print("--- Finished Teacher Training ---")
    return results, tr_network.state_dict()


def distillation_posterior_parkinsons(tr_items, st_items, msc_list, T_steps, verbose=True):
    tr_optim, tr_network, tr_loader_train, tr_loader_valid = tr_items
    st_network, st_optim, st_scheduler, U, tr_st_criterion = st_items
    B, H, criterion, device = msc_list    
    
    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    T = tqdm(range(T_steps), desc="SGLD Sampling", disable=not verbose, bar_format=bar_format)
    train_iterator = itertools.cycle(tr_loader_train)
    results = []
    s = 0

    for t in T:
        tr_network.train()
        inputs, labels = next(train_iterator)
        inputs, labels = inputs.to(device), labels.to(device)        
        tr_optim.zero_grad()

        tr_mean, tr_log_var = tr_network(inputs)
        loss = criterion(input=tr_mean, target=labels, var=torch.exp(tr_log_var))
        loss.backward()
        tr_optim.step()

        # tr_gauss_nll = loss.item() / len(labels)

        if t >= B and (t % H == 0):   
            T.set_postfix(Status="Distilling")
            st_network.train() 
            tr_network.eval()  
            distill_inputs, _ = next(train_iterator)
            distill_inputs = distill_inputs.to(device)


            with torch.no_grad():
                tr_mean, tr_log_var = U(tr_network, distill_inputs)
                tr_sd = torch.exp(0.5 * tr_log_var)

            st_mean, st_log_var = st_network(distill_inputs)
            st_sd = torch.exp(0.5 * st_log_var)

            tr_normal = D.Normal(loc=tr_mean, scale=tr_sd)
            st_normal = D.Normal(loc=st_mean, scale=st_sd)
            st_loss_pr_sample = D.kl.kl_divergence(tr_normal, st_normal)
            st_loss = torch.sum(st_loss_pr_sample)
            
            
            st_optim.zero_grad()
            st_loss.backward()
            st_optim.step()
            s+= 1

            if s % 200 == 0:
                st_scheduler.step()
                print(f"\nStep {t+1}: Student LR decayed.")

            
            # student_nll = validate_network(st_network, tr_loader_valid, criterion, device, verbose=False)
            teacher_nll = validate_network(tr_network, tr_loader_valid, criterion, device, verbose=False)

            results.append({
                't': t + 1,
                'tr_nll': teacher_nll,
                'tr_st_kl': st_loss.item()
            })

       

    print("--- Finished Posterior distillation ---")
    return results, tr_network.state_dict(), st_network.state_dict()