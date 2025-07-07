import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import math
from tqdm.auto import tqdm
from torch.nn.utils import parameters_to_vector
import itertools
import torch.nn.functional as F
import os
import collections
import random



class SGLD(optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0, N=1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, weight_decay=weight_decay, N=N)
        super(SGLD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        lr = group['lr']
        weight_decay = group['weight_decay']
        N = group['N']

        if not hasattr(self, 't_counter'):
            self.t_counter = 0
        self.t_counter += 1

        if self.t_counter % (100000 * 0.096) == 0:
            print(f"Step {self.t_counter}: Optimizer using LR = {lr}")
            
        
        for p in group['params']:
            if p.grad is None:
                continue

        
            #l2 regularization = tau*theta
            prior_grad = -weight_decay * p.data

            #p.grad gives average grad because nn.CrossEntropyLoss() = -1/m * grad
            #N * sum_of_gradients = (N/M) mean_gradient)     
            #in torch we optimize form in prob. We want max so negative. 
            ll_grad =  -N * p.grad
            
            gradient_step = 0.5 * lr * (prior_grad + ll_grad)

            noise = torch.randn_like(p.data) * math.sqrt(lr)

            w_update = gradient_step + noise
            
            #correct update
            p.data.add_(w_update)

            #update to debug gradient
            # p.data.add_(gradient_step)
        # noise = torch.randn_like(p.data) * math.sqrt(lr)
        # p.data.add_(noise)
                


class FFC_Regression(nn.Module):
    def __init__(self, input_size, output_size=1, dropout_rate=0.5):
        super(FFC_Regression, self).__init__()
        self.fc1 = nn.Linear(input_size, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, output_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))      
        x = self.dropout(x)  
        x = F.relu(self.fc2(x))        
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x



#Validate student network
def validate_network(model, validation_loader, criterion, device, verbose=True):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    
    # Create a tqdm progress bar for the validation loop if verbose is True
    val_loop = validation_loader

    with torch.no_grad():
        for inputs, labels in val_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            _, predvalidx = torch.max(outputs, 1)
            correct_predictions += (predvalidx == labels).sum().item()



    nll_val = total_loss / total_samples
    acc_val = correct_predictions / total_samples
    return nll_val,  acc_val




# Equation (9) from "posterior distillation"
def U_s(teacher_model, inputs):
    teacher_model.eval()
    with torch.no_grad():
        targets = teacher_model(inputs)
    return targets


def distil_MNIST(tr_items, st_items, msc_list, T_total=1e10, verbose=True):    
    tr_optim, tr_network, tr_loader_train, tr_loader_valid = tr_items
    # st_network, st_optim, st_scheduler, U, tr_st_criterion = st_items
    B, H, criterion, device = msc_list       
    s = 0

    train_iterator = itertools.cycle(tr_loader_train)
    results = []
    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    print(f"--- Starting Distillation Process for {T_total} steps ---")    
    T = tqdm(range(T_total), desc="Total Steps", disable=not verbose, bar_format=bar_format)


    for t in T:
       
        tr_network.train()        
        inputs, labels = next(train_iterator)
        
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(inputs.size(0), -1)

        tr_optim.zero_grad()
        outputs = tr_network(inputs)
        loss = criterion(outputs, labels)

        #log train
        if t >= B and (t % H == 0):
            tr_nll_train = loss.item()


        loss.backward()
        tr_optim.step()
        

        if t >= B and (t % H == 0):   
            with torch.no_grad():
                T.set_postfix(Status="Distilling")
                # st_network.train() 
                tr_network.eval()  
                teacher_nll = validate_network(tr_network, tr_loader_valid, criterion, device, verbose=False)
                results.append({
                    't': t + 1,
                    'tr_nll_val': teacher_nll,
                    'tr_nll_train': loss.item(),
                    # 'st_nll': student_nll,
                })        
            # distill_inputs, _ = next(train_iterator)
            # distill_inputs = distill_inputs.to(device).view(distill_inputs.size(0), -1)


            # with torch.no_grad():
            #     teacher_logits = U(tr_network, distill_inputs)

            
            # student_logits = st_network(distill_inputs)
            # soft_targets = F.log_softmax(teacher_logits, dim=-1)
            # soft_predictions = F.log_softmax(student_logits, dim=-1)
            # st_loss = tr_st_criterion(soft_predictions, soft_targets)


            # st_optim.zero_grad()
            # st_loss.backward()
            # st_optim.step()
            s+= 1

            # if s % 200 == 0:
            #     st_scheduler.step()
            #     print(f"\nStep {t+1}: Student LR decayed.")

            
            # student_nll = validate_network(st_network, tr_loader_valid, criterion, device, verbose=False)

            
    
    print("--- Finished Distillation Process ---")
    return results, tr_network.state_dict()
    #Make this back once the SGLD works
    # return results, st_network.state_dict(), tr_network.state_dict()





#uses epochs cba to rewrite
def train_teacher_network(tr_optim, tr_network, T_epochs, tr_loader_train, tr_loader_valid, criterion, device, verbose=True):
    num_params = len(parameters_to_vector(tr_network.parameters()))
    # print(f"Teacher network has {num_params} parameters.")
    
    samples_theta = torch.empty((T_epochs, num_params), device='cpu')
    tr_nll = []
    results = []

    # Create a tqdm progress bar for the epochs loop if verbose is True
    T = tqdm(range(T_epochs), desc="Total Progress", disable=not verbose)

    print(f"--- Starting Teacher Training for {T_epochs} epochs ---")
    for t in T:
        tr_network.train()
        
        batch_loop = tqdm(tr_loader_train, desc=f"Epoch {t+1}/{T_epochs}", leave=False, disable=not verbose)
        
        for inputs, labels in batch_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)

            tr_optim.zero_grad()
            outputs = tr_network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            tr_optim.step()

        # Validate and save teacher parameters        
        validation_nll = validate_network(tr_network, tr_loader_valid, criterion, device, verbose=verbose)
        
        # tr_nll.append(validation_nll)
        results.append({
            "nll": validation_nll,
            
        })
        
        current_params = parameters_to_vector(tr_network.parameters()).clone().detach().cpu()
        samples_theta[t] = current_params
        
        if verbose:
            T.set_postfix(Validation_NLL=f"{validation_nll:.4f}")

    print("--- Finished Teacher Training ---")
    # return samples_theta, tr_nll
    return results


def validate_network_bayesian(network, weight_samples, val_loader, device):
    """
    Performs a Bayesian Model Average to evaluate the model.
    This version has a corrected check for the all_targets tensor.
    """
    network.eval()
    all_predictions = []
    all_targets = None

    sample_loop = tqdm(weight_samples, desc="  Validating Teacher (BMA)", leave=False)


    for sample_state_dict in sample_loop:
        network.load_state_dict({k: v.to(device) for k, v in sample_state_dict.items()})
        
        model_preds = []
        
        if all_targets is None:
            batch_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                
                outputs = network(data)
                softmax_probs = F.softmax(outputs, dim=1)
                model_preds.append(softmax_probs)
                
                if all_targets is None:
                    batch_targets.append(target)

        all_predictions.append(torch.cat(model_preds))
        
        if all_targets is None:
            all_targets = torch.cat(batch_targets)


    stacked_preds = torch.stack(all_predictions, dim=0)
    avg_probs = torch.mean(stacked_preds, dim=0)
    
    final_nll = F.nll_loss(torch.log(avg_probs + 1e-9), all_targets, reduction='mean').item()
    _, predicted_labels = torch.max(avg_probs, 1)
    final_accuracy = (predicted_labels == all_targets).sum().item() / len(all_targets)

    return final_nll, final_accuracy

def get_bayesian_train_metrics(network, eval_samples, inputs, labels, device):   
    network.eval() # Ensure model is in eval mode for consistent predictions
    
    train_batch_preds = []
    for sample_state_dict in eval_samples:
        network.load_state_dict({k: v.to(device) for k, v in sample_state_dict.items()})
        batch_outputs = network(inputs)
        train_batch_preds.append(F.softmax(batch_outputs, dim=1))
    
    # Average the probabilities from the 100 models for the current batch
    model_preds = torch.stack(train_batch_preds, dim=0)
    avg_train_probs = torch.mean(model_preds, dim=0)
    
    # Calculate NLL and accuracy from the averaged probabilities
    nll = F.nll_loss(torch.log(avg_train_probs + 1e-9), labels).item()
    _, predicted_labels = torch.max(avg_train_probs, 1)
    accuracy = (predicted_labels == labels).sum().item() / labels.size(0)
    
    return nll, accuracy


#Made specifically for MNIST, tau is precision.
class BayesianRegression():
    def __init__(self, f, n,m, tau=10):
        self.tau = tau
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.w = f.state_dict()
        self.f = f
        self.N = n
        self.M = m
        

    def log_prior(self):
        W = torch.cat([w.view(-1) for w in self.f.parameters()])
        W_sq = torch.dot(W, W)
        return -1/2 * self.tau * W_sq


    def log_likelihood(self, x,y):
        outputs = self.f(x)
        return -self.criterion(outputs, y)

    def log_joint(self, x,y):
        return self.log_prior() + self.log_likelihood(x,y)
    
    def log_joint_gradient(self, x,y):        
        #prior: analytical grad
        w_grad_prior_list = [-self.tau/2 * w.data.view(-1) for w in self.f.parameters()]
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

    def log_likelihood_selfimpl(self, x, y):
        logits = self.f(x)
        log_probs = F.log_softmax(logits, dim=1)
        nll = F.nll_loss(log_probs, y, reduction='sum')
        return -nll

    def log_joint_gradient_selfimpl(self, x, y):
        #prior analytical
        w_grad_prior = torch.cat([-self.tau * w.data.view(-1) for w in self.f.parameters()])
        
        #likelihood
        self.f.zero_grad()
        log_l = self.log_likelihood_selfimpl(x, y)
        log_l.backward()
        w_grad_likelihood = torch.cat([w.grad.view(-1) for w in self.f.parameters()])

        return w_grad_prior + (self.N / self.M) * w_grad_likelihood

    def sgld_step_selfimpl(self, x, y, lr):
        grad = self.log_joint_gradient_selfimpl(x, y)
        with torch.no_grad():
            noise = torch.randn_like(grad) * math.sqrt(lr)
            w_deltas = (lr / 2) * grad + noise
            offset = 0
            for w in self.f.parameters():
                wslice = w.numel()
                w_delta = w_deltas[offset : offset + wslice].view(w.shape)  
                w.add_(w_delta)
                offset += wslice





def bayesian_distillation(tr_items, st_items, msc_items, tr_hyp_par, val_step, T_total=1e10, verbose=True):
    tr_bayers, tr_network, tr_loader_train, tr_loader_valid = tr_items
    st_network, st_optim, st_scheduler, tr_st_criterion = st_items
    B, H, criterion, device = msc_items
    lr_init, decay_gamma, lr_b = tr_hyp_par
    train_iterator = itertools.cycle(tr_loader_train)
    results = []



    #So I dont run out of memory on the GPU
    tr_W_samples = collections.deque(maxlen=1000)

    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    print(f"--- Starting Distillation Process for {T_total} steps ---")
    T = tqdm(range(T_total), desc="Total Steps", disable=not verbose, bar_format=bar_format)
    

    # tr_criterion_nll = nn.CrossEntropyLoss(reduction='mean')

    tr_cur_nll_val = float('inf') # Use infinity as a placeholder
    st_cur_nll_val = float('inf')

    for t in T:
        lr = lr_init * (lr_b + t)**(-decay_gamma)
        T.set_postfix({
            'LR': f"{lr:.2e}",
            'Teacher NLL': f"{tr_cur_nll_val:.4f}",
            'Student NLL': f"{st_cur_nll_val:.4f}"
        })

        inputs, labels = next(train_iterator)
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(inputs.size(0), -1)

        #tr_bayes' have tr_network inside it
        # tr_bayers.sgld_step(inputs, labels, lr)
        tr_bayers.sgld_step_selfimpl(inputs, labels,lr)

        if t >= (B-1000):
            current_weights = {k: v.cpu().clone() for k, v in tr_network.state_dict().items()}
            tr_W_samples.append(current_weights)

        if t >= B and (t % H == 0):
            if len(tr_W_samples) < 100:
                    continue 
            st_network.train()
            tr_network.eval()

            with torch.no_grad():
                teacher_logits = tr_network(inputs)

            # noise_std = 0.001
            # noise = torch.randn_like(inputs) * noise_std
            # noisy_inputs = inputs + noise
          
            #forward pass
            # student_logits = st_network(noisy_inputs)
            student_logits = st_network(inputs)
            tr_targets = F.softmax(teacher_logits, dim=-1)
            st_log_probs = F.log_softmax(student_logits, dim=-1)
            
            #Loss
            st_loss = -torch.sum(tr_targets * st_log_probs) / student_logits.size(0)
            
            st_optim.zero_grad()
            st_loss.backward()
            st_optim.step()
            if st_scheduler:
                st_scheduler.step()



        if t >= B and (t % val_step == 0):
            tr_network.eval()  

            with torch.no_grad():            
                tr_W_samples_rng = random.sample(list(tr_W_samples), 100)                

                tr_nll_train, tr_acc_train = get_bayesian_train_metrics(tr_network, tr_W_samples_rng, inputs, labels, device)
                tr_nll_val, tr_acc_val = validate_network_bayesian(tr_network, tr_W_samples_rng, tr_loader_valid, device)
                st_nll_val, st_acc_val = validate_network(st_network, tr_loader_valid, criterion, device)

                
                tr_cur_nll_val = tr_nll_val
                st_cur_nll_val = st_nll_val              

                
            results.append({
                't': t + 1,
                'tr_nll_val': tr_nll_val,
                'tr_acc_val': tr_acc_val,
                'tr_nll_train': tr_nll_train,
                'tr_acc_train': tr_acc_train,
                'st_nll_val': st_nll_val,
                'st_acc_val': st_acc_val
            })        

            tr_network.train()


    return results, tr_network.state_dict()