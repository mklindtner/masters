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



class SGLD(optim.Optimizer):
    """
    SGLD Optimizer for teacher / From the masters we know L2 regularization (weight decay) is equivalent to the prior.
    """
    def __init__(self, params, lr=1e-3, weight_decay=0, N=1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, weight_decay=weight_decay, N=N)
        super(SGLD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            N = group['N']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                #l2 regularization = prior w. N(0,1/tau)
                #grad L2 = tau*theta
                prior_grad = -weight_decay * p.data

                #This is funky AF:
                    #mean_gradient = p.data.grad = (1/M) * sum_of_gradients
                        #This means it is the mean of the batches gradient                    
                    #(N/M) * sum_of_gradients = (N/M) * (M * mean_gradient) = N * mean_gradient                
                ll_grad =  -N * p.grad
                
                gradient_step = 0.5 * lr * (prior_grad + ll_grad)

                #White noise
                noise = torch.randn_like(p.data) * math.sqrt(lr)

                w_update = gradient_step + noise
                
                #correct update
                p.data.add_(w_update)

                #update to debug gradient
                # p.data.add_(gradient_step)
                


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




### We try again from scratch my distil has problems when using full neuraln etworks, paritculary that I dont know the fucking analytical distirbution so what is my log likelihood? ITS FUCKING GONE MATE:( 

def validate_network(model, validation_loader, criterion, device, verbose=True):
    """Helper function to evaluate the model on the validation set."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    # Create a tqdm progress bar for the validation loop if verbose is True
    val_loop = tqdm(validation_loader, desc="Validating", leave=False, disable=not verbose)
    
    with torch.no_grad():
        for inputs, labels in val_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
    return total_loss / total_samples


def train_teacher_network(tr_optim, tr_network, T_epochs, tr_loader_train, tr_loader_valid, criterion, device, verbose=True):
    num_params = len(parameters_to_vector(tr_network.parameters()))
    # print(f"Teacher network has {num_params} parameters.")
    
    samples_theta = torch.empty((T_epochs, num_params), device='cpu')
    tr_nll = []

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
        tr_nll.append(validation_nll)
        
        current_params = parameters_to_vector(tr_network.parameters()).clone().detach().cpu()
        samples_theta[t] = current_params
        
        if verbose:
            T.set_postfix(Validation_NLL=f"{validation_nll:.4f}")

    print("--- Finished Teacher Training ---")
    return samples_theta, tr_nll


# Equation (9) from "posterior distillation"
def U_s(teacher_model, inputs):
    teacher_model.eval()
    with torch.no_grad():
        targets = teacher_model(inputs)
    return targets




def distillation_posterior_MNIST(tr_items, st_items, msc_list, T_total=1e10, verbose=True):    
    tr_optim, tr_network, tr_loader_train, tr_loader_valid = tr_items
    st_network, st_optim, st_scheduler, U, tr_st_criterion = st_items
    B, H, criterion, device = msc_list       
    V = 500; s = 0

    train_iterator = itertools.cycle(tr_loader_train)
    
    results = []


    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

    print(f"--- Starting Distillation Process for {T_total} steps ---")
    
    # //Modified
    # The tqdm constructor now uses our custom bar_format.
    T = tqdm(range(T_total), desc="Total Steps", disable=not verbose, bar_format=bar_format)


    for t in T:
        tr_network.train()        
        inputs, labels = next(train_iterator)
        
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(inputs.size(0), -1)

        tr_optim.zero_grad()
        outputs = tr_network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        tr_optim.step()

        if t >= B and (t % H == 0):   
            T.set_postfix(Status="Distilling")
            st_network.train() 
            tr_network.eval()  
            distill_inputs, _ = next(train_iterator)
            distill_inputs = distill_inputs.to(device).view(distill_inputs.size(0), -1)


            with torch.no_grad():
                teacher_logits = U(tr_network, distill_inputs)

            #I ahve no idea 
            student_logits = st_network(distill_inputs)
            soft_targets = F.log_softmax(teacher_logits, dim=-1)
            soft_predictions = F.log_softmax(student_logits, dim=-1)
            st_loss = tr_st_criterion(soft_predictions, soft_targets)


            st_optim.zero_grad()
            st_loss.backward()
            st_optim.step()
            s+= 1

            if s % 200 == 0:
                st_scheduler.step()
                print(f"\nStep {t+1}: Student LR decayed.")

            
            student_nll = validate_network(st_network, tr_loader_valid, criterion, device, verbose=False)
            teacher_nll = validate_network(tr_network, tr_loader_valid, criterion, device, verbose=False)
            results.append({
                't': t + 1,
                'tr_nll': teacher_nll,
                'st_nll': student_nll
            })        
    
    print("--- Finished Distillation Process ---")
    return results