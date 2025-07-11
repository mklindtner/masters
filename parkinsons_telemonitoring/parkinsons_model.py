from torch import optim, nn
import torch
import math
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.nn.utils import parameters_to_vector
import itertools
import torch.distributions as D
import collections
import random
import numpy as np
import pandas as pd

class FFC_Regression_Parkinsons(nn.Module):

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
        log_var = self.fc3_log_var(x)
        return mean, log_var

class StudentMeanOnly(nn.Module):
    """A student model architecture that only outputs a single value for the mean."""
    def __init__(self, input_size, dropout_rate=0.5):
        super().__init__()
        # Example architecture
        self.fc1 = nn.Linear(input_size, 400)
        self.fc2 = nn.Linear(400, 400)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc3_mean = nn.Linear(400, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        mean = self.fc3_mean(x)
        return mean

class StudentVarOnly(nn.Module):
    """
    A model architecture that ONLY outputs a single value for the log-variance.
    """
    def __init__(self, input_size, dropout_rate=0.5):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, 400)
        self.fc2 = nn.Linear(400, 400)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc_log_var = nn.Linear(400, 1)

    def forward(self, x):
        """Defines the forward pass of the data through the network."""
        
        # Pass through the shared layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        # The final hidden representation is fed into the single output head
        log_var = self.fc_log_var(x)
        
        return log_var # Return only one value


class BaseStudentTrainer:
    """A base class for student trainers that defines a common interface."""
    def __init__(self, student_network, student_optimizer):
        self.student_network = student_network
        self.optimizer = student_optimizer

    def train_step(self, teacher_network, inputs):
        """This method must be implemented by all subclasses."""
        raise NotImplementedError


class MeanVarianceStudentTrainer(BaseStudentTrainer):
    """Trains the student to match both the mean and variance of the teacher."""
    def __init__(self, student_network, student_optimizer):
        super().__init__(student_network, student_optimizer)
        # self.criterion = D.kl.kl_divergence
        self.criterion = nn.MSELoss()
        self.lambda_weight = 1.0

    def train_step(self, teacher_network, inputs):
        self.student_network.train()
        
        with torch.no_grad(): 
            teacher_mean, teacher_log_var = teacher_network(inputs)         
        student_mean, student_log_var = self.student_network(inputs) 
        
        #stjde tloss function is kl_divergence
        # student_dist = D.Normal(student_mean, torch.exp(student_log_var).sqrt())
        # teacher_dist = D.Normal(teacher_mean, torch.exp(teacher_log_var).sqrt())
        # kl_div_per_element = self.criterion(student_dist, teacher_dist)
        # loss = kl_div_per_element.sum()

        loss_mean = self.criterion(student_mean, teacher_mean)
        loss_var = self.criterion(student_log_var, teacher_log_var)
        total_loss = loss_mean + self.lambda_weight * loss_var

        #optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'mse_mean': loss_mean.item(),
            'mse_var': loss_var.item()
        }

class MeanOnlyStudentTrainer(BaseStudentTrainer):
    """Trains the student to match only the mean of the teacher."""
    def __init__(self, student_network, student_optimizer):
        super().__init__(student_network, student_optimizer)
        self.criterion = nn.MSELoss()

    def train_step(self, teacher_network, inputs):
        self.student_network.train()

        with torch.no_grad():
            # Get the teacher's mean prediction
            teacher_mean, _ = teacher_network(inputs)
            
        # Get the student's mean prediction
        student_mean = self.student_network(inputs)
        
        # The loss is the Mean Squared Error between the two means
        loss = self.criterion(student_mean, teacher_mean)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class VarianceOnlyStudentTrainer(BaseStudentTrainer):
    """Trains the student to match only the variance of the teacher."""
    def __init__(self, student_network, student_optimizer):
        super().__init__(student_network, student_optimizer)
        self.criterion = nn.MSELoss()

    def train_step(self, teacher_network, inputs):
        self.student_network.train()

        with torch.no_grad():
            _, teacher_log_var = teacher_network(inputs)
            
        student_log_var = self.student_network(inputs)
        
        loss = self.criterion(student_log_var, teacher_log_var)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()



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


def validate_student_mse(model, validation_loader, device, verbose=False):
    """Calculates the Mean Squared Error (MSE) for a student model."""
    model.eval()
    total_squared_error = 0.0
    total_samples = 0
    
    val_loop = tqdm(validation_loader, desc="Validating Student (MSE)", leave=False, disable=not verbose)
    
    with torch.no_grad():
        for inputs, labels in val_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Flexibly get the mean prediction
            mean = outputs[0] if isinstance(outputs, tuple) else outputs
            
            batch_error = F.mse_loss(mean, labels, reduction='sum')
            total_squared_error += batch_error.item()
            total_samples += len(labels)
            
    return total_squared_error / total_samples

def validate_student_gnll(model, validation_loader, criterion, device, verbose=False):
    """Calculates the Gaussian NLL for a student model."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    val_loop = tqdm(validation_loader, desc="Validating Student (GNLL)", leave=False, disable=not verbose)
    
    with torch.no_grad():
        for inputs, labels in val_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # This model is expected to return two outputs
            mean, log_var = model(inputs)
            
            loss = criterion(input=mean, target=labels, var=torch.exp(log_var))
            total_loss += loss.item()
            total_samples += len(labels)
            
    return total_loss / total_samples


def tr_validate_network(network, weight_samples, val_loader, criterion, device):
    """Calculates the Gaussian NLL for a set of Weights from the teacher model."""
    network.eval()
    all_means_across_samples = []
    all_vars_across_samples = []
    all_targets = torch.cat([target for _, target in val_loader]).to(device)

    print(f"Performing Bayesian Model Average over {len(weight_samples)} samples...")
    # Loop over each set of weights (each sample from the posterior)
    for sample_state_dict in tqdm(weight_samples, desc="Averaging Samples"):
        network.load_state_dict({k: v.to(device) for k, v in sample_state_dict.items()})
        
        batch_means = []
        batch_vars = []

        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                
                mean, log_var = network(data)
                var = torch.exp(log_var)                
                batch_means.append(mean)
                batch_vars.append(var)
        
        all_means_across_samples.append(torch.cat(batch_means))
        all_vars_across_samples.append(torch.cat(batch_vars))
        
    #[num_samples, num_data_points, 1] 
    avg_means = torch.stack(all_means_across_samples, dim=0)
    avg_vars = torch.stack(all_vars_across_samples, dim=0)

    #get the average mand and vars    
    avg_mean = torch.mean(avg_means, dim=0)
    avg_var = torch.mean(avg_vars, dim=0)
    avg_mean, avg_var = avg_mean.to(device), avg_var.to(device)

    # Ensure all tensors are on the same device
    total_nll = criterion(input=avg_mean, target=all_targets, var=avg_var)
    avg_nll_per_sample = total_nll.item() / len(all_targets)

    return avg_nll_per_sample

# def validate_kl_divergence(teacher_network, student_network, weight_samples, val_loader, device):
#     """
#     Calculates the KL divergence between the student's predictive distribution and
#     the teacher's predictive distribution on the entire validation set.
#     """
#     teacher_network.eval()
#     student_network.eval()

#     all_teacher_means = []
#     all_teacher_vars = []
#     for sample_weights in weight_samples:
#         teacher_network.load_state_dict({k: v.to(device) for k, v in sample_weights.items()})
#         batch_means, batch_vars = [], []
#         with torch.no_grad():
#             for data, _ in val_loader:
#                 data = data.to(device)
#                 mean, log_var = teacher_network(data)
#                 batch_means.append(mean)
#                 batch_vars.append(torch.exp(log_var))

#         all_teacher_means.append(torch.cat(batch_means))
#         all_teacher_vars.append(torch.cat(batch_vars))
    
#     avg_teacher_mean = torch.mean(torch.stack(all_teacher_means), dim=0)
#     avg_teacher_var = torch.mean(torch.stack(all_teacher_vars), dim=0)
#     teacher_dist = D.Normal(avg_teacher_mean, avg_teacher_var.sqrt())

#     student_means, student_vars = [], []
#     with torch.no_grad():
#         for data, _ in val_loader:
#             data = data.to(device)
#             mean, log_var = student_network(data)
#             student_means.append(mean)
#             student_vars.append(torch.exp(log_var))
    
#     student_mean = torch.cat(student_means)
#     student_var = torch.cat(student_vars)
#     student_dist = D.Normal(student_mean, student_var.sqrt())

#     kl_div = D.kl.kl_divergence(student_dist, teacher_dist).sum()
    
#     return kl_div.item() / len(val_loader.dataset)

def tr_test_train(network, weight_samples, inputs, labels, criterion, device):
    """
    Calculates the BMA NLL for a given model and weight ensemble on a single batch.
    """
    network.eval()
    all_means = []
    all_vars = []
    
    # Loop through the teacher samples to get predictions for the batch
    for sample_weights in weight_samples:
        # Load weights, ensuring they are on the correct device
        network.load_state_dict({k: v.to(device) for k, v in sample_weights.items()})
        
        mean, log_var = network(inputs)
        all_means.append(mean)
        all_vars.append(torch.exp(log_var))

    # Average the parameters across all samples
    avg_mean = torch.mean(torch.stack(all_means), dim=0)
    avg_var = torch.mean(torch.stack(all_vars), dim=0)
    
    # Calculate the NLL using the averaged predictive distribution
    total_nll = criterion(input=avg_mean, target=labels, var=avg_var)
    
    # Return the average NLL per sample
    return total_nll.item() / len(labels)

def bayesian_distillation_parkin(tr_items, st_items, msc_items, tr_hyp_par, val_step, T_total=1e6, verbose=True):
    tr_bayers, tr_network, tr_loader_train, tr_loader_valid = tr_items
    st_network, st_optim, st_trainer = st_items
    B, H, eval_criterion, device = msc_items
    lr_init, decay_gamma, lr_b = tr_hyp_par

    train_iterator = itertools.cycle(tr_loader_train)
    results = []
    st_losses = [] 

    W_len, sample_sz = 1000, 100
    tr_W_samples = collections.deque(maxlen=W_len)

    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    print(f"--- Starting Distillation Process for {T_total} steps ---")
    T = tqdm(range(T_total), desc="Total Steps", disable=not verbose, bar_format=bar_format)


    for t in T:
        lr = lr_init * (lr_b + t)**(-decay_gamma)
        T.set_postfix(LR=f"{lr:.2e}") 
        inputs, labels = next(train_iterator)
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(inputs.size(0), -1)

        #The tr_network should be inside here
        tr_network.train()
        tr_bayers.sgld_step(inputs, labels, lr)

        if t >= (B-W_len):
            current_weights = {k: v.cpu().clone() for k, v in tr_network.state_dict().items()}
            tr_W_samples.append(current_weights)

        if t >= B and (t % H == 0):
            loss_val = st_trainer.train_step(teacher_network=tr_network,inputs=inputs)
            st_losses.append(loss_val)


        if t >= B and (t % val_step == 0):
            if len(tr_W_samples) < W_len:
                    continue 

            T.set_postfix_str("VALIDATING...")

            st_mse_val, st_nll_val = float('nan'), float('nan')
            st_total_loss_train, st_mse_mean_train, st_mse_var_train = float('nan'), float('nan'), float('nan')

            with torch.no_grad():       
                tr_W_samples_rng = random.sample(list(tr_W_samples), sample_sz)
                tr_nll_val = tr_validate_network(tr_network, tr_W_samples_rng, tr_loader_valid, eval_criterion, device)
                tr_nll_train = tr_test_train(tr_network, tr_W_samples_rng, inputs, labels, eval_criterion, device)

                student_mode_name = st_trainer.__class__.__name__
                
                if student_mode_name == 'MeanVarianceStudentTrainer':
                    st_nll_val = validate_student_gnll(st_trainer.student_network, tr_loader_valid, eval_criterion, device)
                    st_losses_df = pd.DataFrame(st_losses)
                    st_total_loss_train = st_losses_df['total_loss'].iloc[-1]
                    # st_mse_mean_train = st_losses_df['mse_mean'].mean()
                    st_mse_mean_train = st_losses_df['mse_mean'].iloc[-1]
                    st_mse_var_train = st_losses_df['mse_var'].iloc[-1]

                elif student_mode_name == 'MeanOnlyStudentTrainer':
                    st_mse_val = validate_student_mse(st_trainer.student_network, tr_loader_valid, device)
                    # st_mse_mean_train = np.mean(st_losses)
                    st_mse_mean_train = st_losses[-1]

                elif student_mode_name == 'VarianceOnlyStudentTrainer':
                    st_mse_val = validate_student_mse(st_trainer.student_network, tr_loader_valid, device)
                    # st_mse_var_train = np.mean(st_losses)
                    st_mse_var_train = st_losses[-1]
              
                

            results.append({
                't': t + 1,
                'tr_nll_train': tr_nll_train,
                'tr_nll_val': tr_nll_val,
                'st_nll_val': st_nll_val,       # Will be NaN for MSE modes 
                'st_mse_val': st_mse_val,       # Will be NaN for GNLL mode
                'st_total_loss_train': st_total_loss_train,
                'st_mse_mean_train': st_mse_mean_train,
                'st_mse_var_train': st_mse_var_train,
            })
            st_losses = []
            tr_network.train()

    final_teacher_samples = random.sample(list(tr_W_samples), 100)

    return results, final_teacher_samples, st_trainer.student_network.state_dict()