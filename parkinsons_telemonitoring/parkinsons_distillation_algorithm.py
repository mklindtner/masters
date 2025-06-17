import itertools
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from parkinsons_model import validate_network


def distillation_posterior_MNIST(tr_items, st_items, msc_list, T_total=1e10, verbose=True):    
    tr_optim, tr_network, tr_loader_train, tr_loader_valid = tr_items
    st_network, st_optim, st_scheduler, U, tr_st_criterion = st_items
    B, H, criterion, device = msc_list       
    V = 500; s = 0

    #Setup iterator so I dont get a stupid out of elements err
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
            # teacher_targets = U(tr_network, distill_inputs).detach()
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

            # T.set_postfix(Distill_Loss=f"distill_loss: {distill_loss.item():.4f}", Student_NLL=f"st_nll: {student_nll:.4f}", loss=f"tr_nll: {teacher_nll:.4f}")
        # elif t < B:
        #     T.set_postfix(Status="Teacher Burn-in")

    
    print("--- Finished Distillation Process ---")
    return results