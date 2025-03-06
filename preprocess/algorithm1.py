#Generalized Posterior Expectation Distillation
import torch
import torch.optim as optim
from torch.distributions import multivariate_normal




class GPED:
    def __init__(self,g, t_dataloader, st_dataloader, priors, st_optim, etas, alphas):
        self.g = g
        self.t_dataloader = t_dataloader
        self.st_dataloader = st_dataloader
        self.priors = priors
        self.st_optim = st_optim
        self.etas = etas
        self.alphas = alphas



    def Us(self, y,xi,theta_t):
        return self.g(y,xi,theta_t)


    def Ucirc(self, g_yis, theta_t, m_is, y, x_i):
        g_yi = self.g(y, x_i, theta_t)
        m_is_new = m_is + 1
        g_yis_new = (m_is * g_yis + g_yi) / m_is_new
        return g_yis_new, m_is_new


    def sample_teacher(self, step, teacher, xi,yi, criterion, opt):
        teacher.train()
        opt.zero_grad()

        M = len(xi)
        N = len(self.t_dataloader)*self.t_dataloader.batch_size

        ty_pred = teacher(xi)
        
        # tloss = criterion(ty_pred, yi)
            
        # tloss.backward()

        # #Teacher: keep only the latest weights in memory
        # log_prior_grad = {}
        # for name, param in teacher.named_parameters():
        #     log_prior_grad[name] = (param - self.priors) / (param ** 2).sum().sqrt()


        # #Teacher: Update likelihood and prior manually
        # with torch.no_grad():
        #     for name, param in teacher.named_parameters():
        #         grad_log_likelihood = param.grad

        #         #z_t = torch.rand_like(param)
        #         grad_log_prior = log_prior_grad[name]
        #         delta_theta = (self.etas[step] / 2) * (grad_log_prior + (N / M) * grad_log_likelihood) + torch.rand_like(param)
        #         param.add_(delta_theta)



    def update_student(self, step, student, gyis, criterion, opt, lambda_reg, reg):
        xi,_ = next(self.st_dataloader)

        student.train()
        opt.zero_grad()

        Nmark = len(self.st_dataloader)*self.st_dataloader.batch_size
        Mmark = len(xi)
            
        pred = student(xi)
        loss = criterion(pred, gyis)
        loss.backward()

        self.phi = 0
        phi_update = self.alphas[step]*(Nmark/Mmark)*loss + lambda_reg*reg

        ... = ?

        return ?, step+1



    def gped(self,U,teacher,f,):
        #initialize parameters
        s = 0
        gyis = 0
        mis = torch.empty((len(self.st_dataloader)*self.st_dataloader.batch_size),len(self.st_dataloader))

        #Initialize teacher,student parameters
        theta_t = teacher.paremeters().data 
        phi_s = f.parameters().data    




# #Instead of Dataset, we are going to give the dataloader
#     #Where both datasets have M=64 as minibatch
# #the length
# """
# dataloader is expected to have batchsize S = |M|
# dataloader unlabelled must have same length as dataloader
# teacher = p(y|x,theta) in the paper
# priors = theta^0 in paper
# T is the epochs

# """
# def GPED(dataloader, dataloader_unlabelled,teacher,priors, g, f,U,ell,R, H,B,lambda_reg, etas,alphas):
    
#     #Initialize teacher,student parameters
#     theta_t = teacher.paremeters().data 
#     phi_s = f.parameters().data    

#     #ttheta_grad = {name: param.clone() for name, param in teacher.named_parameters()}  # Prior parameters of teacher model
    
#     #Initialize parameters
#     T = 300
#     M = dataloader.batch_size
#     Mmark = dataloader_unlabelled.batch_size


#     #I am understanding burn in to be the number of epochs we want to burn before training student.
#     #|S| = M, where M is the batchsize
#     #i = one data point i, i.e. 1 observation

#     mis = torch.empty((len(dataloader_unlabelled)*Mmark,Mmark))
#     mis[:,0] = 0
#     gyis = torch.empty((len(dataloader_unlabelled)*Mmark, Mmark))
#     gyis[:,0] = 0

#     #S number of samples
#     #N is the total number of obersvations for labelled dataloader
#     #Nmark is the total number of observations for unlabelled dataloader
#     S = 0
#     N = len(dataloader)*M
#     Nmark = len(dataloader_unlabelled)*Mmark
    
#     #crossentropy because they didnt sepcify the loss function for the teacher

#     tcriterion = torch.nn.CrossEntropyLoss()
#     optimizer = optim.SGD(teacher.parameters(), lr=0.01)


#     for t in range(T):
#         #Get the iterable used for 1-step for the unlabelled dataloader 
#         #if I do this in the loop, i end upgenerating a new iterator for every batch, which seems inefficient
#         #Ofc dont do this, if you want to load the entire dataloader everytime mod is hit
#         st_dl = iter(dataloader_unlabelled)


#         for st_datapoints, labels in dataloader:
            
#             ty_pred = teacher(st_datapoints)
#             tloss = tcriterion(ty_pred, labels)
            
#             optimizer.zero_grad()
#             tloss.backward()
        
#             #holds latest version of theta_t
#                 #Unsure if
#                     #Should hold all a parameters of teacher
#                     #This is the right place to update??
#             theta_t = teacher.parameters().data

#             #Teacher: keep only the latest weights in memory
#             log_prior_grad = {}
#             for name, param in teacher.named_parameters():
#                 log_prior_grad[name] = (param - priors) / (param ** 2).sum().sqrt()


#             #Teacher: Update likelihood and prior manually
#             with torch.no_grad():
#                 for name, param in teacher.named_parameters():
#                     grad_log_likelihood = param.grad

#                     #z_t = torch.rand_like(param)
#                     grad_log_prior = log_prior_grad[name]
#                     delta_theta = (etas[t] / 2) * (grad_log_prior + (N / M) * grad_log_likelihood) + torch.rand_like(param)
#                     param.add_(delta_theta)
                

#             if t % H == 0 and t > B:
#                 #next input from student dataloader
#                 st_datapoints, st_labels = next(st_dl)
                                
#                 for i, input in enumerate(st_datapoints):
#                     # est_mean = gyis[id, theta_t,S]
#                     gyis[i, theta_t, S+1] = U(gyis[i,S], theta_t, mis[i, S])

#                     #Problem A
#                     mis[i,S+1] += mis[i, S] + 1


#                 #Update students parameters

#                 phi_s = alphas[S]*(Nmark/Mmark) ... + lambda_reg
#                 S += 1
            



#Plan

#Set for experiement 1
#Set T
#Generate learnings rates (etas)
#generate noise N(0,eta_t * I)


#define prior 
#theta0 = multivariate_normal.MultivariateNormal(torch.zeros(2), torch.eye(2)).sample()


#Problem A

 #I am understading the paper as haivng to keep track of which data case i, (i.e. 64-sample) I need to keep track of
                            #However because I only sample sporadically this means I will always not sample from the "backend" of the batch if i dont shuffle (a)
                                #However if I do shuffle I can't automatically keep track of the actual id of the sample (b)
                                    #e.g. I could get sample2 in one epoch for i=0 and sample32 for another epoch where i=0
                                        #To fix this I need ot write my own dataloader and return the index from the specific sample
                            #For now I go with (a) and just not use the last of the samples
                        #I am also assuming that a batch is a single data case
                        # mis_id = i*dataloader.batch_size + sample_iteartion
