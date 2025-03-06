from torch import nn
from models import TeacherFCNNMNIST, StudentFCNNMNIST
from algorithm1 import GPED
from prepare_data import MNIST_teacher_data


#MNIST experiment1

nll = nn.NLLLoss()
tloader_train,tloader_test = MNIST_teacher_data()
BMNIST = 1000
TMNIST = 10e6
#Predictive posterior distribution i U_s 
etas = 4*-10e6
priors = [0,10] #[mean,precision]
a_s = -10e3
st_dropout = 0.5
H_posterior = 100

