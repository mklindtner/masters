import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from experiment1_models import SGLD
import torch

from experiment1_models import FFC_Regression, U_s


# For MNIST, the mean is 0.1307 and the standard deviation is 0.3081.
transform = transforms.Compose([
    transforms.ToTensor(), # Converts a PIL Image or numpy.ndarray to a torch.FloatTensor
    transforms.Normalize((0.1307,), (0.3081,)) # Normalizes the tensor with mean and std
])
BATCH_SIZE = 64

trainset = torchvision.datasets.MNIST(root='./data',
                                      train=True,
                                      download=True,
                                      transform=transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data',
                                     train=False,
                                     download=True,
                                     transform=transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


criterion = nn.CrossEntropyLoss()
tr_st_criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
softmax = nn.Softmax()

print("MNIST dataset loaded successfully.")
print("-" * 30)

dataiter = iter(trainloader)
images, labels = next(dataiter)

print(f"Shape of one batch of images: {images.shape}")
print(f"Shape of one batch of labels: {labels.shape}")
print(f"Number of training batches: {len(trainloader)}")
# print(f"Number of test batches: {len(testloader)}")
print(f"Total training examples: {len(trainset)}")
# print(f"Total test examples: {len(testset)}")

#MNIST details




#msc. Parameters
device = None

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Apple MPS is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("No GPU available. Using CPU.")

INPUT_FEATURES = 784  
OUTPUT_FEATURES = 10

st_model = FFC_Regression(input_size=INPUT_FEATURES, output_size=OUTPUT_FEATURES).to(device)
tr_model = FFC_Regression(input_size=INPUT_FEATURES, output_size=OUTPUT_FEATURES).to(device)


T = int(1e6)
# T = int(10e6)
B = 10
H = 100


#teacher parameters - this is from the appendix of "Generalized Bayesian Posterior Expectation Distillation for Deep Neural Networks" under appendix A.2 section "Model and Distillation Hyper-Parameters"
tr_lr = 4*1e-6
tau = 10
N = 60000 #length of the dataset
M = BATCH_SIZE
l2_regularization = tau / N
tr_optim = SGLD(tr_model.parameters(), lr=tr_lr, weight_decay=l2_regularization, N=N)


#Student parameters
st_lr_init = 1e-3
st_droput = 0.5
st_optim = optim.Adam(st_model.parameters(), lr=st_lr_init)
st_scheduler = optim.lr_scheduler.StepLR(st_optim, step_size=200, gamma=0.5)

#Setup relevant items
msc_items = [B,H, criterion, device]
st_items = [st_model, st_optim, st_scheduler, U_s, tr_st_criterion]
tr_items = [tr_optim, tr_model, trainloader, testloader]





