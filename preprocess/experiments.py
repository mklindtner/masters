from models import StudentFCNNMNIST, TeacherFCNNMNIST
from prepare_data import MNIST_distillation_data
from hyperparams import device, epochsMNIST
from algorithm1 import GPED

def experiment1():
    data_loader,test_loader = MNIST_distillation_data()
    
    teacher_model = TeacherFCNNMNIST().to(device)
    student_model = StudentFCNNMNIST().to(device)

    student_model.train()
    etas = 
    GPED(data_loader, test_loader, teacher_model, student_model, )
    #for epoch in range(epochsMNIST):
        

if __name__ == "__main__":
    # data_dir = "../data"
    # train_loader, test_loader = data_preprocess_mnist(save=False)
    # train_model(train_loader, test_loader)
    experiment1()