# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from models import StudentModel, TeacherModel
from prepare_data import load_data
from hyperparams import device

def train_model(train_loader, test_loader, epochs=10, lr=0.001):
    # Initialize teacher and student models
    teacher_model = TeacherModel().to(device)
    student_model = StudentModel().to(device)

    # Assume teacher model is pretrained
    teacher_model.eval()

    # Define the loss function and optimizer
    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = optim.Adam(student_model.parameters(), lr=lr)

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Get teacher predictions
            with torch.no_grad():
                teacher_outputs = torch.log_softmax(teacher_model(inputs), dim=1)

            # Get student predictions
            student_outputs = torch.log_softmax(student_model(inputs), dim=1)

            # Compute KL divergence loss
            loss = criterion(student_outputs, teacher_outputs)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Save the trained model
    torch.save(student_model.state_dict(), "student_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    data_dir = "../data"
    train_loader, test_loader = load_data(data_dir, save=False)
    train_model(train_loader, test_loader)