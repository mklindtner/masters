from torch.utils.data import TensorDataset, DataLoader
import torch
import joblib
from constants import SCALER_X_FILE, SCALER_Y_FILE, TENSOR_DATA_FILE


def parkinsons_dataloaders(batch_size=64):    
    data = torch.load(TENSOR_DATA_FILE)
    X_train_tensor = data['X_train']
    y_train_tensor = data['y_train']
    X_test_tensor = data['X_test']
    y_test_tensor = data['y_test']
    
    scaler_X = joblib.load(SCALER_X_FILE)
    scaler_y = joblib.load(SCALER_Y_FILE)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, scaler_X, scaler_y