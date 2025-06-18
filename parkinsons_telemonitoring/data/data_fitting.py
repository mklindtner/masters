import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib


df = pd.read_csv('data/parkinsons_data.csv')
y = df[['total_UPDRS']]
X_full = df.drop(columns=['total_UPDRS', 'motor_UPDRS'])
X = X_full.drop(columns=['HNR'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

scaler_X.fit(X_train)
scaler_y.fit(y_train)

X_train_scaled = scaler_X.transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)

X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)

TENSOR_DATA_FILE = 'data/fitted_tensors.pt'
SCALER_X_FILE = 'data/scaler_X.joblib'
SCALER_Y_FILE = 'data/scaler_y.joblib'

# --- 1. Save all tensors in a single dictionary file ---
torch.save({
    'X_train': X_train_tensor,
    'y_train': y_train_tensor,
    'X_test': X_test_tensor,
    'y_test': y_test_tensor,
}, TENSOR_DATA_FILE)


joblib.dump(scaler_X, SCALER_X_FILE)
joblib.dump(scaler_y, SCALER_Y_FILE)

print(f"Tensors saved to '{TENSOR_DATA_FILE}'")
