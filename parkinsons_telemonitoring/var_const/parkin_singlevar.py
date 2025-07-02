# plot_linear_fit.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from constants import path_parkin_data_csv
import pandas as pd
# Import your existing data loading function
from parkinsons_telemonitoring.data.parkinsons_dataloaders import parkinsons_dataloaders

# --- 1. Define the Simple Linear Model ---
class SimpleLinearRegression(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# --- 2. New Plotting Function ---
def plot_predictions_vs_actuals(y_true, y_pred):
    """Generates and saves a scatter plot of predicted vs. actual values."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create the scatter plot of data points
    ax.scatter(y_true, y_pred, alpha=0.5, label='Model Predictions')
    
    # Create the 'line of perfect prediction' (y=x)
    lims = [
        min(ax.get_xlim(), ax.get_ylim()),
        max(ax.get_xlim(), ax.get_ylim()),
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Prediction')
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Actual UPDRS Scores", fontsize=12)
    ax.set_ylabel("Predicted UPDRS Scores", fontsize=12)
    ax.set_title("Linear Regression: Predicted vs. Actual Values", fontsize=14)
    ax.legend()
    
    # Save the figure and show it
    output_filename = "linear_regression_fit.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nPlot saved to {output_filename}")


def statistics():
    """
    Loads data using the DataLoader, extracts all target values,
    and computes summary statistics on the standardized data.
    """
    print("Loading and processing data through DataLoader...")
    
    # We can use any batch size here, as we will concatenate all batches.
    trainloader, testloader, _, _ = parkinsons_dataloaders(batch_size=64)
    n_train_samples = len(trainloader.dataset)
    n_test_samples = len(testloader.dataset)
    
    # --- 1. Extract all scaled labels from the DataLoader --- 
    all_labels = []
    for _, labels in trainloader:            
        # The labels are already scaled by the DataLoader's internal logic
        all_labels.append(labels)

    # --- 2. Combine all batches into a single tensor ---
    # torch.cat joins the list of tensors into one large tensor
    all_labels_tensor = torch.cat(all_labels)

    # Convert to a NumPy array for easy calculation
    labels_np = all_labels_tensor.cpu().numpy()

    print(f"Extracted {len(labels_np)} scaled data points.")

    # --- 3. Calculate and return statistics ---
    stats = {
        "min": np.min(labels_np),
        "max": np.max(labels_np),
        "variance": np.var(labels_np),
        "mean": np.mean(labels_np),
        "median": np.median(labels_np),
        "train_length": n_train_samples,
        "val_length": n_test_samples
    }
    return stats



def foo():
    df = pd.read_csv(path_parkin_data_csv)
    return df.size

# --- 3. Main Training and Evaluation Function ---
def train_and_plot():
    """Trains the model and calls the plotting function."""
    
    LEARNING_RATE = 0.01
    EPOCHS = 100
    BATCH_SIZE = 64
    INPUT_FEATURES = 18

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader, _, _ = parkinsons_dataloaders(BATCH_SIZE)
    model = SimpleLinearRegression(INPUT_FEATURES).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training simple linear model for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print("Training complete.")

    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    predictions_np = np.concatenate(all_predictions)
    labels_np = np.concatenate(all_labels)
    
    # Call the new plotting function
    plot_predictions_vs_actuals(labels_np, predictions_np)

    # --- NEW: Calculate the variance of the residuals (MSE) ---
    residuals = labels_np - predictions_np
    residual_variance = np.var(residuals)
    
    return residual_variance


if __name__ == '__main__':
    res_var = train_and_plot()
    stats = statistics()
    print(f"="*30)
    for key, value in stats.items():
        print(f"  {key.capitalize():<10}: {value:.4f}")    
    print(f"  MSE Variance:{res_var:.4f}")
    print(f"="*30)
    