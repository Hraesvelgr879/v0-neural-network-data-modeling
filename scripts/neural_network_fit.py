"""
Neural Network Scatter Plot Fitting
This script trains a feedforward neural network to model nonlinear relationships in scatter plot data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

# ============================================================================
# HYPERPARAMETERS - Adjust these to tune the model
# ============================================================================
LEARNING_RATE = 0.01      # Controls how quickly the model learns (0.001 - 0.1 typical)
EPOCHS = 5000             # Number of training iterations (1000 - 10000 typical)
HIDDEN_LAYER_SIZE = 64    # Number of neurons in hidden layers (32 - 128 typical)
RANDOM_SEED = 42          # For reproducible results

# Set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data_from_points():
    """
    Manually defined data points from the scatter plot.
    Returns: numpy arrays of x and y coordinates
    """
    data_points = [
        (2, 4), (3, 7), (4, 9), (5, 6), (6, 8), (7, 10), (8, 9), (9, 5),
        (10, 7), (11, 8), (12, 11), (13, 13), (14, 13), (15, 12), (16, 14),
        (17, 12), (18, 10), (19, 13), (20, 15), (22, 17)
    ]
    
    x_data = np.array([point[0] for point in data_points], dtype=np.float32)
    y_data = np.array([point[1] for point in data_points], dtype=np.float32)
    
    return x_data, y_data

def load_data_from_csv(filepath):
    """
    Load data from a CSV file with columns: x, y
    Args:
        filepath: Path to the CSV file
    Returns: numpy arrays of x and y coordinates
    """
    x_data = []
    y_data = []
    
    with open(filepath, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            x_data.append(float(row['x']))
            y_data.append(float(row['y']))
    
    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.float32)

# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class FeedforwardNN(nn.Module):
    """
    Simple feedforward neural network with two hidden layers.
    Uses ReLU activation for nonlinear modeling capability.
    """
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(FeedforwardNN, self).__init__()
        
        # Define network layers
        # Input layer -> First hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # First hidden layer -> Second hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Second hidden layer -> Output layer
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # ReLU activation function for nonlinearity
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x: Input tensor
        Returns: Output predictions
        """
        # Pass through first layer with ReLU activation
        x = self.relu(self.fc1(x))
        
        # Pass through second layer with ReLU activation
        x = self.relu(self.fc2(x))
        
        # Output layer (no activation for regression)
        x = self.fc3(x)
        
        return x

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def normalize_data(x_data, y_data):
    """
    Normalize data to improve training stability.
    Returns: normalized data and scaling parameters for inverse transform
    """
    x_mean, x_std = x_data.mean(), x_data.std()
    y_mean, y_std = y_data.mean(), y_data.std()
    
    x_normalized = (x_data - x_mean) / x_std
    y_normalized = (y_data - y_mean) / y_std
    
    return x_normalized, y_normalized, (x_mean, x_std, y_mean, y_std)

def denormalize_predictions(y_pred, y_mean, y_std):
    """
    Convert normalized predictions back to original scale.
    """
    return y_pred * y_std + y_mean

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, x_train, y_train, learning_rate, epochs):
    """
    Train the neural network model.
    Args:
        model: Neural network model
        x_train: Training input data
        y_train: Training target data
        learning_rate: Learning rate for optimizer
        epochs: Number of training iterations
    Returns: List of loss values during training
    """
    # Mean Squared Error loss function (standard for regression)
    criterion = nn.MSELoss()
    
    # Adam optimizer (adaptive learning rate, works well in practice)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Store loss history for visualization
    loss_history = []
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass: compute predictions
        predictions = model(x_train)
        
        # Compute loss
        loss = criterion(predictions, y_train)
        
        # Backward pass: compute gradients
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute new gradients
        
        # Update weights
        optimizer.step()
        
        # Store loss
        loss_history.append(loss.item())
        
        # Print progress every 500 epochs
        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return loss_history

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(x_original, y_original, x_plot, y_pred, loss_history):
    """
    Create visualization of original data, predictions, and training loss.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Data and predictions
    ax1.scatter(x_original, y_original, color='blue', s=50, alpha=0.6, 
                label='Original Data', zorder=3)
    ax1.plot(x_plot, y_pred, color='red', linewidth=2, 
             label='Neural Network Prediction', zorder=2)
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_title('Neural Network Fit to Scatter Data', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training loss
    ax2.plot(loss_history, color='green', linewidth=1.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (MSE)', fontsize=12)
    ax2.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.savefig('neural_network_fit.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'neural_network_fit.png'")
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to orchestrate the entire process.
    """
    print("=" * 70)
    print("NEURAL NETWORK SCATTER PLOT FITTING")
    print("=" * 70)
    
    # Load data (choose one method)
    print("\n[1/5] Loading data...")
    
    # Option 1: Load from manual points
    x_data, y_data = load_data_from_points()
    
    # Option 2: Load from CSV (uncomment to use)
    # csv_path = Path(__file__).parent / 'data.csv'
    # x_data, y_data = load_data_from_csv(csv_path)
    
    print(f"   Loaded {len(x_data)} data points")
    
    # Normalize data
    print("\n[2/5] Preprocessing data...")
    x_normalized, y_normalized, (x_mean, x_std, y_mean, y_std) = normalize_data(x_data, y_data)
    
    # Convert to PyTorch tensors
    x_train = torch.from_numpy(x_normalized.reshape(-1, 1))
    y_train = torch.from_numpy(y_normalized.reshape(-1, 1))
    
    # Initialize model
    print("\n[3/5] Initializing neural network...")
    model = FeedforwardNN(input_size=1, hidden_size=HIDDEN_LAYER_SIZE, output_size=1)
    print(f"   Architecture: 1 -> {HIDDEN_LAYER_SIZE} -> {HIDDEN_LAYER_SIZE} -> 1")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print(f"\n[4/5] Training model (LR={LEARNING_RATE}, Epochs={EPOCHS})...")
    loss_history = train_model(model, x_train, y_train, LEARNING_RATE, EPOCHS)
    print(f"   Final loss: {loss_history[-1]:.6f}")
    
    # Generate predictions for smooth curve
    print("\n[5/5] Generating predictions and plotting...")
    x_plot = np.linspace(x_data.min() - 1, x_data.max() + 1, 200)
    x_plot_normalized = (x_plot - x_mean) / x_std
    x_plot_tensor = torch.from_numpy(x_plot_normalized.reshape(-1, 1))
    
    # Make predictions
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        y_pred_normalized = model(x_plot_tensor).numpy()
    
    # Denormalize predictions
    y_pred = denormalize_predictions(y_pred_normalized, y_mean, y_std)
    
    # Plot results
    plot_results(x_data, y_data, x_plot, y_pred, loss_history)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nTo adjust the model:")
    print(f"  • Increase EPOCHS if loss is still decreasing")
    print(f"  • Adjust LEARNING_RATE (try 0.001 - 0.1)")
    print(f"  • Modify HIDDEN_LAYER_SIZE for model complexity")
    print("=" * 70)

if __name__ == "__main__":
    main()
