import numpy as np
from perceptron import Perceptron
from visualizer import Visualizer

# --- 1. DATA DEFINITION ---
X = np.array([[3,3],[3,3],[2,3],[4,2],[5,1],[1,5],[5,3],[-1,5],[1,6],[5,5],
              [1,1],[2,1],[0,3],[-1,3],[6,-1],[0,0],[-1,-1],[3,1]])
y = np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0])

# --- 2. EXECUTION ---
# Initialize the model
model = Perceptron(n_features=2, learning_rate=0.1, epochs=100)

# Train the model
model.train(X, y)

# Access Results
print(f"Final Weights: {model.weights}")
print(f"Final Bias: {model.bias}")

# --- 3. VISUALIZATION ---
viz = Visualizer(X, y)
viz.plot_loss(model.epoch_loss)
viz.animate_learning(model.history)