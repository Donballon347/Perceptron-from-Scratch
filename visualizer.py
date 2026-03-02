import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.x_min, self.x_max = X[:, 0].min() - 2, X[:, 0].max() + 1
        self.y_min, self.y_max = X[:, 1].min() - 2, X[:, 1].max() + 1

    def plot_loss(self, loss_history):
        plt.figure(figsize=(8, 4))
        plt.plot(loss_history, color='blue')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.grid(True, alpha=0.3)
        plt.show()

    def animate_learning(self, history):
        plt.ion() 
        fig, ax = plt.subplots(figsize=(8, 6))
        xx = np.linspace(self.x_min, self.x_max, 100)

        for i, (weights, bias) in enumerate(history):
            ax.cla()
            # Plot classes
            ax.scatter(self.X[self.y == 1, 0], self.X[self.y == 1, 1], c='blue', label='Class 1')
            ax.scatter(self.X[self.y == 0, 0], self.X[self.y == 0, 1], c='red', marker='x', label='Class 0')
            
            # Boundary line
            if weights[1] != 0:
                yy = -(weights[0] * xx + bias) / weights[1]
                ax.plot(xx, yy, 'g--', label=f'Epoch {i}')
            
            ax.set_ylim(self.y_min, self.y_max)
            ax.set_title(f'Iteration: {i}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.draw()
            plt.pause(0.01)

        plt.ioff()
        plt.show()