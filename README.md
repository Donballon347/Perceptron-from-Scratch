# Simple Perceptron with Real-time Visualization

A Python implementation of a single-layer Perceptron from scratch using NumPy. This project demonstrates binary classification with a Sigmoid activation function, Cross-Entropy loss, and an animated decision boundary that updates during training.

## Mathematical Foundation

### 1. Weighted Sum
The input features $X$ are multiplied by weights $w$ and added to a bias $b$:
<img width="233" height="85" alt="image" src="https://github.com/user-attachments/assets/d94cee6a-8968-4d13-a33c-393c8f98039f" />
<img width="233" height="85" alt="image" src="https://github.com/user-attachments/assets/d94cee6a-8968-4d13-a33c-393c8f98039f" />

### 2. Activation Function (Sigmoid)
The sum is squashed into a $(0, 1)$ range:
<img width="179" height="66" alt="image" src="https://github.com/user-attachments/assets/0a8fc48c-d44e-4dac-a8ff-3702673b41bb" />
<img width="179" height="66" alt="image" src="https://github.com/user-attachments/assets/0a8fc48c-d44e-4dac-a8ff-3702673b41bb" />

### 3. Loss Function (Cross-Entropy)
We measure the error between the prediction $\hat{y}$ and the actual target $y$:
<img width="422" height="43" alt="image" src="https://github.com/user-attachments/assets/a7025a45-c30c-450c-bbb5-da9644c76d98" />
<img width="422" height="43" alt="image" src="https://github.com/user-attachments/assets/a7025a45-c30c-450c-bbb5-da9644c76d98" />
