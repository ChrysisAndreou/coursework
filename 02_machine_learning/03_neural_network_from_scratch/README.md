# Neural Network for-Binary-Classification-From-Scratch

This repository contains a Python implementation of a neural network from scratch to solve a binary classification problem. The project is part of the MAI612 - Machine Learning course.

## Goal

The primary goal of this project is to implement a neural network, including its structure, feed-forward mechanism, and backpropagation algorithm, without relying on any machine learning libraries like TensorFlow or PyTorch. This hands-on approach aims to foster a deeper understanding of the inner workings of neural networks.

## Features

*   **Neural Network from Scratch:** A `NeuralNetwork` class built using only NumPy.
*   **Customizable Architecture:** Easily set the number of input, hidden, and output nodes.
*   **Hyperparameter Tuning:** Dynamically adjust the learning rate and momentum.
*   **Stochastic Gradient Descent (SGD):** The network is trained using SGD for weight updates.
*   **Comprehensive Experiments:** The project includes a series of experiments to analyze the impact of different hyperparameters on the model's performance.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/neural-network-binary-classification.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd neural-network-binary-classification
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main script `code.py` contains the `NeuralNetwork` class and the experiments conducted on the breast cancer dataset. To run the code and see the results, simply execute the script:

```bash
python code.py
```

The script will:
1.  Define the `NeuralNetwork` class.
2.  Load the breast cancer dataset from scikit-learn.
3.  Run a series of experiments to evaluate the impact of:
    *   Learning Rate
    *   Number of Hidden Nodes
    *   Momentum
4.  Perform hyperparameter finetuning to find the best combination of learning rate, hidden nodes, and momentum.
5.  Print the results and generate plots to visualize the training process and outcomes.

## Experiments and Results

This project conducts several experiments to understand the behavior of the neural network under different hyperparameter settings.

### B.1 Training Example

*   **Settings:**
    *   `learning_rate`: 0.0001
    *   `momentum`: 0
    *   `hidden_nodes`: 256
    *   `epochs`: 1400
*   **Observation:** The training loss decreases steadily, but the test loss fluctuates, indicating potential overfitting.

### B.2.1 Learning Rate Experiments

*   **Learning Rates Tested:** `[0.0001, 0.001, 0.01, 0.1, 1]`
*   **Findings:** A learning rate of `0.0001` provided the most stable training, while higher learning rates led to instability and divergence.

### B.2.2 Hidden Layer Nodes Experiments

*   **Hidden Nodes Tested:** `[8, 16, 32, 64, 128, 256, 512]`
*   **Findings:** The accuracy improves as the number of hidden nodes increases up to a certain point (`32` nodes), after which the model starts to overfit.

### B.2.3 Momentum Experiments

*   **Momentum Values Tested:** `[0, 0.1, 0.25, 0.5, 1]`
*   **Findings:** In the initial experiments, momentum did not significantly improve performance and, at higher values, led to instability. However, the final hyperparameter tuning revealed that a higher momentum (`0.8`) with other well-tuned parameters yields the best results.

### B.2.4 Final Neural Network – Hyperparameter Finetuning

A grid search was performed to find the optimal combination of hyperparameters.

*   **Best Hyperparameters:**
    *   `learning_rate`: 5e-05
    *   `hidden_nodes`: 256
    *   `momentum`: 0.8
    *   `batch_size`: 32
*   **Results with Best Hyperparameters:**
    *   **Final Training Loss:** 0.0667
    *   **Test Accuracy:** 96.49%

## Conclusion

This project successfully demonstrates the implementation of a neural network from scratch. The experiments highlight the critical role of hyperparameter tuning in achieving optimal model performance. While building from the ground up provides invaluable insights, for practical applications, using established libraries like TensorFlow or PyTorch is recommended due to their optimized performance and additional features.
