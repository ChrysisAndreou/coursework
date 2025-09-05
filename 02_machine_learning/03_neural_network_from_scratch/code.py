import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from itertools import product
import pandas as pd
# part A 
class NeuralNetwork:
    def __init__(self, input_size, hidden_nodes, output_size, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.hidden_nodes = hidden_nodes
        # Initialize weights and biases with dtype=np.float64 , float128 does not work on my machine
        np.random.seed(2)
        # this initialization is more suitable for sigmoid function than the one suggested in the assignment
        self.input_weights = (np.random.randn(input_size, hidden_nodes) * np.sqrt(1. / input_size)).astype(np.float64)
        self.hidden_weights = (np.random.randn(hidden_nodes, output_size) * np.sqrt(1. / hidden_nodes)).astype(np.float64)
        self.input_bias = np.zeros((1, hidden_nodes), dtype=np.float64)
        self.hidden_bias = np.zeros((1, output_size), dtype=np.float64)
        # Initialize previous weight updates for momentum
        self.input_weights_update = np.zeros_like(self.input_weights, dtype=np.float64)
        self.hidden_weights_update = np.zeros_like(self.hidden_weights, dtype=np.float64)
        self.input_bias_update = np.zeros_like(self.input_bias, dtype=np.float64)
        self.hidden_bias_update = np.zeros_like(self.hidden_bias, dtype=np.float64)
        # Activations
        self.hidden_activation = None
        self.output_activation = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # x is the activation output of sigmoid function
        return x * (1 - x)

    def forward(self, X):
        # Input to hidden layer
        z_hidden = np.dot(X, self.input_weights) + self.input_bias
        self.hidden_activation = self.sigmoid(z_hidden)
        # Hidden to output layer
        z_output = np.dot(self.hidden_activation, self.hidden_weights) + self.hidden_bias
        self.output_activation = self.sigmoid(z_output)
        return self.output_activation

    def backward(self, X, y, output):
        # Output layer error
        output_error = y - output  # Error at output
        delta_output = output_error * self.sigmoid_derivative(output)
        # Hidden layer error
        hidden_error = np.dot(delta_output, self.hidden_weights.T)
        delta_hidden = hidden_error * self.sigmoid_derivative(self.hidden_activation)
        # Weight updates with momentum
        hidden_weights_update = self.learning_rate * np.dot(self.hidden_activation.T, delta_output) + self.momentum * self.hidden_weights_update
        input_weights_update = self.learning_rate * np.dot(X.T, delta_hidden) + self.momentum * self.input_weights_update
        # Bias updates
        hidden_bias_update = self.learning_rate * np.sum(delta_output, axis=0, keepdims=True) + self.momentum * self.hidden_bias_update
        input_bias_update = self.learning_rate * np.sum(delta_hidden, axis=0, keepdims=True) + self.momentum * self.input_bias_update
        # Update weights and biases
        self.hidden_weights += hidden_weights_update
        self.input_weights += input_weights_update
        self.hidden_bias += hidden_bias_update
        self.input_bias += input_bias_update
        # Store updates for momentum
        self.hidden_weights_update = hidden_weights_update
        self.input_weights_update = input_weights_update
        self.hidden_bias_update = hidden_bias_update
        self.input_bias_update = input_bias_update

    def fit(self, X_train, y_train, X_test, y_test, epochs, print_every_200=False, batch_size=None):
        train_losses = []
        test_losses = []
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle the data at the beginning of each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            # Momentum with pure SGD is unstable due to noisy single-sample updates, 
            # hindering consistent velocity buildup. Mini-batch gradient descent 
            # offers smoother updates, aligning better with momentum's need for 
            # consistent directional acceleration.
            if batch_size is None:
                # Stochastic Gradient Descent
                for i in range(n_samples):
                    X_i = X_train[i:i+1]
                    y_i = y_train[i:i+1]
                    output = self.forward(X_i)
                    self.backward(X_i, y_i.reshape(-1, 1), output)
            else:
                # Batch Gradient Descent
                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    X_batch = X_train[start:end]
                    y_batch = y_train[start:end]
                    output = self.forward(X_batch)
                    self.backward(X_batch, y_batch.reshape(-1, 1), output)
            
            # Compute train loss for the entire dataset
            train_output = self.forward(X_train)
            train_loss = np.mean((y_train.reshape(-1, 1) - train_output) ** 2)
            train_losses.append(train_loss)
            
            # Compute test loss for the entire dataset
            test_output = self.forward(X_test)
            test_loss = np.mean((y_test.reshape(-1, 1) - test_output) ** 2)
            test_losses.append(test_loss)
            
            # Print the loss every 200 epochs if enabled
            if print_every_200 and (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss}, Test Loss: {test_loss}")
        
        return train_losses, test_losses

    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)

# part B1 
# Load the dataset and split into train/test sets
from sklearn import datasets
from sklearn.model_selection import train_test_split
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Network parameters
input_size = X_train.shape[1]
hidden_nodes = 256
output_size = 1
learning_rate = 0.0001
momentum = 0
epochs = 1400

# Initialize the neural network
nn = NeuralNetwork(input_size, hidden_nodes, output_size, learning_rate, momentum)

# Print the title for part B1
print("\nPart B1: Training and Testing the Neural Network")

# Use the fit method for training
train_losses, test_losses = nn.fit(X_train, y_train, X_test, y_test, epochs, print_every_200=True)

# Analysis and Explanation
print("\nPart B1: Analysis and Explanation:")
print("1. Training loss is decreasing consistently, but test loss shows significant fluctuations after early epochs.")
print("2. The model might be overfitting, as indicated by the unstable test loss.")
print("3. Use Early Stopping to prevent overfitting by halting training when the test loss stops improving.")
print("4. Apply regularization (e.g., L2 regularization or dropout) to enhance generalization.")
print("5. Experimenting with different learning rates or increasing the momentum might also help stabilize the training process.")
print("6. Use a learning rate scheduler to reduce the learning rate gradually, helping stabilize the training process.")

# Plot the average train and test loss
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss over Epochs')
plt.legend()
plt.show()

# B.2.1 Learning Rate Experiments

# Print the title for part B2.1
print("\nPart B2.1: Learning Rate Experiments")

# Settings
momentum = 0
hidden_nodes = 32
epochs = 250
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]

final_train_losses = []
final_test_losses = []
train_accuracies = []
test_accuracies = []

for lr in learning_rates:
    # Initialize the neural network with the current learning rate
    nn = NeuralNetwork(input_size, hidden_nodes, output_size, lr, momentum)
    
    # Use the fit method for training
    train_losses, test_losses = nn.fit(X_train, y_train, X_test, y_test, epochs, print_every_200=False)
    
  
    # Print final training and test loss
    final_train_loss = train_losses[-1]
    final_test_loss = test_losses[-1]
    final_train_losses.append(final_train_loss)
    final_test_losses.append(final_test_loss)
    print(f"Learning Rate {lr}: Final Training Loss: {final_train_loss}, Final Test Loss: {final_test_loss}")
    
    # Calculate accuracy on the train set
    train_predictions = nn.predict(X_train)
    train_accuracy = np.mean(train_predictions.flatten() == y_train)
    
    # Calculate accuracy on the test set
    test_predictions = nn.predict(X_test)
    test_accuracy = np.mean(test_predictions.flatten() == y_test)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    
    print(f"Learning Rate {lr}: Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

    # Plot train/test loss over epochs
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Learning Rate: {lr}')
    plt.legend()
    plt.show()
# Plot final training and test loss for each learning rate
plt.figure()
plt.plot(learning_rates, final_train_losses, marker='o', label='Final Train Loss')
plt.plot(learning_rates, final_test_losses, marker='o', label='Final Test Loss')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Final Train and Test Loss vs Learning Rate')
plt.xscale('log')
plt.legend()
plt.show()

# Plot train and test accuracy for each learning rate
plt.figure()
plt.plot(learning_rates, train_accuracies, marker='o', label='Train Accuracy')
plt.plot(learning_rates, test_accuracies, marker='o', label='Test Accuracy')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy vs Learning Rate')
plt.xscale('log')
plt.legend()
plt.show()

# Comment on results
print("Part B2.1 Comments on results:")
print("1. Learning Rate 0.0001: The model shows stable and consistent learning with slow but steady convergence.")
print("2. Learning Rate 0.001: The model converges faster but exhibits instability due to slight overshooting.")
print("3. Learning Rate 0.01: The model fails to converge, showing flat loss curves and poor learning due to an excessively high learning rate.")
print("4. Learning Rate 0.1: The model diverges with high fluctuations in loss, indicating severe instability and ineffective learning.")
print("5. Learning Rate 1: The model exhibits extreme divergence with no signs of convergence, resulting in very poor performance.")
print("6. Final Train and Test Loss vs Learning Rate: Increasing the learning rate beyond 0.001 leads to a sharp increase in loss, highlighting instability and poor convergence at higher rates.")
print("7. Train and Test Accuracy vs Learning Rate: Accuracy drops drastically at learning rates above 0.001, indicating significant underfitting and divergence.")

# B.2.2 Hidden Layer Nodes Experiments

# Print the title for part B2.2
print("\nPart B2.2: Hidden Layer Nodes Experiments")

# Settings
learning_rate = 0.0001
momentum = 0
epochs = 250
hidden_nodes_list = [8, 16, 32, 64, 128, 256, 512]

final_train_losses_nodes = []
accuracies_nodes = []

for hidden_nodes in hidden_nodes_list:
    # Initialize the neural network with the current number of hidden nodes
    nn = NeuralNetwork(input_size, hidden_nodes, output_size, learning_rate, momentum)
    
    # Use the fit method for training
    train_losses, test_losses = nn.fit(X_train, y_train, X_test, y_test, epochs, print_every_200=False)
    
    # Plot train/test loss over epochs
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Hidden Nodes: {hidden_nodes}')
    plt.legend()
    plt.show()
    
    # Print final training loss
    final_train_loss = train_losses[-1]
    final_train_losses_nodes.append(final_train_loss)
    print(f"Final Training Loss for hidden nodes {hidden_nodes}: {final_train_loss}")
    
    # Calculate accuracy on the test set
    predictions = nn.predict(X_test)
    accuracy = np.mean(predictions.flatten() == y_test)
    accuracies_nodes.append(accuracy)
    print(f"Accuracy for hidden nodes {hidden_nodes}: {accuracy}")

# Plot final training loss for each number of hidden nodes
plt.figure()
plt.plot(hidden_nodes_list, final_train_losses_nodes, marker='o')
plt.xlabel('Number of Hidden Nodes')
plt.ylabel('Final Training Loss')
plt.title('Final Training Loss vs Number of Hidden Nodes')
plt.xscale('log')
plt.show()

# Plot accuracy for each number of hidden nodes
plt.figure()
plt.plot(hidden_nodes_list, accuracies_nodes, marker='o')
plt.xlabel('Number of Hidden Nodes')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Hidden Nodes')
plt.xscale('log')
plt.show()

# Comment on results
print("Part B2.2 Comments on results:")
print("1. Hidden Nodes: 8 - The model with 8 hidden nodes shows limited learning capacity, resulting in high loss and low accuracy.")
print("2. Hidden Nodes: 16 - Increasing to 16 hidden nodes improves accuracy significantly, but slight overfitting begins to appear.")
print("3. Hidden Nodes: 32 - With 32 hidden nodes, the model achieves high accuracy with minimal overfitting, indicating an optimal balance of capacity.")
print("4. Hidden Nodes: 64 - The model continues to reduce training loss but shows a widening gap between training and test loss, suggesting increasing overfitting.")
print("5. Hidden Nodes: 128 - The model shows clear signs of overfitting despite further reductions in training loss, with a slight decrease in accuracy.")
print("6. Hidden Nodes: 256 - Increasing to 256 nodes reduces both training and test loss effectively, maintaining high accuracy with manageable overfitting.")
print("7. Hidden Nodes: 512 - The model with 512 hidden nodes achieves the lowest losses but shows persistent overfitting without further gains in accuracy.")
print("8. Training Loss vs Hidden Nodes - Training loss consistently decreases as the number of hidden nodes increases, but with diminishing returns beyond a certain point.")
print("9. Accuracy vs Hidden Nodes - Accuracy increases sharply initially and stabilizes around 32 nodes, with little to no improvement from further increasing the hidden nodes.")


# B.2.3 Momentum Experiments
# Print the title for part B2.3
print("\nPart B2.3: Momentum Experiments")

# Settings
learning_rate = 0.001
hidden_nodes = 32
epochs = 250
momentum_values = [0, 0.1, 0.25, 0.5, 1]

# Initialize lists to store final losses and accuracies
final_train_losses_momentum = []
final_test_losses_momentum = []
accuracies_momentum = []

for momentum in momentum_values:
    # Initialize the neural network with the current momentum
    nn = NeuralNetwork(input_size, hidden_nodes, output_size, learning_rate, momentum)
    
    # Use the fit method for training with batch size
    train_losses, test_losses = nn.fit(X_train, y_train, X_test, y_test, epochs, print_every_200=False)
    
    # Plot train/test loss over epochs
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Momentum: {momentum}')
    plt.legend()
    plt.show()
    
    # Print final training and test loss
    final_train_loss = train_losses[-1]
    final_test_loss = test_losses[-1]
    final_train_losses_momentum.append(final_train_loss)
    final_test_losses_momentum.append(final_test_loss)
    print(f"Final Training Loss for momentum {momentum}: {final_train_loss}")
    print(f"Final Test Loss for momentum {momentum}: {final_test_loss}")
    
    # Calculate accuracy on the test set
    predictions = nn.predict(X_test)
    accuracy = np.mean(predictions.flatten() == y_test)
    accuracies_momentum.append(accuracy)
    print(f"Accuracy for momentum {momentum}: {accuracy}")

# Plot final training and test loss for each momentum
plt.figure()
plt.plot(momentum_values, final_train_losses_momentum, marker='o', label='Final Training Loss')
plt.plot(momentum_values, final_test_losses_momentum, marker='x', label='Final Test Loss')
plt.xlabel('Momentum')
plt.ylabel('Loss')
plt.title('Final Training and Test Loss vs Momentum')
plt.legend()
plt.show()

# Plot accuracy for each momentum
plt.figure()
plt.plot(momentum_values, accuracies_momentum, marker='o')
plt.xlabel('Momentum')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Momentum')
plt.show()

# Comment on results
print("Part B2.3 Comments on results:")
print("1. Graph 1 (Momentum = 0): Without momentum, the model shows decent performance but experiences noticeable fluctuations in loss values.")
print("2. Graph 2 (Momentum = 0.1): A small momentum helps reduce fluctuations slightly but leads to a minor increase in loss and a drop in accuracy.")
print("3. Graph 3 (Momentum = 0.25): Increasing the momentum further results in unstable training, higher losses, and a significant drop in accuracy.")
print("4. Graph 4 (Momentum = 0.5): The model exhibits plateauing behavior, indicating poor convergence and minimal learning progress.")
print("5. Graph 5 (Momentum = 1): The model fails completely with a momentum of 1, as both training and test losses remain high and constant throughout.")
print("6. Graph 6 (Final Loss vs. Momentum): Higher momentum values lead to consistently increasing losses, showing a negative impact on model performance.")
print("7. Graph 7 (Accuracy vs. Momentum): The accuracy decreases significantly when the momentum exceeds 0.1, indicating that higher momentum values lead to instability and poor convergence. This underscores the importance of carefully tuning the momentum, along with other hyperparameters, to maintain stable learning and achieve good generalization performance.")
print("Momentum hyperparameter is not functioning as expected, it is not able to improve the performance of the model, I suspect this is due to the fact that the other hyperparameters are not tuned properly, as we will see in the next part momentum of 0.8 gives the best results")


# part B2.4  Final Neural Network – Hyperparameter finetuning
# Load the dataset and split into train/test sets
from sklearn import datasets
from sklearn.model_selection import train_test_split

print("\nPart B2.4: Final Neural Network – Hyperparameter finetuning")
# Load the dataset
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

# Split data into train+val and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Further split X_temp into X_train and X_val
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/9, random_state=0)

# Network parameters
input_size = X_train.shape[1]
output_size = 1

# Define the hyperparameter grid
learning_rates = [0.00005, 0.0001]
hidden_nodes_list = [256, 512]
momentum_list = [0.8, 0.9]  # Added momentum as a tunable parameter
batch_sizes = [32, 64]  # Added batch size as a tunable parameter
epochs = 250

# Create all combinations of hyperparameters
param_grid = list(product(learning_rates, hidden_nodes_list, momentum_list, batch_sizes))

# Initialize a list to store results
results = []

# Loop over all combinations
for learning_rate, hidden_nodes, momentum, batch_size in param_grid:
    print(f"Training with learning_rate={learning_rate}, hidden_nodes={hidden_nodes}, momentum={momentum}, batch_size={batch_size}")
    
    # Initialize the neural network with current hyperparameters
    nn = NeuralNetwork(
        input_size=input_size,
        hidden_nodes=hidden_nodes,
        output_size=output_size,
        learning_rate=learning_rate,
        momentum=momentum
    )
    
    # Train the model
    train_losses, val_losses = nn.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        print_every_200=False,
        batch_size=batch_size
    )
    
    # Evaluate on validation set
    y_val_pred = nn.predict(X_val)
    val_accuracy = np.mean(y_val_pred.flatten() == y_val)
    print(f"Validation Accuracy: {val_accuracy}")
    
    # Store the results
    results.append({
        'learning_rate': learning_rate,
        'hidden_nodes': hidden_nodes,
        'momentum': momentum,
        'batch_size': batch_size,
        'val_accuracy': val_accuracy
    })

# Create a DataFrame from results
results_df = pd.DataFrame(results)

# Sort results by validation accuracy for better visualization
results_df = results_df.sort_values(by='val_accuracy', ascending=False)

# Create a label for each combination
results_df['combination'] = results_df.apply(
    lambda row: f"lr={row['learning_rate']}, hn={row['hidden_nodes']}, m={row['momentum']}, bs={row['batch_size']}",
    axis=1
)

# Plot line graph of all combinations and their accuracy
plt.figure(figsize=(12, 6))
plt.plot(results_df['combination'], results_df['val_accuracy'], marker='o')
plt.title("Validation Accuracy for Different Hyperparameter Combinations")
plt.xlabel("Hyperparameter Combination")
plt.ylabel("Validation Accuracy")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
plt.show()

# Select the best hyperparameter combination based on validation accuracy
best_result = results_df.loc[results_df['val_accuracy'].idxmax()]
print("Best hyperparameter combination based on validation accuracy:")
print(best_result)

# After finding the best hyperparameter combination
best_learning_rate = best_result['learning_rate']
best_hidden_nodes = best_result['hidden_nodes']
best_momentum = best_result['momentum']
best_batch_size = best_result['batch_size']

# Initialize the neural network with the best hyperparameters
best_nn = NeuralNetwork(
    input_size=input_size,
    hidden_nodes=best_hidden_nodes,
    output_size=output_size,
    learning_rate=best_learning_rate,
    momentum=best_momentum
)

# Train the model with the best hyperparameters
train_losses, test_losses = best_nn.fit(
    X_train, y_train,
    X_test, y_test,
    epochs=epochs,
    print_every_200=False,
    batch_size=best_batch_size
)

# Plot the average train/test loss during training
plt.figure(figsize=(12, 6))
plt.plot(range(epochs), train_losses, label='Train Loss')
plt.plot(range(epochs), test_losses, label='Test Loss')
plt.title("Average Train/Test Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print the final training loss
final_train_loss = train_losses[-1]
print(f"Final Training Loss: {final_train_loss}")

# Evaluate on the test set
y_test_pred = best_nn.predict(X_test)
test_accuracy = np.mean(y_test_pred.flatten() == y_test)
print(f"Test Accuracy: {test_accuracy}")