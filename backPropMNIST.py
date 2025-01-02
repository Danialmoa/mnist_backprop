####################################
# Backpropation experiments on MNIST 
#   Based on the Python script on Backpropagation that for experiments MNIST character recognition, extend it properly to address the following requirements:
#       1. Improve the script by a modular organization and save the weights to a file; -> Done
#       2. Design a module for carrying out the test on the given MNIST csv data on the basis of
#           saved weight files;
#       3. Discuss appropriate choices of the mini-batches -> Done
#       4. Discuss the initialization of the weights and the network architecture; -> Done
#       5. Discuss the role of the stopping criterion
#       6. Design a module for plotting the results

# Author: Danial Moafi
# Date: 2024-11-25
####################################

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from image_augmentation import ImageAugmentation


class DataLoader:
    def __init__(self, train_path, test_path):
        """
        Initialize the data loader
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = self.load_data(self.train_path)
        self.test_data = self.load_data(self.test_path)
        
    def __len__(self):
        return len(self.train_data)

    def load_data(self, path):
        """
        Load the data from the csv file
        """
        return pd.read_csv(path)
    
    @property
    def get_train_data(self):
        train_labels = self.train_data['5'].to_numpy()
        train_data = self.train_data.drop(columns=['5']).to_numpy()
        return train_data, train_labels
    
    @property
    def get_test_data(self):
        test_labels = self.test_data['7'].to_numpy()
        test_data = self.test_data.drop(columns=['7']).to_numpy()
        return test_data, test_labels
    
    
class StopCriterion:
    def __init__(self, criteria, patience=5):
        self.criteria = criteria
        self.max_epochs = None
        self.loss_threshold = None
        self.patience = patience
    
        # Early Stopping
        self.best_loss = float('inf')
        self.waited_epochs = 0
        self.best_weights = None
        
        # Loss Tracking
        self.train_loss = []
        self.val_loss = []
        self.loss_window = 5
    
    def set_max_epochs(self, max_epochs):
        """
        Set the maximum number of epochs
        """
        self.max_epochs = max_epochs
        

    def check_early_stopping(self, val_loss, weights):
        """
        Check if the early stopping criterion is met
        Check if the validation loss is less than the best loss
        And it has waited for more than the patience epochs
        """
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.waited_epochs = 0
            self.best_weights = weights
        else:
            self.waited_epochs += 1
            if self.waited_epochs >= self.patience:
                return True, 'Early stopping'
        return False
    
    def check_loss_plateau(self, train_loss):
        """
        Check if the loss plateau criterion is met
        """
        if len(self.train_loss) < self.loss_window:
            return False
        
        recent_losses = self.train_loss[-self.loss_window:]
        mean_loss = np.mean(recent_losses)
        std_loss = np.std(recent_losses)
        
        return std_loss < (mean_loss * 0.001)
        
    def check_max_epochs(self, epochs):
        return epochs >= self.max_epochs

        
    def __call__(self, current_epoch, train_loss, val_loss, weights):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        
        for criterion in self.criteria:
            if criterion == 'max_epochs' and self.check_max_epochs(current_epoch):
                return True, 'Maximum epochs reached', weights
            elif criterion == 'early_stopping' and self.check_early_stopping(val_loss, weights):
                return True, 'Early stopping', self.best_weights
            elif criterion == 'loss_plateau' and self.check_loss_plateau(train_loss):
                return True, 'Loss plateau', weights

        return False, None, weights
    
    
class ActivationFunction:
    def __init__(self, function):
        """
        Initialize the activation function
        """
        self.function = function
        if not hasattr(self, function):
            raise ValueError(f"Function {function} not found in {self.__class__.__name__}")
        
    def __call__(self, x):
        return getattr(self, self.function)(x)
        
    def relu(self, x):
        """
        ReLU activation function
        """
        return np.maximum(0, x)
    
    def softmax(self, x):
        """
        Softmax activation function
            I've used the stable version of the softmax function to avoid numerical instability
            based on this article https://dl.acm.org/doi/pdf/10.1145/3510003.3510095
        """
        x_stable = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def derivative_relu(self, x):
        """
        Derivative of the ReLU activation function
        """
        return np.where(x > 0, 1, 0)
    

class WeightsBiases:
    def __init__(self):
        """
        Initialize the weights and biases
        """
        self.weights = None
        self.biases = None
        
    def __str__(self):
        """
        Print the shape of the weights and biases
        """
        weights_shapes = [w.shape for w in self.weights]
        biases_shapes = [b.shape for b in self.biases]
        return f"Weights shapes: {weights_shapes}\nBiases shapes: {biases_shapes}"
    
    def bias_initialize(self, network):
        """
        Bias initialization
        """
        self.biases = [np.random.randn(network.hidden_size[0]),
                       *[np.random.randn(network.hidden_size[i+1]) for i in range(network.num_hidden_layers-1)],
                       np.random.randn(network.output_size)]
    
    def wieght_zero_initialize(self, network):
        """
        Weight zero initialization
        """
        self.weights = [np.zeros((network.input_size, network.hidden_size[0])),
                        *[np.zeros((network.hidden_size[i], network.hidden_size[i+1])) for i in range(network.num_hidden_layers-1)],
                        np.zeros((network.hidden_size[-1], network.output_size))]
        
    def weights_initialize_random(self, network):
        """
        Random initalization of weights
        """
        weights_scale = 1
        self.weights = [
                np.random.randn(network.input_size, network.hidden_size[0]) * weights_scale,
                *[np.random.randn(network.hidden_size[i], network.hidden_size[i+1]) * weights_scale
                  for i in range(network.num_hidden_layers-1)],
                np.random.randn(network.hidden_size[-1], network.output_size) * weights_scale
            ]
        
    def he_initialize(self, network):
        """
        He initialization of weights and biases
        """
        weights_scale = np.sqrt(2/network.input_size)
        self.weights = [
                np.random.randn(network.input_size, network.hidden_size[0]) * weights_scale,
                *[np.random.randn(network.hidden_size[i], network.hidden_size[i+1]) * weights_scale
                  for i in range(network.num_hidden_layers-1)],
                np.random.randn(network.hidden_size[-1], network.output_size) * weights_scale
            ]
        
    def xavier_initialize(self, network):
        """
        Xavier initialization of weights and biases
        """
        weights_scale = 2/np.sqrt(network.input_size + network.output_size)
        self.weights = [
                np.random.randn(network.input_size, network.hidden_size[0]) * weights_scale,
                *[np.random.randn(network.hidden_size[i], network.hidden_size[i+1]) * weights_scale
                  for i in range(network.num_hidden_layers-1)],
                np.random.randn(network.hidden_size[-1], network.output_size) * weights_scale
            ]

        
    def save(self, path):
        """
        Save the weights and biases to a file
        """
        os.makedirs(path, exist_ok=True)
        np.savez(path + '/weights.npz', *self.weights)
        np.savez(path + '/biases.npz', *self.biases)
        
    def load(self, path):
        """
        Load the weights and biases from a file
        """
        weights_dict = np.load(path + '/weights.npz')
        biases_dict = np.load(path + '/biases.npz')
        
        self.weights = [weights_dict[f'arr_{i}'] for i in range(len(weights_dict.files))]
        self.biases = [biases_dict[f'arr_{i}'] for i in range(len(biases_dict.files))]
        
        print("Weights and biases loaded")
    

class NeuralNetwork:
    def __init__(self, 
                learning_rate=None, 
                epochs=None, 
                batch_size=None, 
                input_size=784, 
                hidden_size=None, 
                output_size=10, 
                weights_method='xavier', 
                model_name=None,
                visualization=False,
                path=None,
                augmentation=False,
                remove_pixels=False):
        """
        Initialize the neural network
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        if self.hidden_size:
            self.num_hidden_layers = len(self.hidden_size)
        self.output_size = output_size
        self.activation_function = ActivationFunction('relu')
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights_biases = WeightsBiases()
        self.weights_method = weights_method
        self.model_name = model_name
        self.path = path
        self.augmentation = augmentation
        self.remove_pixels = remove_pixels
        
        if self.path is None:
            self.check_run_number()
        else:
            self.initialize_weights_biases(self.path)
        
        # Initialize the visualization
        self.visualization = visualization
        
        # Stopping criterion
        self.stop_criterion = StopCriterion(
            criteria=['max_epochs', 'early_stopping', 'loss_plateau'],
            patience=5,
        )
        self.stop_criterion.set_max_epochs(self.epochs)
        self.augmentation = augmentation
    
    def check_run_number(self):
        if os.path.exists(f"models/{self.model_name}"):
            run_numbers = [int(f.split('_')[1]) for f in os.listdir(f"models/{self.model_name}")]
            if len(run_numbers) == 0:
                self.run_number = 1
            else:
                self.run_number = max(run_numbers) + 1
        else:
            os.makedirs(f"models/{self.model_name}", exist_ok=True)
            self.run_number = 1
        
    def initialize_weights_biases(self, path=None):
        """
        Initialize the weights and biases
        """
        if path:
            self.weights_biases.load(path=path)
        else:
            self.weights_biases.bias_initialize(self)
            if self.weights_method == 'random':
                self.weights_biases.weights_initialize_random(self)
            elif self.weights_method == 'zero':
                self.weights_biases.wieght_zero_initialize(self)
            elif self.weights_method == 'he':
                self.weights_biases.he_initialize(self)
            elif self.weights_method == 'xavier':
                self.weights_biases.xavier_initialize(self)
            else:
                raise ValueError(f"Method {self.weights_method} not found")
            
        self.weights = self.weights_biases.weights
        self.biases = self.weights_biases.biases

        
            
    def normalize_data(self, X):
        """
        Normalize the data
        """
        return X / 255.0
    
    def shuffle_data(self, X, y):
        """
        Shuffle the data
        """
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        return X[indices], y[indices]
    
    def forward_propagation(self, X):
        """
        Forward propagation
        """
        cache = {
            'A': [X],
            'Z' : []
        }
        current_input = X
        for i in range(len(self.weights) - 1):
            Z = np.dot(current_input, self.weights[i]) + self.biases[i]
            cache['Z'].append(Z)
            
            # Apply activation function
            A = self.activation_function(Z)
            cache['A'].append(A)
            current_input = A

        # Output layer (using softmax)
        Z_out = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        cache['Z'].append(Z_out)

        # Softmax activation for output layer
        A_out = self.activation_function.softmax(Z_out)
        cache['A'].append(A_out)
        return A_out, cache
    
    def loss(self, y_true, y_pred):
        """
        Loss function
        """
        # y_true is a one-hot encoded vector
        # y_pred is a softmax output
        # cross-entropy loss = - sum(y_true * log(y_pred))
        return -np.sum(y_true * np.log(np.clip(y_pred, 1e-12, 1 - 1e-12))) / y_true.shape[0]
    
    def back_propagation(self, X, y):
        """
        Back propagation
        """
        # X is the input data
        # y is the true labels
        # return the gradients of the weights and biases
        
        total_gradients = None

        for i in range(0, X.shape[0], self.batch_size):
            batch_X = X[i:i + self.batch_size]
            batch_y = y[i:i + self.batch_size]

            A_out, cache = self.forward_propagation(batch_X)
            m = batch_X.shape[0]
            
            batch_gradients = {
                'dZ': [],
                'dW': [],
                'db': []
            }   
            for j in range(len(self.weights) - 1, -1, -1):
                if j == len(self.weights) - 1:
                    # For the output layer
                    # Eq 1 
                    dZ = A_out - batch_y # Softmax derivative in cross-entropy loss and softmax -> ai - yi 
                    dW = np.dot(cache['A'][j].T, dZ) / m # cache['A'][j] = aj is the output of the previous layer
                    db = np.sum(dZ, axis=0) / m
                else:
                    # For the hidden layers
                    # Eq 2
                    dZ = np.dot(dZ, self.weights[j+1].T) * self.activation_function.derivative_relu(cache['Z'][j]) # dz(l) = dz(l+1) * w(l+1) * f'(z(l))
                    dW = np.dot(cache['A'][j].T, dZ) / m # dw(l) = a(l) * dz(l)
                    db = np.sum(dZ, axis=0) / m

                
                batch_gradients['dZ'] = dZ
                batch_gradients['dW'] = dW
                batch_gradients['db'] = db
                
                # Update weights and biases
                self.weights_biases.weights[j] -= self.learning_rate * batch_gradients['dW']
                self.weights_biases.biases[j] -= self.learning_rate * batch_gradients['db']
                
                
        if total_gradients is None:
            total_gradients = batch_gradients
        else:
            for key in total_gradients:
                total_gradients[key] += batch_gradients[key]
        return total_gradients
    
    def train(self, X, y):
        y = np.eye(self.output_size)[y]
        X = self.normalize_data(X)
        
        len_data = X.shape[0]
        valideation_ratio = 0.1
        X_val = X[:int(len_data * valideation_ratio)]
        y_val = y[:int(len_data * valideation_ratio)]
        X = X[int(len_data * valideation_ratio):]
        y = y[int(len_data * valideation_ratio):]

        if self.augmentation:
            for _ in range(2):
                # Augment the data  
                X_augmented, y_augmented = ImageAugmentation(X, y).augment()
                X = np.concatenate([X, X_augmented])
                y = np.concatenate([y, y_augmented])
            
            if self.remove_pixels:
                for _ in range(2):
                    X_augmented, y_augmented = ImageAugmentation(X, y).remove_pixels_square(0.05)
                    print(X_augmented.shape)
                    #X_augmented, y_augmented = ImageAugmentation(X_augmented, y_augmented).remove_pixels_randomly(0.1)
                    X = np.concatenate([X, X_augmented])
                    y = np.concatenate([y, y_augmented])
                
            X, y = self.shuffle_data(X, y)
            print('new data shape', X.shape, y.shape)
            
        for epoch in range(self.epochs):
            # Shuffle the data
            X, y = self.shuffle_data(X, y)
            
            train_loss = 0
            self.back_propagation(X, y)
            predictions, _ = self.forward_propagation(X)
            
            train_loss = self.loss(y, predictions)
            accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
            
            # validation
            predictions_val, _ = self.forward_propagation(X_val)
            accuracy_val = np.mean(np.argmax(predictions_val, axis=1) == np.argmax(y_val, axis=1))
            val_loss = self.loss(y_val, predictions_val)
            
            # check stopping criterion
            current_weights = self.weights_biases.weights.copy()
            stop, reason, best_weights = self.stop_criterion(epoch, train_loss, val_loss, current_weights)
            if stop:
                print(f"Stopping criterion reached: {reason} at epoch {epoch+1}")
                self.weights_biases.weights = best_weights
                self.weights_biases.save(f"models/{self.model_name}/run_{self.run_number}")
                break
            
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                self.weights_biases.save(f"models/{self.model_name}/run_{self.run_number}")
            
            # Update the visualization
            if self.visualization:
                self.visualization.update(self.model_name, train_loss, accuracy)
                self.visualization.update(self.model_name + '_val', val_loss, accuracy_val)
            
            print(f"Epoch {epoch+1}/{self.epochs} - "
                  f"Train loss: {train_loss:.4f} - "
                  f"Train accuracy: {accuracy:.4f} - "
                  f"Val loss: {val_loss:.4f} - "
                  f"Val accuracy: {accuracy_val:.4f}")
           
            print('--'*10)

    
    def test(self, X, y):
        X = self.normalize_data(X)
        predictions, _ = self.forward_propagation(X)
        return predictions
    
    def save_weights_biases(self):
        self.weights_biases.save(self.path)
        
    def accuracy(self, y_true, y_pred):
        y_pred = np.argmax(y_pred, axis=1)
        return np.mean(y_true == y_pred)
    
    
class Visualization:
    def __init__(self):
        self.history = {} # For storing the history of the training
        
        self.losses = []
        self.accuracies = []
        
        # Setup the plot
        plt.ion()  # Turn on interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.fig.suptitle('Training Metrics')
        

        # Setup axes
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training Loss')
        
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.set_title('Training Accuracy')
        
        plt.show()
    
    def add_model(self, model_name):
        self.history[model_name] = {
            'losses': [],
            'accuracies': [],
            'line_loss': None,
            'line_acc': None
        }

    def update(self, model_name, loss, accuracy):
        if model_name not in self.history:
            self.add_model(model_name)
        
        self.history[model_name]['losses'].append(loss)
        self.history[model_name]['accuracies'].append(accuracy)
        # Update lines
        x = list(range(len(self.history[model_name]['losses'])))
        if self.history[model_name]['line_loss'] is None:
            # Create new lines with random color
            color = np.random.rand(3,)
            line_loss, = self.ax1.plot(x, self.history[model_name]['losses'], 
                                     label=model_name, color=color)
            line_acc, = self.ax2.plot(x, self.history[model_name]['accuracies'], 
                                    label=model_name, color=color)
            
            self.history[model_name]['line_loss'] = line_loss
            self.history[model_name]['line_acc'] = line_acc
            
            # Update legends
            self.ax1.legend()
            self.ax2.legend()
        else:
            # Update existing lines
            self.history[model_name]['line_loss'].set_data(x, self.history[model_name]['losses'])
            self.history[model_name]['line_acc'].set_data(x, self.history[model_name]['accuracies'])
        
        # Adjust axes limits
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # Draw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        
    def save(self, filename):
        plt.savefig(filename)
        
    def save_history(self, filename):
        """Save the metrics history to a file"""
        history_data = {
            model: {
                'losses': self.history[model]['losses'],
                'accuracies': self.history[model]['accuracies']
            }
            for model in self.history
        }
        np.save(filename, history_data)

    def load_history(self, filename):
        """Load metrics history from a file"""
        history_data = np.load(filename, allow_pickle=True).item()
        for model_name, data in history_data.items():
            self.add_model(model_name)
            for loss, acc in zip(data['losses'], data['accuracies']):
                self.update(model_name, None, loss, acc)


if __name__ == "__main__":
    data_train_path = "datasets/mnist/mnist_train.csv"
    data_test_path = "datasets/mnist/mnist_test.csv"
    
    data = DataLoader(data_train_path, data_test_path)
    train_data, train_labels = data.get_train_data
    test_data, test_labels = data.get_test_data
    

    run_number = 1
    epochs = 200
    hidden_size = [100, 100]
    model_name = 'two_layer_100_100_augmentation_remove_pixels'
    learning_rate = 0.001
    visualization = Visualization()
    
    batch_size = 8
    network = NeuralNetwork(
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        model_name=model_name,
        visualization=visualization,
        augmentation=True,
        remove_pixels=True
        )
    network.initialize_weights_biases()
    network.train(train_data, train_labels)
    
    predictions = network.test(test_data, test_labels)
    accuracy = network.accuracy(test_labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")
        
    visualization.save(f'models/{model_name}/run_{run_number}/loss_graph.png')



