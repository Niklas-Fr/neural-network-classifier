import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# loading the two moons train and test set
train_data = dict(np.load("two_moons.npz", allow_pickle=True))
test_data = dict(np.load("two_moons_test.npz", allow_pickle=True))

train_samples = train_data["samples"]
train_labels = train_data["labels"][:, None] # expand to N x K=2 for broadcasting
test_samples = test_data["samples"]
test_labels = test_data["labels"][:, None]

# auxiliary functions

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def d_relu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))

def binary_cross_entropy(predictions: np.ndarray, labels : np.ndarray) -> float:
    return -np.mean(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))

def d_binary_cross_entropy(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return - (1 / labels.shape[0]) * ((labels / predictions) - ((1 - labels) / (1 - predictions)))
    

def init_weights(neurons_per_hidden_layer: np.ndarray,
                 input_dimension: int, output_dimension: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    
    current_input_dimension = input_dimension
    weights = []
    biases = []

    for current_output_dimension in neurons_per_hidden_layer + [output_dimension]:
        
        bias = np.zeros((1, current_output_dimension))
        
        sigma = 2 / np.sqrt(current_input_dimension) # xavier activation function relu corrected
        weight = np.random.normal(0, sigma, size=(current_input_dimension, current_output_dimension)) # normal distributed weights
        
        biases.append(bias)
        weights.append(weight)
        
        current_input_dimension = current_output_dimension
        
    return weights, biases


def feed_forward(weights: List[np.ndarray], biases: List[np.ndarray], input: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    zs = [] # preactivations
    hs = [] # hidden layer output

    h = input
    hs.append(h)
    for i in range(len(weights) - 1):
        z = h @ weights[i] + biases[i]
        zs.append(z)
        h = relu(z)
        hs.append(h)
    
    # output layer
    
    z = h @ weights[-1] + biases[-1]   
    zs.append(z)
    y = sigmoid(z)
    
    return y, zs, hs


def backward_pass(grad_loss: np.ndarray, weights: List[np.ndarray], biases: List[np.ndarray],
                  zs: np.ndarray, hs: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
       
    d_weights = [None] * len(weights)
    d_biases = [None] * len(biases)
    
    grad = grad_loss * d_sigmoid(zs[-1])
    #print(f"grad {grad_loss.shape}")
    #print(f"sigmoid {d_sigmoid(zs[-1]).shape}")
    #print(f"z {zs[-1].shape}")
    #print(f"start grad {grad.shape}")
    
    for i in reversed(range(len(weights))):
        d_weights[i] = np.mean(np.expand_dims(hs[i], axis=2) * np.expand_dims(grad, axis=1), axis=0)
        d_biases[i] = np.mean(grad, axis=0)
        
        grad = (grad @ weights[i].T) * d_relu(zs[i - 1])
        
    return d_weights, d_biases
        
#TODO: introduce batch gradient descent      
def train_neural_network(train_samples: np.ndarray,
                         train_labels: np.ndarray,
                         test_samples: np.ndarray,
                         test_labels: np.ndarray,
                         layers: List[int], 
                         num_iterations: int, 
                         learning_rate: float,
                         momentum_constant: float) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    
    train_losses = []
    test_losses = []
    
    batch_size = 1
    batches_per_iteration = int(train_samples.shape[0] / batch_size)
    
    weights, biases = init_weights(layers, input_dimension=train_samples.shape[-1], output_dimension=train_labels.shape[-1])
    
    momentum_weights = [np.zeros(w.shape) for w in weights]
    momentum_biases = [np.zeros(b.shape) for b in biases]
    
    for i in range(num_iterations):
        rand_index = np.random.permutation(train_samples.shape[0])
        for j in range(batches_per_iteration):
            sample_batch = train_samples[rand_index[j * batch_size : (j + 1) * batch_size]]
            label_batch = train_labels[rand_index[j * batch_size : (j + 1) * batch_size]]

            prediction, zs, hs = feed_forward(weights, biases, sample_batch)
            train_losses.append(binary_cross_entropy(prediction, label_batch))
        
            grad_loss = d_binary_cross_entropy(prediction, label_batch)
        
            d_weights, d_biases = backward_pass(grad_loss, weights, biases, zs, hs)
        
            for k in range(len(d_weights)):
                momentum_weights[k] = momentum_constant * momentum_weights[k] + (1 - momentum_constant) * d_weights[k]
                momentum_biases[k] = momentum_constant * momentum_biases[k] + (1 - momentum_constant) * d_biases[k]
            
                weights[k] -= learning_rate * momentum_weights[k]
                biases[k] -= learning_rate * momentum_biases[k]
            
            # TODO
            test_losses.append(binary_cross_entropy(feed_forward(weights, biases, test_samples)[0], test_labels))
            
            
    return weights, biases, train_losses, test_losses
        
        

num_iterations = 50

weights, biases, train_losses, test_losses = train_neural_network(train_samples,
                                                              train_labels,
                                                              test_samples,
                                                              test_labels,
                                                              layers=[64, 64],
                                                              num_iterations=num_iterations,
                                                              learning_rate=5e-2,
                                                              momentum_constant=0.99)



print(f"Train Loss {np.round(train_losses[-1], 2)}")
print(f"Test Loss {np.round(test_losses[-1], 2)}")


plt.title("Loss")
plt.plot(train_losses)
plt.plot(test_losses)
plt.grid(visible=True)
plt.xlabel("Gradient steps")
plt.ylabel("Binary Cross Entropy Loss")
plt.legend(["Test Loss", "Train Loss"])
plt.show()

# TODO: plot