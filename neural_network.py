import numpy as np
import pickle
from typing import List, Tuple

class NeuralNetwork:
    def __init__(self, input_size: int = 11, hidden_sizes: List[int] = [16, 12], output_size: int = 3):
        """
        Initialize neural network
        input_size: Size of input layer (game state features)
        hidden_sizes: List of hidden layer sizes
        output_size: Size of output layer (actions: straight, right, left)
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Create layer sizes list
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            # Xavier initialization
            weight_matrix = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(2.0 / self.layer_sizes[i])
            bias_vector = np.zeros((1, self.layer_sizes[i + 1]))
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        
        activation = inputs
        
        # Pass through all layers
        for i in range(len(self.weights)):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            
            # Use ReLU for hidden layers, softmax for output
            if i < len(self.weights) - 1:
                activation = self.relu(z)
            else:
                activation = self.softmax(z)
        
        return activation
    
    def predict(self, inputs: np.ndarray) -> int:
        """Predict action given game state"""
        output = self.forward(inputs)
        return np.argmax(output[0])
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def get_weights_biases(self) -> np.ndarray:
        """Get all weights and biases as a flat array for genetic algorithm"""
        params = []
        
        for weight_matrix in self.weights:
            params.extend(weight_matrix.flatten())
        
        for bias_vector in self.biases:
            params.extend(bias_vector.flatten())
        
        return np.array(params)
    
    def set_weights_biases(self, params: np.ndarray):
        """Set weights and biases from a flat array"""
        param_idx = 0
        
        # Set weights
        for i, weight_matrix in enumerate(self.weights):
            weight_size = weight_matrix.size
            new_weights = params[param_idx:param_idx + weight_size].reshape(weight_matrix.shape)
            self.weights[i] = new_weights
            param_idx += weight_size
        
        # Set biases
        for i, bias_vector in enumerate(self.biases):
            bias_size = bias_vector.size
            new_biases = params[param_idx:param_idx + bias_size].reshape(bias_vector.shape)
            self.biases[i] = new_biases
            param_idx += bias_size
    
    def get_total_params(self) -> int:
        """Get total number of parameters in the network"""
        total = 0
        for weight_matrix in self.weights:
            total += weight_matrix.size
        for bias_vector in self.biases:
            total += bias_vector.size
        return total
    
    def copy(self) -> 'NeuralNetwork':
        """Create a copy of this neural network"""
        new_nn = NeuralNetwork(self.input_size, self.hidden_sizes, self.output_size)
        new_nn.set_weights_biases(self.get_weights_biases())
        return new_nn
    
    def save(self, filename: str):
        """Save the neural network to a file"""
        data = {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'weights': self.weights,
            'biases': self.biases
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filename: str) -> 'NeuralNetwork':
        """Load a neural network from a file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        nn = cls(data['input_size'], data['hidden_sizes'], data['output_size'])
        nn.weights = data['weights']
        nn.biases = data['biases']
        
        return nn

class SnakeAI:
    def __init__(self, neural_network: NeuralNetwork):
        """Snake AI that uses a neural network to make decisions"""
        self.neural_network = neural_network
        self.fitness = 0
        self.games_played = 0
        self.total_score = 0
        self.total_steps = 0
    
    def get_action(self, game_state: np.ndarray) -> int:
        """Get action from neural network given game state"""
        return self.neural_network.predict(game_state)
    
    def calculate_fitness(self, score: int, steps: int) -> float:
        """Calculate fitness based on game performance"""
        # Fitness function that rewards both survival time and eating food
        # Higher score is more important than just surviving longer
        fitness = score * 100 + steps * 0.1
        
        # Bonus for higher scores (exponential reward for eating more food)
        if score > 0:
            fitness += score ** 2 * 10
        
        return fitness
    
    def update_fitness(self, score: int, steps: int):
        """Update fitness statistics"""
        self.games_played += 1
        self.total_score += score
        self.total_steps += steps
        
        game_fitness = self.calculate_fitness(score, steps)
        
        # Use average fitness across games
        self.fitness = ((self.fitness * (self.games_played - 1)) + game_fitness) / self.games_played
    
    def reset_fitness(self):
        """Reset fitness statistics"""
        self.fitness = 0
        self.games_played = 0
        self.total_score = 0
        self.total_steps = 0
    
    def copy(self) -> 'SnakeAI':
        """Create a copy of this AI"""
        new_ai = SnakeAI(self.neural_network.copy())
        new_ai.fitness = self.fitness
        return new_ai

if __name__ == "__main__":
    # Test the neural network
    nn = NeuralNetwork()
    print(f"Neural Network created with {nn.get_total_params()} parameters")
    
    # Test forward pass
    test_input = np.random.rand(20)
    output = nn.forward(test_input)
    print(f"Test input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    print(f"Predicted action: {nn.predict(test_input)}")
    
    # Test AI
    ai = SnakeAI(nn)
    action = ai.get_action(test_input)
    print(f"AI predicted action: {action}")
    
    # Test fitness calculation
    ai.update_fitness(5, 100)
    print(f"Fitness after game: {ai.fitness}")
