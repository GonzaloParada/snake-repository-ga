#!/usr/bin/env python3
"""
PyTorch Neural Network for Snake AI
Replaces the NumPy implementation with PyTorch for better performance and features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from typing import List, Tuple

class SnakeNet(nn.Module):
    """PyTorch Neural Network for Snake AI"""
    
    def __init__(self, input_size: int = 11, hidden_sizes: List[int] = [16, 12], output_size: int = 3):
        super(SnakeNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        # Remove softmax to match NumPy behavior - let argmax work on raw outputs
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through the network"""
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        # Ensure input is 2D (batch_size, features)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        return self.network(x)
    
    def get_weights(self):
        """Get all network weights as a flat numpy array"""
        weights = []
        for param in self.parameters():
            weights.extend(param.data.flatten().numpy())
        return np.array(weights)
    
    def set_weights(self, weights):
        """Set network weights from a flat numpy array"""
        weights = torch.FloatTensor(weights)
        idx = 0
        
        for param in self.parameters():
            param_size = param.numel()
            param.data = weights[idx:idx + param_size].view(param.shape)
            idx += param_size
    
    def get_total_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


class SnakeAI:
    """PyTorch-based Snake AI with genetic algorithm compatibility"""
    
    def __init__(self, input_size: int = 11, hidden_sizes: List[int] = [16, 12], output_size: int = 3):
        self.network = SnakeNet(input_size, hidden_sizes, output_size)
        self.fitness = 0.0
        self.games_played = 0
        self.total_score = 0
        self.total_steps = 0
        
        # Set to evaluation mode (no training)
        self.network.eval()
    
    def get_action(self, state: np.ndarray) -> int:
        """Get action from neural network"""
        with torch.no_grad():  # No gradient computation needed
            output = self.network(state)
            
            # Convert to numpy and get action
            if isinstance(output, torch.Tensor):
                output = output.numpy()
            
            # Handle batch dimension
            if output.ndim > 1:
                output = output[0]
            
            return int(np.argmax(output))
    
    def get_weights(self) -> np.ndarray:
        """Get network weights for genetic algorithm"""
        return self.network.get_weights()
    
    def set_weights(self, weights: np.ndarray):
        """Set network weights from genetic algorithm"""
        self.network.set_weights(weights)
    
    def get_total_params(self) -> int:
        """Get total number of parameters"""
        return self.network.get_total_params()
    
    def calculate_fitness(self, score: int, steps: int) -> float:
        """Calculate fitness based on game performance"""
        # Same fitness function as before
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
        
        # Update running average fitness
        if self.games_played == 1:
            self.fitness = game_fitness
        else:
            # Weighted average favoring recent performance
            self.fitness = (self.fitness * 0.7) + (game_fitness * 0.3)
    
    def reset_stats(self):
        """Reset fitness statistics"""
        self.fitness = 0.0
        self.games_played = 0
        self.total_score = 0
        self.total_steps = 0
    
    def get_average_score(self) -> float:
        """Get average score across all games"""
        return self.total_score / max(self.games_played, 1)
    
    def get_average_steps(self) -> float:
        """Get average steps across all games"""
        return self.total_steps / max(self.games_played, 1)
    
    def save(self, filepath: str):
        """Save the neural network model"""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'input_size': self.network.input_size,
            'hidden_sizes': self.network.hidden_sizes,
            'output_size': self.network.output_size,
            'fitness': self.fitness,
            'games_played': self.games_played,
            'total_score': self.total_score,
            'total_steps': self.total_steps
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str):
        """Load a neural network model"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Create new instance
        ai = cls(
            input_size=checkpoint['input_size'],
            hidden_sizes=checkpoint['hidden_sizes'],
            output_size=checkpoint['output_size']
        )
        
        # Load model state
        ai.network.load_state_dict(checkpoint['model_state_dict'])
        ai.fitness = checkpoint.get('fitness', 0.0)
        ai.games_played = checkpoint.get('games_played', 0)
        ai.total_score = checkpoint.get('total_score', 0)
        ai.total_steps = checkpoint.get('total_steps', 0)
        
        return ai


def test_pytorch_network():
    """Test the PyTorch neural network"""
    print("Testing PyTorch Neural Network...")
    
    # Create network
    ai = SnakeAI()
    print(f"Network created with {ai.get_total_params()} parameters")
    
    # Test forward pass
    test_input = np.random.rand(11)
    action = ai.get_action(test_input)
    print(f"Test input shape: {test_input.shape}")
    print(f"Action: {action}")
    
    # Test weight manipulation
    original_weights = ai.get_weights()
    print(f"Total weights: {len(original_weights)}")
    
    # Modify weights
    new_weights = original_weights + np.random.normal(0, 0.1, original_weights.shape)
    ai.set_weights(new_weights)
    
    # Test again
    action2 = ai.get_action(test_input)
    print(f"Action after weight change: {action2}")
    
    # Test fitness
    ai.update_fitness(5, 100)
    print(f"Fitness after game: {ai.fitness}")
    
    print(" PyTorch network test completed!")


if __name__ == "__main__":
    test_pytorch_network()
