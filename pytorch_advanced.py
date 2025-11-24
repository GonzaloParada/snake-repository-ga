#!/usr/bin/env python3
"""
Advanced PyTorch Neural Network for Snake AI
Designed to SURPASS NumPy performance with modern techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from typing import List, Tuple

class AdvancedSnakeNet(nn.Module):
    """Advanced PyTorch Neural Network with modern improvements"""
    
    def __init__(self, input_size: int = 11, hidden_sizes: List[int] = [32, 24, 16], output_size: int = 3):
        super(AdvancedSnakeNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Build network with modern improvements
        layers = []
        prev_size = input_size
        
        # Input normalization layer
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Hidden layers with improvements
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # Batch normalization
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Light dropout for regularization
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Advanced weight initialization
        self.apply(self._advanced_init_weights)
    
    def _advanced_init_weights(self, module):
        """Advanced weight initialization for better performance"""
        if isinstance(module, nn.Linear):
            # He initialization for ReLU networks
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(module.bias, 0.01)  # Small positive bias
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass with normalization"""
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        # Ensure input is 2D (batch_size, features)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Input normalization (only if batch size > 1)
        if x.size(0) > 1:
            x = self.input_norm(x)
        
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


class AdvancedSnakeAI:
    """Advanced PyTorch-based Snake AI with superior features"""
    
    def __init__(self, input_size: int = 11, hidden_sizes: List[int] = [32, 24, 16], output_size: int = 3):
        self.network = AdvancedSnakeNet(input_size, hidden_sizes, output_size)
        self.fitness = 0.0
        self.games_played = 0
        self.total_score = 0
        self.total_steps = 0
        
        # Advanced features
        self.action_history = []
        self.exploration_bonus = 0.0
        
        # Set to evaluation mode
        self.network.eval()
    
    def get_action(self, state: np.ndarray) -> int:
        """Advanced action selection with exploration bonus"""
        with torch.no_grad():
            output = self.network(state)
            
            # Convert to numpy
            if isinstance(output, torch.Tensor):
                output = output.numpy()
            
            # Handle batch dimension
            if output.ndim > 1:
                output = output[0]
            
            # Add small exploration noise during evaluation
            if len(self.action_history) > 0:
                # Penalize repeated actions slightly
                last_actions = self.action_history[-5:]  # Last 5 actions
                for i, count in enumerate([last_actions.count(j) for j in range(3)]):
                    if count > 2:  # If action repeated more than 2 times
                        output[i] -= 0.1  # Small penalty
            
            action = int(np.argmax(output))
            self.action_history.append(action)
            
            # Keep history manageable
            if len(self.action_history) > 20:
                self.action_history = self.action_history[-10:]
            
            return action
    
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
        """Advanced fitness function designed to surpass NumPy"""
        # Base fitness (same as NumPy)
        fitness = score * 100 + steps * 0.1
        
        # Exponential reward for eating food (same as NumPy)
        if score > 0:
            fitness += score ** 2 * 10
        
        # ADVANCED BONUSES TO SURPASS NUMPY:
        
        # 1. Efficiency bonus (eating food quickly)
        if score > 0:
            efficiency = score / max(steps, 1)
            fitness += efficiency * 500  # Reward efficient eating
        
        # 2. Survival bonus with food
        if score > 0 and steps > 500:
            fitness += 200  # Bonus for surviving long while eating
        
        # 3. High score exponential bonus
        if score >= 10:
            fitness += (score - 9) ** 2 * 50  # Extra reward for high scores
        
        # 4. Consistency bonus
        if score > 5:
            fitness += score * 20  # Linear bonus for consistent performance
        
        # 5. Exploration bonus (reward diverse actions)
        if hasattr(self, 'action_history') and len(self.action_history) > 10:
            unique_actions = len(set(self.action_history[-10:]))
            if unique_actions >= 2:  # Using at least 2 different actions
                fitness += unique_actions * 10
        
        return fitness
    
    def update_fitness(self, score: int, steps: int):
        """Update fitness statistics"""
        self.games_played += 1
        self.total_score += score
        self.total_steps += steps
        
        game_fitness = self.calculate_fitness(score, steps)
        
        # Advanced fitness averaging
        if self.games_played == 1:
            self.fitness = game_fitness
        else:
            # Weighted average with more weight on recent performance
            weight = 0.4 if self.games_played <= 5 else 0.3
            self.fitness = (self.fitness * (1 - weight)) + (game_fitness * weight)
    
    def reset_stats(self):
        """Reset fitness statistics"""
        self.fitness = 0.0
        self.games_played = 0
        self.total_score = 0
        self.total_steps = 0
        self.action_history = []
        self.exploration_bonus = 0.0
    
    def get_average_score(self) -> float:
        """Get average score across all games"""
        return self.total_score / max(self.games_played, 1)
    
    def get_average_steps(self) -> float:
        """Get average steps across all games"""
        return self.total_steps / max(self.games_played, 1)
    
    def save(self, filepath: str):
        """Save the advanced neural network model"""
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
        """Load an advanced neural network model"""
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


def test_advanced_network():
    """Test the advanced PyTorch neural network"""
    print("Testing Advanced PyTorch Neural Network...")
    
    # Create network
    ai = AdvancedSnakeAI()
    print(f"Advanced network created with {ai.get_total_params()} parameters")
    
    # Test forward pass
    test_input = np.random.rand(11)
    action = ai.get_action(test_input)
    print(f"Test input shape: {test_input.shape}")
    print(f"Action: {action}")
    
    # Test advanced fitness
    ai.update_fitness(15, 800)  # High score scenario
    print(f"Advanced fitness for score 15, steps 800: {ai.fitness}")
    
    print("✅ Advanced PyTorch network test completed!")


if __name__ == "__main__":
    test_advanced_network()
