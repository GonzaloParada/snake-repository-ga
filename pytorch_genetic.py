#!/usr/bin/env python3
"""
PyTorch-based Genetic Algorithm for Snake AI
"""

import numpy as np
import pickle
import time
from typing import List, Optional
from pytorch_network import SnakeAI
from snake_game import SnakeGame

class PyTorchGeneticAlgorithm:
    """Genetic Algorithm using PyTorch neural networks"""
    
    def __init__(self, 
                 population_size: int = 100,
                 mutation_rate: float = 0.1,
                 mutation_strength: float = 0.5,
                 elite_percentage: float = 0.2,
                 crossover_rate: float = 0.8,
                 games_per_individual: int = 3):
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_percentage = elite_percentage
        self.crossover_rate = crossover_rate
        self.games_per_individual = games_per_individual
        
        self.population: List[SnakeAI] = []
        self.best_individual: Optional[SnakeAI] = None
        self.best_fitness = 0.0
        self.generation = 0
        self.fitness_history = []
        
    def create_initial_population(self):
        """Create initial random population"""
        print(f"Creating initial population of {self.population_size} individuals...")
        
        self.population = []
        for i in range(self.population_size):
            individual = SnakeAI()
            self.population.append(individual)
            
            if (i + 1) % 20 == 0:
                print(f"Created {i + 1}/{self.population_size} individuals")
        
        print(f"✅ Initial population created!")
    
    def evaluate_individual(self, individual: SnakeAI) -> float:
        """Evaluate a single individual by playing games"""
        individual.reset_stats()
        
        for game_num in range(self.games_per_individual):
            game = SnakeGame(headless=True)
            state = game.reset()
            
            while not game.game_over:
                action = individual.get_action(state)
                state, reward, done = game.step(action)
            
            individual.update_fitness(game.score, game.steps)
            game.close()
        
        return individual.fitness
    
    def evaluate_population(self):
        """Evaluate entire population"""
        print(f"Evaluating generation {self.generation}...")
        
        for i, individual in enumerate(self.population):
            fitness = self.evaluate_individual(individual)
            
            if (i + 1) % 10 == 0:
                print(f"Evaluated {i + 1}/{self.population_size} individuals")
        
        # Sort by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Update best individual
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_individual = self.population[0]
        
        # Record statistics
        avg_fitness = np.mean([ind.fitness for ind in self.population])
        self.fitness_history.append({
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': avg_fitness
        })
        
        print(f"Generation {self.generation}: Best={self.best_fitness:.2f}, Avg={avg_fitness:.2f}")
    
    def selection(self) -> List[SnakeAI]:
        """Select parents for reproduction using tournament selection"""
        num_elite = int(self.population_size * self.elite_percentage)
        
        # Elite selection
        parents = self.population[:num_elite].copy()
        
        # Tournament selection for the rest
        tournament_size = 3
        while len(parents) < self.population_size:
            # Tournament
            tournament = np.random.choice(self.population, tournament_size, replace=False)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents
    
    def crossover(self, parent1: SnakeAI, parent2: SnakeAI) -> SnakeAI:
        """Create offspring through crossover"""
        child = SnakeAI()
        
        if np.random.random() < self.crossover_rate:
            # Get parent weights
            weights1 = parent1.get_weights()
            weights2 = parent2.get_weights()
            
            # Uniform crossover
            mask = np.random.random(len(weights1)) < 0.5
            child_weights = np.where(mask, weights1, weights2)
            
            child.set_weights(child_weights)
        else:
            # No crossover, just copy one parent
            if np.random.random() < 0.5:
                child.set_weights(parent1.get_weights())
            else:
                child.set_weights(parent2.get_weights())
        
        return child
    
    def mutate(self, individual: SnakeAI):
        """Mutate an individual"""
        weights = individual.get_weights()
        
        # Gaussian mutation
        mutation_mask = np.random.random(len(weights)) < self.mutation_rate
        mutations = np.random.normal(0, self.mutation_strength, len(weights))
        weights[mutation_mask] += mutations[mutation_mask]
        
        individual.set_weights(weights)
    
    def create_next_generation(self):
        """Create next generation through selection, crossover, and mutation"""
        print("Creating next generation...")
        
        # Selection
        parents = self.selection()
        
        # Create new population
        new_population = []
        
        # Keep elite unchanged
        num_elite = int(self.population_size * self.elite_percentage)
        for i in range(num_elite):
            elite = SnakeAI()
            elite.set_weights(parents[i].get_weights())
            new_population.append(elite)
        
        # Create offspring
        while len(new_population) < self.population_size:
            parent1 = np.random.choice(parents)
            parent2 = np.random.choice(parents)
            
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
    
    def save_best_model(self, filepath: str):
        """Save the best model"""
        if self.best_individual:
            self.best_individual.save(filepath)
    
    def save_training_history(self, filepath: str):
        """Save training history"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.fitness_history, f)
    
    def train(self, generations: int, save_interval: int = 10, use_multiprocessing: bool = False):
        """Train the population for specified generations"""
        print(f"🧠 PYTORCH TRAINING - SNAKE AI")
        print("=" * 50)
        print(f"Training parameters:")
        print(f"  Population size: {self.population_size}")
        print(f"  Generations: {generations}")
        print(f"  Games per individual: {self.games_per_individual}")
        print(f"  Mutation rate: {self.mutation_rate}")
        print(f"  Mutation strength: {self.mutation_strength}")
        print(f"  Elite percentage: {self.elite_percentage}")
        print(f"  Using PyTorch: ✅")
        print()
        
        start_time = time.time()
        
        # Create initial population if needed
        if not self.population:
            self.create_initial_population()
        
        # Training loop
        for gen in range(generations):
            self.evaluate_population()
            
            # Save best model periodically
            if (self.generation) % save_interval == 0:
                self.save_best_model(f"models/best_snake_pytorch_gen_{self.generation}.pth")
            
            # Create next generation (except for last generation)
            if gen < generations - 1:
                self.create_next_generation()
        
        # Final save
        self.save_best_model("models/best_snake_pytorch_final.pth")
        self.save_training_history("models/pytorch_training_history.pkl")
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best fitness achieved: {self.best_fitness:.2f}")
        print(f"Saved best model to models/best_snake_pytorch_final.pth")
        print(f"Saved training history to models/pytorch_training_history.pkl")


if __name__ == "__main__":
    # Test PyTorch genetic algorithm
    ga = PyTorchGeneticAlgorithm(
        population_size=20,  # Small for testing
        games_per_individual=2
    )
    
    print("Testing PyTorch Genetic Algorithm...")
    ga.train(generations=3)
    print("✅ Test completed!")
