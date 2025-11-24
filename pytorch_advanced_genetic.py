#!/usr/bin/env python3
"""
Advanced PyTorch Genetic Algorithm - Designed to SURPASS NumPy
"""

import numpy as np
import pickle
import time
from typing import List, Optional
from pytorch_advanced import AdvancedSnakeAI
from snake_game import SnakeGame

class AdvancedGeneticAlgorithm:
    """Advanced Genetic Algorithm designed to surpass NumPy performance"""
    
    def __init__(self, 
                 population_size: int = 150,  # Larger population
                 mutation_rate: float = 0.15,
                 mutation_strength: float = 0.3,
                 elite_percentage: float = 0.15,
                 crossover_rate: float = 0.85,
                 games_per_individual: int = 5):  # Better evaluation
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_percentage = elite_percentage
        self.crossover_rate = crossover_rate
        self.games_per_individual = games_per_individual
        
        self.population: List[AdvancedSnakeAI] = []
        self.best_individual: Optional[AdvancedSnakeAI] = None
        self.best_fitness = 0.0
        self.generation = 0
        self.fitness_history = []
        
        # Advanced features
        self.diversity_bonus = True
        self.adaptive_mutation = True
        self.tournament_size = 5  # Larger tournament
        
    def create_initial_population(self):
        """Create initial population with diversity"""
        print(f"Creating advanced population of {self.population_size} individuals...")
        
        self.population = []
        for i in range(self.population_size):
            individual = AdvancedSnakeAI()
            
            # Add diversity to initial population
            if i > 0:  # Keep first individual with default initialization
                weights = individual.get_weights()
                # Add varied initialization
                noise_scale = 0.5 + (i % 10) * 0.1  # Varying noise levels
                noise = np.random.normal(0, noise_scale, weights.shape)
                individual.set_weights(weights + noise)
            
            self.population.append(individual)
            
            if (i + 1) % 30 == 0:
                print(f"Created {i + 1}/{self.population_size} individuals")
        
        print(f"✅ Advanced population created with diversity!")
    
    def evaluate_individual(self, individual: AdvancedSnakeAI) -> float:
        """Advanced evaluation with multiple scenarios"""
        individual.reset_stats()
        
        # Evaluate on multiple games for better assessment
        for game_num in range(self.games_per_individual):
            # Vary game conditions slightly for robustness
            speed = 1000 if game_num < 3 else 500  # Some slower games
            game = SnakeGame(headless=True, speed=speed)
            state = game.reset()
            
            while not game.game_over:
                action = individual.get_action(state)
                state, reward, done = game.step(action)
            
            individual.update_fitness(game.score, game.steps)
            game.close()
        
        return individual.fitness
    
    def evaluate_population(self):
        """Evaluate population with progress tracking"""
        print(f"Evaluating advanced generation {self.generation}...")
        
        fitness_values = []
        for i, individual in enumerate(self.population):
            fitness = self.evaluate_individual(individual)
            fitness_values.append(fitness)
            
            if (i + 1) % 25 == 0:
                print(f"Evaluated {i + 1}/{self.population_size} individuals")
        
        # Sort by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Update best individual
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_individual = self.population[0]
        
        # Calculate statistics
        avg_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)
        top_10_avg = np.mean(fitness_values[:10])
        
        self.fitness_history.append({
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': avg_fitness,
            'std_fitness': std_fitness,
            'top_10_avg': top_10_avg
        })
        
        print(f"Generation {self.generation}: Best={self.best_fitness:.2f}, Avg={avg_fitness:.2f}, Top10={top_10_avg:.2f}")
        
        # Adaptive mutation based on diversity
        if self.adaptive_mutation:
            if std_fitness < 50:  # Low diversity
                self.mutation_rate = min(0.25, self.mutation_rate * 1.1)
                print(f"  Low diversity detected, increasing mutation to {self.mutation_rate:.3f}")
            elif std_fitness > 200:  # High diversity
                self.mutation_rate = max(0.05, self.mutation_rate * 0.9)
                print(f"  High diversity, decreasing mutation to {self.mutation_rate:.3f}")
    
    def tournament_selection(self, tournament_size: int = None) -> AdvancedSnakeAI:
        """Tournament selection with configurable size"""
        if tournament_size is None:
            tournament_size = self.tournament_size
            
        tournament = np.random.choice(self.population, tournament_size, replace=False)
        return max(tournament, key=lambda x: x.fitness)
    
    def advanced_crossover(self, parent1: AdvancedSnakeAI, parent2: AdvancedSnakeAI) -> AdvancedSnakeAI:
        """Advanced crossover with multiple strategies"""
        child = AdvancedSnakeAI()
        
        if np.random.random() < self.crossover_rate:
            weights1 = parent1.get_weights()
            weights2 = parent2.get_weights()
            
            # Choose crossover strategy
            strategy = np.random.choice(['uniform', 'blend', 'arithmetic'])
            
            if strategy == 'uniform':
                # Standard uniform crossover
                mask = np.random.random(len(weights1)) < 0.5
                child_weights = np.where(mask, weights1, weights2)
            elif strategy == 'blend':
                # Blend crossover (average with noise)
                alpha = 0.5 + np.random.normal(0, 0.1)
                alpha = np.clip(alpha, 0.2, 0.8)
                child_weights = alpha * weights1 + (1 - alpha) * weights2
            else:  # arithmetic
                # Arithmetic crossover
                child_weights = 0.5 * weights1 + 0.5 * weights2
            
            child.set_weights(child_weights)
        else:
            # No crossover, copy better parent
            if parent1.fitness > parent2.fitness:
                child.set_weights(parent1.get_weights())
            else:
                child.set_weights(parent2.get_weights())
        
        return child
    
    def advanced_mutation(self, individual: AdvancedSnakeAI):
        """Advanced mutation with multiple strategies"""
        weights = individual.get_weights()
        
        # Multiple mutation strategies
        if np.random.random() < 0.7:  # Gaussian mutation (most common)
            mutation_mask = np.random.random(len(weights)) < self.mutation_rate
            mutations = np.random.normal(0, self.mutation_strength, len(weights))
            weights[mutation_mask] += mutations[mutation_mask]
        
        elif np.random.random() < 0.9:  # Uniform mutation
            mutation_mask = np.random.random(len(weights)) < self.mutation_rate
            mutations = np.random.uniform(-self.mutation_strength, self.mutation_strength, len(weights))
            weights[mutation_mask] += mutations[mutation_mask]
        
        else:  # Creep mutation (small changes)
            mutation_mask = np.random.random(len(weights)) < self.mutation_rate * 2
            mutations = np.random.normal(0, self.mutation_strength * 0.1, len(weights))
            weights[mutation_mask] += mutations[mutation_mask]
        
        individual.set_weights(weights)
    
    def create_next_generation(self):
        """Create next generation with advanced techniques"""
        print("Creating advanced next generation...")
        
        # Calculate elite count
        num_elite = int(self.population_size * self.elite_percentage)
        
        new_population = []
        
        # Keep elite unchanged
        for i in range(num_elite):
            elite = AdvancedSnakeAI()
            elite.set_weights(self.population[i].get_weights())
            new_population.append(elite)
        
        # Create offspring with advanced techniques
        while len(new_population) < self.population_size:
            # Select parents using tournament selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Create child
            child = self.advanced_crossover(parent1, parent2)
            self.advanced_mutation(child)
            
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
    
    def train(self, generations: int, save_interval: int = 5):
        """Advanced training to surpass NumPy performance"""
        print(f"🚀 ADVANCED PYTORCH TRAINING - SURPASS NUMPY!")
        print("=" * 60)
        print(f"Target: BEAT NumPy's 28.00 average score!")
        print(f"Advanced parameters:")
        print(f"  Population size: {self.population_size} (vs NumPy's 100)")
        print(f"  Generations: {generations}")
        print(f"  Games per individual: {self.games_per_individual} (vs NumPy's 3)")
        print(f"  Network: [32,24,16] + BatchNorm + Dropout (vs NumPy's [16,12])")
        print(f"  Advanced fitness function with 5 bonus types")
        print(f"  Adaptive mutation and diverse crossover")
        print(f"  Tournament selection (size {self.tournament_size})")
        print()
        
        start_time = time.time()
        
        # Create initial population
        if not self.population:
            self.create_initial_population()
        
        # Training loop
        for gen in range(generations):
            self.evaluate_population()
            
            # Save best model periodically
            if (self.generation) % save_interval == 0:
                self.save_best_model(f"models/best_snake_advanced_gen_{self.generation}.pth")
                
                # Check if we're beating NumPy
                if self.best_individual:
                    avg_score = self.best_individual.get_average_score()
                    if avg_score > 20:
                        print(f"🎯 GETTING CLOSE! Average score: {avg_score:.2f}")
                    if avg_score > 28:
                        print(f"🏆 SURPASSED NUMPY! Average score: {avg_score:.2f}")
            
            # Create next generation (except for last generation)
            if gen < generations - 1:
                self.create_next_generation()
        
        # Final save
        self.save_best_model("models/best_snake_advanced_final.pth")
        self.save_training_history("models/advanced_training_history.pkl")
        
        training_time = time.time() - start_time
        print(f"\nAdvanced training completed in {training_time:.2f} seconds")
        print(f"Best fitness achieved: {self.best_fitness:.2f}")
        
        if self.best_individual:
            final_avg = self.best_individual.get_average_score()
            print(f"Final average score: {final_avg:.2f}")
            if final_avg > 28:
                print("🏆 SUCCESS! PyTorch SURPASSED NumPy!")
            elif final_avg > 20:
                print("🎯 CLOSE! Almost beat NumPy's 28.00!")
            else:
                print("📈 Good progress, may need more training...")
        
        print(f"Saved best model to models/best_snake_advanced_final.pth")


if __name__ == "__main__":
    # Test advanced genetic algorithm
    ga = AdvancedGeneticAlgorithm(
        population_size=50,  # Small for testing
        games_per_individual=3
    )
    
    print("Testing Advanced Genetic Algorithm...")
    ga.train(generations=5)
    print("✅ Test completed!")
