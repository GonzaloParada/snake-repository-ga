import numpy as np
import random
import multiprocessing as mp
from typing import List, Tuple
import time
import os
from snake_game import SnakeGame
from neural_network import NeuralNetwork, SnakeAI

class GeneticAlgorithm:
    def __init__(self, 
                 population_size: int = 100,
                 mutation_rate: float = 0.1,
                 mutation_strength: float = 0.5,
                 elite_percentage: float = 0.2,
                 crossover_rate: float = 0.8,
                 games_per_individual: int = 3):
        """
        Initialize genetic algorithm
        
        Args:
            population_size: Number of individuals in population
            mutation_rate: Probability of mutation for each gene
            mutation_strength: Standard deviation of mutation noise
            elite_percentage: Percentage of best individuals to keep unchanged
            crossover_rate: Probability of crossover between parents
            games_per_individual: Number of games each AI plays for evaluation
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_percentage = elite_percentage
        self.crossover_rate = crossover_rate
        self.games_per_individual = games_per_individual
        
        self.generation = 0
        self.population = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None
        
        # Create initial population
        self._create_initial_population()
    
    def _create_initial_population(self):
        """Create initial random population"""
        print(f"Creating initial population of {self.population_size} individuals...")
        self.population = []
        
        for i in range(self.population_size):
            nn = NeuralNetwork()
            ai = SnakeAI(nn)
            self.population.append(ai)
            
            if (i + 1) % 20 == 0:
                print(f"Created {i + 1}/{self.population_size} individuals")
    
    def evaluate_individual(self, ai: SnakeAI) -> float:
        """Evaluate a single individual by playing multiple games"""
        ai.reset_fitness()
        
        for game_num in range(self.games_per_individual):
            game = SnakeGame(speed=1000, headless=True)  # Headless mode for training
            state = game.reset()
            
            while not game.game_over:
                action = ai.get_action(state)
                state, reward, done = game.step(action)
            
            ai.update_fitness(game.score, game.steps)
            game.close()
        
        return ai.fitness
    
    def evaluate_population_parallel(self):
        """Evaluate entire population using multiprocessing"""
        print(f"Evaluating generation {self.generation}...")
        
        # Use multiprocessing to evaluate individuals in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            fitnesses = pool.map(self.evaluate_individual, self.population)
        
        # Update fitness values
        for i, fitness in enumerate(fitnesses):
            self.population[i].fitness = fitness
        
        # Sort population by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Update statistics
        best_fitness = self.population[0].fitness
        avg_fitness = np.mean([ai.fitness for ai in self.population])
        
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        
        # Update best individual
        if self.best_individual is None or best_fitness > self.best_individual.fitness:
            self.best_individual = self.population[0].copy()
        
        print(f"Generation {self.generation}: Best={best_fitness:.2f}, Avg={avg_fitness:.2f}")
    
    def evaluate_population_sequential(self):
        """Evaluate population sequentially (for debugging or single-core systems)"""
        print(f"Evaluating generation {self.generation}...")
        
        for i, ai in enumerate(self.population):
            fitness = self.evaluate_individual(ai)
            if (i + 1) % 10 == 0:
                print(f"Evaluated {i + 1}/{self.population_size} individuals")
        
        # Sort population by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Update statistics
        best_fitness = self.population[0].fitness
        avg_fitness = np.mean([ai.fitness for ai in self.population])
        
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        
        # Update best individual
        if self.best_individual is None or best_fitness > self.best_individual.fitness:
            self.best_individual = self.population[0].copy()
        
        print(f"Generation {self.generation}: Best={best_fitness:.2f}, Avg={avg_fitness:.2f}")
    
    def selection(self) -> Tuple[SnakeAI, SnakeAI]:
        """Tournament selection to choose two parents"""
        tournament_size = 5
        
        def tournament():
            competitors = random.sample(self.population, tournament_size)
            return max(competitors, key=lambda x: x.fitness)
        
        parent1 = tournament()
        parent2 = tournament()
        return parent1, parent2
    
    def crossover(self, parent1: SnakeAI, parent2: SnakeAI) -> Tuple[SnakeAI, SnakeAI]:
        """Create two offspring using crossover"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Get parent genomes
        genome1 = parent1.neural_network.get_weights_biases()
        genome2 = parent2.neural_network.get_weights_biases()
        
        # Single-point crossover
        crossover_point = random.randint(1, len(genome1) - 1)
        
        child1_genome = np.concatenate([genome1[:crossover_point], genome2[crossover_point:]])
        child2_genome = np.concatenate([genome2[:crossover_point], genome1[crossover_point:]])
        
        # Create offspring
        child1 = SnakeAI(NeuralNetwork())
        child2 = SnakeAI(NeuralNetwork())
        
        child1.neural_network.set_weights_biases(child1_genome)
        child2.neural_network.set_weights_biases(child2_genome)
        
        return child1, child2
    
    def mutate(self, individual: SnakeAI):
        """Mutate an individual"""
        genome = individual.neural_network.get_weights_biases()
        
        # Add Gaussian noise to genes based on mutation rate
        mutation_mask = np.random.random(len(genome)) < self.mutation_rate
        mutations = np.random.normal(0, self.mutation_strength, len(genome))
        genome[mutation_mask] += mutations[mutation_mask]
        
        individual.neural_network.set_weights_biases(genome)
    
    def create_next_generation(self):
        """Create the next generation using selection, crossover, and mutation"""
        print("Creating next generation...")
        
        new_population = []
        
        # Keep elite individuals
        elite_count = int(self.population_size * self.elite_percentage)
        for i in range(elite_count):
            new_population.append(self.population[i].copy())
        
        # Create offspring to fill the rest of the population
        while len(new_population) < self.population_size:
            parent1, parent2 = self.selection()
            child1, child2 = self.crossover(parent1, parent2)
            
            self.mutate(child1)
            self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        self.population = new_population[:self.population_size]
        self.generation += 1
    
    def train(self, generations: int, save_interval: int = 10, use_multiprocessing: bool = True):
        """Train the population for specified number of generations"""
        print(f"Starting training for {generations} generations...")
        print(f"Population size: {self.population_size}")
        print(f"Games per individual: {self.games_per_individual}")
        print(f"Using multiprocessing: {use_multiprocessing}")
        
        start_time = time.time()
        
        for gen in range(generations):
            # Evaluate population
            if use_multiprocessing:
                self.evaluate_population_parallel()
            else:
                self.evaluate_population_sequential()
            
            # Save best model periodically
            if gen % save_interval == 0:
                self.save_best_model(f"models/best_snake_gen_{self.generation}.pkl")
            
            # Create next generation (except for the last generation)
            if gen < generations - 1:
                self.create_next_generation()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best fitness achieved: {max(self.best_fitness_history):.2f}")
        
        # Save final model
        self.save_best_model("models/best_snake_final.pkl")
        self.save_training_history("models/training_history.pkl")
    
    def save_best_model(self, filename: str):
        """Save the best individual"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.best_individual.neural_network.save(filename)
        print(f"Saved best model to {filename}")
    
    def save_training_history(self, filename: str):
        """Save training history"""
        import pickle
        
        history = {
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'generation': self.generation,
            'population_size': self.population_size,
            'mutation_rate': self.mutation_rate,
            'mutation_strength': self.mutation_strength,
            'elite_percentage': self.elite_percentage,
            'crossover_rate': self.crossover_rate,
            'games_per_individual': self.games_per_individual
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(history, f)
        
        print(f"Saved training history to {filename}")

def worker_evaluate_individual(ai_and_games):
    """Worker function for multiprocessing evaluation"""
    ai, games_per_individual = ai_and_games
    ai.reset_fitness()
    
    for game_num in range(games_per_individual):
        game = SnakeGame(speed=1000, headless=True)  # Headless mode for training
        state = game.reset()
        
        while not game.game_over:
            action = ai.get_action(state)
            state, reward, done = game.step(action)
        
        ai.update_fitness(game.score, game.steps)
        game.close()
    
    return ai.fitness

if __name__ == "__main__":
    # Test genetic algorithm
    print("Testing Genetic Algorithm...")
    
    # Create a small population for testing
    ga = GeneticAlgorithm(population_size=20, games_per_individual=2)
    
    # Train for a few generations
    ga.train(generations=5, use_multiprocessing=False)
    
    print("Test completed!")
