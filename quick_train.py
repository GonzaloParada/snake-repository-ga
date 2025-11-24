#!/usr/bin/env python3
"""
Quick training script for testing the genetic algorithm
Uses smaller parameters for faster training and testing
"""

import os
import sys
from genetic_algorithm import GeneticAlgorithm

def quick_train():
    """Quick training with small parameters for testing"""
    print("🚀 QUICK TRAINING - SNAKE AI")
    print("=" * 40)
    print("Using small parameters for fast testing...")
    
    # Small parameters for quick testing
    ga = GeneticAlgorithm(
        population_size=20,      # Small population
        mutation_rate=0.15,      # Slightly higher mutation
        mutation_strength=0.3,   # Lower mutation strength
        elite_percentage=0.25,   # Keep more elite
        crossover_rate=0.7,      # Lower crossover rate
        games_per_individual=2   # Fewer games per individual
    )
    
    print(f"Training parameters:")
    print(f"  Population size: 20")
    print(f"  Generations: 10")
    print(f"  Games per individual: 2")
    print(f"  Mutation rate: 0.15")
    print(f"  Using multiprocessing: False (for stability)")
    print()
    
    try:
        # Train for just 10 generations
        ga.train(
            generations=10,
            save_interval=5,
            use_multiprocessing=False  # Disable multiprocessing for stability
        )
        
        print("\n✅ Quick training completed successfully!")
        print("🎮 Now you can test the trained AI:")
        print("   python main.py play")
        print("   python main.py play --games 5 --no-visual")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        if ga.best_individual is not None:
            print("💾 Saving current best model...")
            ga.save_best_model("models/best_snake_interrupted.pkl")
            print("✅ Model saved! You can still test it with:")
            print("   python main.py play models/best_snake_interrupted.pkl")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("💡 Try running with different parameters or check the error above")

if __name__ == "__main__":
    quick_train()
