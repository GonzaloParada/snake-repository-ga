#!/usr/bin/env python3
"""
Better training script with improved parameters for better results
"""

import os
import sys
from genetic_algorithm import GeneticAlgorithm

def better_train():
    """Better training with improved parameters"""
    print("🧠 BETTER TRAINING - SNAKE AI")
    print("=" * 50)
    print("Using improved parameters for better results...")
    
    # Better parameters for improved training
    ga = GeneticAlgorithm(
        population_size=100,      # Larger population
        mutation_rate=0.2,       # Higher mutation for exploration
        mutation_strength=0.8,   # Stronger mutations
        elite_percentage=0.1,    # Keep fewer elite (more diversity)
        crossover_rate=0.8,      # Good crossover rate
        games_per_individual=5   # More games for better evaluation
    )
    
    print(f"Training parameters:")
    print(f"  Population size: 50")
    print(f"  Generations: 30")
    print(f"  Games per individual: 5")
    print(f"  Mutation rate: 0.2")
    print(f"  Mutation strength: 0.8")
    print(f"  Elite percentage: 0.1")
    print(f"  Using multiprocessing: False")
    print()
    print("⏱️  This will take about 5-10 minutes...")
    print()
    
    try:
        # Train for more generations
        ga.train(
            generations=30,
            save_interval=5,
            use_multiprocessing=False
        )
        
        print("\n✅ Better training completed successfully!")
        print("🎮 Test the improved AI:")
        print("   python main.py play")
        print("   python main.py play --games 10 --no-visual")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        if ga.best_individual is not None:
            print("💾 Saving current best model...")
            ga.save_best_model("models/best_snake_interrupted.pkl")
            print("✅ Model saved!")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")

if __name__ == "__main__":
    better_train()
