#!/usr/bin/env python3
"""
Original working training script - back to basics
"""

import os
import sys
from genetic_algorithm import GeneticAlgorithm

def original_train():
    """Train with original working configuration"""
    print("🔄 ORIGINAL TRAINING - SNAKE AI")
    print("=" * 50)
    print("Back to the working configuration...")
    
    # Original working parameters
    ga = GeneticAlgorithm(
        population_size=100,     # Good population size
        mutation_rate=0.2,       # Original mutation rate
        mutation_strength=0.8,   # Original strength
        elite_percentage=0.2,    # Original elitism
        crossover_rate=0.8,      # Original crossover
        games_per_individual=5   # Original evaluation
    )
    
    print(f"Training parameters (ORIGINAL):")
    print(f"  Population size: 100")
    print(f"  Generations: 50")
    print(f"  Games per individual: 3")
    print(f"  Mutation rate: 0.1")
    print(f"  Mutation strength: 0.5")
    print(f"  Elite percentage: 0.2")
    print(f"  Network: 11 inputs, [16,12] layers")
    print(f"  Fitness: Original simple function")
    print()
    print("⏱️  This will take about 10-15 minutes...")
    print()
    
    try:
        # Train with original settings
        ga.train(
            generations=50,
            save_interval=10,
            use_multiprocessing=False
        )
        
        print("\n✅ Original training completed successfully!")
        print("🎮 Test the ORIGINAL AI:")
        print("   python main.py play")
        print("   python main.py play --games 10 --no-visual")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        if ga.best_individual is not None:
            print("💾 Saving current best model...")
            ga.save_best_model("models/best_snake_original_interrupted.pkl")
            print("✅ Model saved!")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")

if __name__ == "__main__":
    original_train()
