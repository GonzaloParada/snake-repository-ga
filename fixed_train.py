#!/usr/bin/env python3
"""
Fixed training script with corrected fitness function
"""

import os
import sys
from genetic_algorithm import GeneticAlgorithm

def fixed_train():
    """Fixed training with corrected fitness function"""
    print("🔧 FIXED TRAINING - SNAKE AI")
    print("=" * 50)
    print("Using corrected fitness function...")
    
    # Good parameters with corrected fitness
    ga = GeneticAlgorithm(
        population_size=60,       # Moderate population
        mutation_rate=0.15,       # Moderate mutation
        mutation_strength=0.4,    # Moderate strength
        elite_percentage=0.15,    # Keep more good individuals
        crossover_rate=0.8,       # Good crossover rate
        games_per_individual=3    # Fewer games but better fitness
    )
    
    print(f"Training parameters:")
    print(f"  Population size: 60")
    print(f"  Generations: 40")
    print(f"  Games per individual: 3")
    print(f"  Mutation rate: 0.15")
    print(f"  Mutation strength: 0.4")
    print(f"  Elite percentage: 0.15")
    print(f"  Using multiprocessing: False")
    print()
    print("🔧 FIXES APPLIED:")
    print("  - Corrected fitness function")
    print("  - Penalty for dying too quickly")
    print("  - Better survival rewards")
    print("  - More balanced parameters")
    print()
    print("⏱️  This will take about 8-12 minutes...")
    print()
    
    try:
        # Train with corrected fitness
        ga.train(
            generations=40,
            save_interval=8,
            use_multiprocessing=False
        )
        
        print("\n✅ Fixed training completed successfully!")
        print("🎮 Test the FIXED AI:")
        print("   python main.py play")
        print("   python main.py play --games 10 --no-visual")
        print()
        print("📊 Compare with broken model:")
        print("   python main.py play models/best_snake_final.pkl --games 5 --no-visual")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        if ga.best_individual is not None:
            print("💾 Saving current best model...")
            ga.save_best_model("models/best_snake_fixed_interrupted.pkl")
            print("✅ Model saved!")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")

if __name__ == "__main__":
    fixed_train()
