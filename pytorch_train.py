#!/usr/bin/env python3
"""
PyTorch Training Script for Snake AI
Uses the proven parameters from the successful NumPy implementation
"""

import os
import sys
from pytorch_genetic import PyTorchGeneticAlgorithm

def pytorch_train():
    """Train with PyTorch using proven parameters"""
    print("🔥 PYTORCH TRAINING - SNAKE AI")
    print("=" * 50)
    print("Using PyTorch with proven genetic algorithm parameters...")
    
    # Use the exact same parameters that gave us 21.60 average score
    ga = PyTorchGeneticAlgorithm(
        population_size=100,     # Same as successful NumPy version
        mutation_rate=0.1,       # Proven mutation rate
        mutation_strength=0.5,   # Proven strength
        elite_percentage=0.2,    # Proven elitism
        crossover_rate=0.8,      # Proven crossover
        games_per_individual=3   # Proven evaluation
    )
    
    print(f"Training parameters (PYTORCH + PROVEN CONFIG):")
    print(f"  Population size: 100")
    print(f"  Generations: 50")
    print(f"  Games per individual: 3")
    print(f"  Mutation rate: 0.1")
    print(f"  Mutation strength: 0.5")
    print(f"  Elite percentage: 0.2")
    print(f"  Network: 11 inputs, [16,12] layers (PyTorch)")
    print(f"  Fitness: score * 100 + steps * 0.1 + score^2 * 10")
    print(f"  Expected performance: 21+ average score")
    print()
    print("⏱️  This will take about 10-15 minutes...")
    print("🚀 PyTorch should be faster than NumPy!")
    print()
    
    try:
        # Train with proven settings
        ga.train(
            generations=50,
            save_interval=10,
            use_multiprocessing=False
        )
        
        print("\n✅ PyTorch training completed successfully!")
        print("🎮 Test the PyTorch AI:")
        print("   python pytorch_play.py")
        print("   python pytorch_play.py --games 10 --no-visual")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        if ga.best_individual is not None:
            print("💾 Saving current best model...")
            ga.save_best_model("models/best_snake_pytorch_interrupted.pth")
            print("✅ Model saved!")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    pytorch_train()
