#!/usr/bin/env python3
"""
Improved PyTorch Training Script for Snake AI
Fixed initialization and parameters to match NumPy performance
"""

import os
import sys
from pytorch_genetic import PyTorchGeneticAlgorithm

def pytorch_retrain():
    """Retrain PyTorch model with corrected parameters"""
    print("🔥 PYTORCH RETRAINING - SNAKE AI (FIXED)")
    print("=" * 50)
    print("Fixed initialization and parameters to match NumPy performance...")
    
    # More aggressive parameters to overcome poor initialization
    ga = PyTorchGeneticAlgorithm(
        population_size=100,     # Same population
        mutation_rate=0.2,       # Higher mutation to explore more
        mutation_strength=0.8,   # Stronger mutations
        elite_percentage=0.1,    # Less elitism, more exploration
        crossover_rate=0.9,      # More crossover
        games_per_individual=5   # Better evaluation
    )
    
    print(f"Training parameters (PYTORCH FIXED):")
    print(f"  Population size: 100")
    print(f"  Generations: 30 (faster iterations)")
    print(f"  Games per individual: 5 (better evaluation)")
    print(f"  Mutation rate: 0.2 (higher exploration)")
    print(f"  Mutation strength: 0.8 (stronger changes)")
    print(f"  Elite percentage: 0.1 (less elitism)")
    print(f"  Network: 11 inputs, [16,12] layers (Fixed init)")
    print(f"  Target: Match 21+ average score from NumPy")
    print()
    print("⏱️  This will take about 8-12 minutes...")
    print("🎯 Should achieve much better performance!")
    print()
    
    try:
        # Train with corrected settings
        ga.train(
            generations=30,  # Fewer generations but better parameters
            save_interval=5,
            use_multiprocessing=False
        )
        
        print("\n✅ PyTorch retraining completed!")
        print("🎮 Test the FIXED PyTorch AI:")
        print("   python pytorch_play.py")
        print("   python pytorch_play.py --games 10 --no-visual")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        if ga.best_individual is not None:
            print("💾 Saving current best model...")
            ga.save_best_model("models/best_snake_pytorch_fixed.pth")
            print("✅ Model saved!")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    pytorch_retrain()
