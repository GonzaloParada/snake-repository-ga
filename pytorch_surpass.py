#!/usr/bin/env python3
"""
PyTorch Training to SURPASS NumPy Performance
Target: Beat 28.00 average score
"""

import os
import sys
from pytorch_advanced_genetic import AdvancedGeneticAlgorithm

def surpass_numpy():
    """Train PyTorch model to surpass NumPy's 28.00 average score"""
    print("🏆 PYTORCH MISSION: SURPASS NUMPY!")
    print("=" * 60)
    print("Current standings:")
    print("  🥇 NumPy Champion: 28.00 average score")
    print("  🥈 PyTorch Current: 3.70 average score")
    print("  🎯 Mission: BEAT 28.00!")
    print()
    
    # Advanced configuration designed to win
    ga = AdvancedGeneticAlgorithm(
        population_size=120,     # Large population for diversity
        mutation_rate=0.15,      # Balanced exploration
        mutation_strength=0.3,   # Moderate changes
        elite_percentage=0.15,   # Keep good solutions
        crossover_rate=0.85,     # High recombination
        games_per_individual=7   # Excellent evaluation (vs NumPy's 3)
    )
    
    print(f"🚀 ADVANCED PYTORCH CONFIGURATION:")
    print(f"  Population: 120 (vs NumPy's 100)")
    print(f"  Games per eval: 7 (vs NumPy's 3) - Better assessment!")
    print(f"  Network: [32,24,16] + BatchNorm + Dropout")
    print(f"  Advanced fitness: 5 bonus types")
    print(f"  Adaptive mutation + Tournament selection")
    print(f"  Multiple crossover strategies")
    print(f"  Action history tracking")
    print()
    print("🎯 ADVANTAGES OVER NUMPY:")
    print("  ✅ Larger, deeper network")
    print("  ✅ Batch normalization")
    print("  ✅ Advanced fitness function")
    print("  ✅ Better evaluation (7 games vs 3)")
    print("  ✅ Adaptive parameters")
    print("  ✅ Modern initialization")
    print()
    print("⏱️  Estimated time: 15-20 minutes")
    print("🏆 Expected result: 30+ average score!")
    print()
    
    try:
        # Train to surpass NumPy
        ga.train(
            generations=25,  # Focused training
            save_interval=5
        )
        
        print("\n" + "="*60)
        print("🏁 TRAINING COMPLETED!")
        
        if ga.best_individual:
            final_score = ga.best_individual.get_average_score()
            print(f"🎯 Final PyTorch Score: {final_score:.2f}")
            print(f"🥇 NumPy Benchmark: 28.00")
            
            if final_score > 28:
                print("🏆 SUCCESS! PYTORCH SURPASSED NUMPY!")
                print(f"🎉 Victory margin: +{final_score - 28:.2f} points!")
            elif final_score > 25:
                print("🎯 VERY CLOSE! Almost there!")
                print("💡 Suggestion: Try more generations or larger population")
            elif final_score > 15:
                print("📈 GOOD PROGRESS! Significant improvement!")
                print("💡 Suggestion: The advanced features are working!")
            else:
                print("🔄 NEEDS MORE WORK")
                print("💡 Suggestion: Check network architecture or fitness function")
        
        print("\n🎮 Test the ADVANCED PyTorch AI:")
        print("   python pytorch_advanced_play.py")
        print("   python pytorch_advanced_play.py --games 10 --no-visual")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        if ga.best_individual is not None:
            print("💾 Saving current best model...")
            ga.save_best_model("models/best_snake_advanced_interrupted.pth")
            print("✅ Model saved!")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    surpass_numpy()
