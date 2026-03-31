#!/usr/bin/env python3
"""
Dynamic Step Limits Analysis
Show how step limits scale with snake length and board utilization
"""

def analyze_dynamic_step_limits():
    """Analyze how step limits change with snake growth"""
    print("📊 DYNAMIC STEP LIMITS ANALYSIS")
    print("=" * 60)
    
    # Game parameters
    grid_width, grid_height = 20, 20
    board_size = grid_width * grid_height  # 400
    
    # Dynamic limit parameters
    base_steps_without_food = 150
    steps_per_segment = 8
    
    print(f"🎮 Board size: {grid_width}x{grid_height} = {board_size} cells")
    print(f"⚙️  Base steps: {base_steps_without_food}")
    print(f"⚙️  Steps per segment: {steps_per_segment}")
    print(f"⚙️  Board utilization bonus: up to 200 steps")
    print()
    
    # Analyze different snake lengths
    snake_lengths = [3, 10, 20, 30, 50, 75, 100, 150, 200, 300]
    
    print(f"{'Score':<6} {'Length':<7} {'Utilization':<12} {'Dynamic Limit':<14} {'vs Fixed 200':<12}")
    print("-" * 65)
    
    for length in snake_lengths:
        if length > board_size:
            continue
            
        score = length - 3  # Score = length - initial length
        board_utilization = length / board_size
        
        # Dynamic limit calculation
        dynamic_limit = (
            base_steps_without_food +           # Base: 150
            (length * steps_per_segment) +      # +8 per segment  
            (board_utilization * 200)           # +up to 200 for utilization
        )
        
        # Comparison with fixed limit
        vs_fixed = "✅ Better" if dynamic_limit > 200 else "❌ Worse"
        if dynamic_limit == 200:
            vs_fixed = "🟰 Same"
        
        print(f"{score:<6} {length:<7} {board_utilization:<12.2%} {dynamic_limit:<14.0f} {vs_fixed:<12}")
    
    print()
    print("💡 KEY INSIGHTS:")
    print("=" * 30)
    
    # Calculate some key points
    small_snake = base_steps_without_food + (10 * steps_per_segment) + (10/board_size * 200)
    medium_snake = base_steps_without_food + (50 * steps_per_segment) + (50/board_size * 200)
    large_snake = base_steps_without_food + (150 * steps_per_segment) + (150/board_size * 200)
    
    print(f"🐍 Small snake (10 segments): {small_snake:.0f} steps")
    print(f"🐍 Medium snake (50 segments): {medium_snake:.0f} steps") 
    print(f"🐍 Large snake (150 segments): {large_snake:.0f} steps")
    print()
    print(f"📈 Growth rate: ~{steps_per_segment} steps per food eaten")
    print(f"🎯 At score 30: {base_steps_without_food + 33 * steps_per_segment + (33/board_size * 200):.0f} steps allowed")
    print(f"🎯 At score 50: {base_steps_without_food + 53 * steps_per_segment + (53/board_size * 200):.0f} steps allowed")
    print(f"🎯 At score 100: {base_steps_without_food + 103 * steps_per_segment + (103/board_size * 200):.0f} steps allowed")
    print()
    print("✅ ADVANTAGES:")
    print("   • Scales naturally with snake growth")
    print("   • Allows for complex navigation in late game")
    print("   • Prevents infinite loops in early game")
    print("   • Considers board space utilization")

def compare_fixed_vs_dynamic():
    """Compare fixed vs dynamic step limits across different scenarios"""
    print("\n🔄 FIXED vs DYNAMIC COMPARISON")
    print("=" * 50)
    
    scenarios = [
        {"score": 5, "description": "Early game"},
        {"score": 15, "description": "Mid game"},
        {"score": 30, "description": "Good score"},
        {"score": 50, "description": "High score"},
        {"score": 100, "description": "Excellent score"},
        {"score": 200, "description": "Near perfect"}
    ]
    
    base_steps = 150
    steps_per_segment = 8
    board_size = 400
    
    print(f"{'Scenario':<15} {'Score':<6} {'Fixed':<6} {'Dynamic':<8} {'Difference':<11} {'Benefit'}")
    print("-" * 70)
    
    for scenario in scenarios:
        score = scenario["score"]
        snake_length = score + 3  # Initial length is 3
        
        if snake_length > board_size:
            continue
            
        fixed_limit = 200
        
        board_utilization = snake_length / board_size
        dynamic_limit = base_steps + (snake_length * steps_per_segment) + (board_utilization * 200)
        
        difference = dynamic_limit - fixed_limit
        benefit = "✅ Much better" if difference > 100 else "✅ Better" if difference > 0 else "❌ Worse"
        
        print(f"{scenario['description']:<15} {score:<6} {fixed_limit:<6} {dynamic_limit:<8.0f} {difference:<+11.0f} {benefit}")

def recommend_parameters():
    """Recommend optimal parameters for different use cases"""
    print("\n⚙️  PARAMETER RECOMMENDATIONS")
    print("=" * 40)
    
    recommendations = {
        "Conservative (Safe)": {
            "base": 120,
            "per_segment": 6,
            "description": "Prevents most loops, may limit very high scores"
        },
        "Balanced (Recommended)": {
            "base": 150,
            "per_segment": 8,
            "description": "Good balance, allows high scores with loop prevention"
        },
        "Aggressive (High Scores)": {
            "base": 200,
            "per_segment": 10,
            "description": "Maximizes high score potential, minimal loop prevention"
        },
        "Adaptive (Smart)": {
            "base": 100,
            "per_segment": 12,
            "description": "Grows quickly with snake, very permissive for long snakes"
        }
    }
    
    for name, params in recommendations.items():
        print(f"\n🎯 {name}:")
        print(f"   Base steps: {params['base']}")
        print(f"   Per segment: {params['per_segment']}")
        print(f"   Description: {params['description']}")
        
        # Calculate example limits
        score_30_limit = params['base'] + (33 * params['per_segment']) + (33/400 * 200)
        score_100_limit = params['base'] + (103 * params['per_segment']) + (103/400 * 200)
        
        print(f"   At score 30: {score_30_limit:.0f} steps")
        print(f"   At score 100: {score_100_limit:.0f} steps")

def main():
    """Main analysis function"""
    analyze_dynamic_step_limits()
    compare_fixed_vs_dynamic()
    recommend_parameters()
    
    print(f"\n🎯 CONCLUSION:")
    print(f"Dynamic step limits are MUCH better for high scores!")
    print(f"Current settings (150 base + 8 per segment) are well balanced.")

if __name__ == "__main__":
    main()
