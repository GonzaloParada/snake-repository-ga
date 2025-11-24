#!/usr/bin/env python3
"""
Snake Game Genetic Algorithm
Main script to train AI or run trained models

Usage:
    python main.py train [options]    - Train a new AI model
    python main.py play [model_path]  - Play with a trained AI model
    python main.py human             - Play the game as human
    python main.py demo              - Run a demo with random AI
"""

import argparse
import os
import sys
import time
import pygame
from snake_game import SnakeGame
from neural_network import NeuralNetwork, SnakeAI
from genetic_algorithm import GeneticAlgorithm

def train_ai(args):
    """Train AI using genetic algorithm"""
    print("=" * 50)
    print("TRAINING SNAKE AI WITH GENETIC ALGORITHM")
    print("=" * 50)
    
    # Create genetic algorithm with specified parameters
    ga = GeneticAlgorithm(
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        elite_percentage=args.elite_percentage,
        crossover_rate=args.crossover_rate,
        games_per_individual=args.games_per_individual
    )
    
    print(f"Training parameters:")
    print(f"  Population size: {args.population_size}")
    print(f"  Generations: {args.generations}")
    print(f"  Games per individual: {args.games_per_individual}")
    print(f"  Mutation rate: {args.mutation_rate}")
    print(f"  Mutation strength: {args.mutation_strength}")
    print(f"  Elite percentage: {args.elite_percentage}")
    print(f"  Crossover rate: {args.crossover_rate}")
    print(f"  Multiprocessing: {not args.no_multiprocessing}")
    print()
    
    # Start training
    ga.train(
        generations=args.generations,
        save_interval=args.save_interval,
        use_multiprocessing=not args.no_multiprocessing
    )
    
    print("\nTraining completed!")
    print(f"Best model saved to: models/best_snake_final.pkl")
    print(f"Training history saved to: models/training_history.pkl")

def play_with_ai(model_path: str, visual: bool = True, games: int = 1):
    """Play game with trained AI"""
    print(f"Loading AI model from: {model_path}")
    
    try:
        nn = NeuralNetwork.load(model_path)
        ai = SnakeAI(nn)
        print("AI model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found!")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    total_score = 0
    total_steps = 0
    
    for game_num in range(games):
        print(f"\nGame {game_num + 1}/{games}")
        
        game = SnakeGame(speed=10 if visual else 1000)
        if visual:
            game.init_display()
        
        state = game.reset()
        
        while not game.game_over:
            if visual:
                # Handle pygame events to prevent window from freezing
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game.close()
                        return
            
            action = ai.get_action(state)
            state, reward, done = game.step(action)
            
            if visual:
                game.render()
        
        # Game over - show final state and wait for input if visual
        if visual:
            game.render()  # Show final state
            
            # Mark the collision point
            head_x, head_y = game.snake[0]
            collision_rect = pygame.Rect(
                head_x * game.block_size, 
                head_y * game.block_size, 
                game.block_size, 
                game.block_size
            )
            
            # Determine cause of death and mark collision
            if head_x < 0 or head_x >= game.grid_width or head_y < 0 or head_y >= game.grid_height:
                death_cause = "Hit Wall"
                # Draw dramatic red explosion effect for wall collision
                center = collision_rect.center
                for i in range(3):
                    radius = (i + 1) * 8
                    color_intensity = 255 - (i * 50)
                    pygame.draw.circle(game.screen, (color_intensity, 0, 0), center, radius, 2)
                
                # Draw red X over the head
                pygame.draw.line(game.screen, (255, 0, 0), 
                               collision_rect.topleft, collision_rect.bottomright, 4)
                pygame.draw.line(game.screen, (255, 0, 0), 
                               collision_rect.topright, collision_rect.bottomleft, 4)
                
                # Draw "WALL" text near collision
                font_collision = pygame.font.Font(None, 24)
                wall_text = font_collision.render("WALL!", True, (255, 255, 255))
                text_pos = (center[0] - wall_text.get_width() // 2, center[1] - 30)
                game.screen.blit(wall_text, text_pos)
                
            elif (head_x, head_y) in game.snake[1:]:
                death_cause = "Hit Self"
                # Draw dramatic self-collision effect
                center = collision_rect.center
                
                # Pulsing red circles around head
                for i in range(4):
                    radius = (i + 1) * 6
                    alpha = 255 - (i * 40)
                    color = (255, max(0, 100 - i * 25), 0)
                    pygame.draw.circle(game.screen, color, center, radius, 2)
                
                # Draw collision spark effect
                spark_length = 15
                import math
                for angle in range(0, 360, 45):
                    rad = math.radians(angle)
                    end_x = center[0] + int(spark_length * math.cos(rad))
                    end_y = center[1] + int(spark_length * math.sin(rad))
                    pygame.draw.line(game.screen, (255, 255, 0), center, (end_x, end_y), 2)
                
                # Mark the body segment that was hit with yellow outline
                for i, segment in enumerate(game.snake[1:], 1):
                    if segment == (head_x, head_y):
                        body_rect = pygame.Rect(
                            segment[0] * game.block_size,
                            segment[1] * game.block_size,
                            game.block_size,
                            game.block_size
                        )
                        pygame.draw.rect(game.screen, (255, 255, 0), body_rect, 3)
                        
                        # Draw "SELF" text
                        font_collision = pygame.font.Font(None, 20)
                        self_text = font_collision.render("SELF!", True, (255, 255, 0))
                        text_pos = (body_rect.centerx - self_text.get_width() // 2, body_rect.centery - 25)
                        game.screen.blit(self_text, text_pos)
                        break
                        
            elif game.steps >= game.max_steps:
                death_cause = "Max Steps Reached"
                # Draw timeout warning
                center = collision_rect.center
                pygame.draw.rect(game.screen, (255, 165, 0), collision_rect, 4)
                
                # Draw clock symbol
                pygame.draw.circle(game.screen, (255, 165, 0), center, game.block_size // 3, 2)
                pygame.draw.line(game.screen, (255, 165, 0), center, 
                               (center[0], center[1] - game.block_size // 4), 2)
                pygame.draw.line(game.screen, (255, 165, 0), center, 
                               (center[0] + game.block_size // 5, center[1]), 2)
                
                # Draw "TIME!" text
                font_collision = pygame.font.Font(None, 24)
                time_text = font_collision.render("TIME!", True, (255, 165, 0))
                text_pos = (center[0] - time_text.get_width() // 2, center[1] - 35)
                game.screen.blit(time_text, text_pos)
                
            else:
                death_cause = "Unknown"
                # Draw purple question mark effect
                center = collision_rect.center
                pygame.draw.rect(game.screen, (128, 0, 128), collision_rect, 3)
                
                # Draw question mark
                font_collision = pygame.font.Font(None, 32)
                question_text = font_collision.render("?", True, (255, 255, 255))
                text_pos = (center[0] - question_text.get_width() // 2, center[1] - question_text.get_height() // 2)
                game.screen.blit(question_text, text_pos)
            
            # Display game over information
            font = pygame.font.Font(None, 36)
            font_small = pygame.font.Font(None, 24)
            
            # Create semi-transparent overlay
            overlay = pygame.Surface((game.width, game.height))
            overlay.set_alpha(128)
            overlay.fill((0, 0, 0))
            game.screen.blit(overlay, (0, 0))
            
            # Game over text
            game_over_text = font.render("GAME OVER", True, (255, 255, 255))
            score_text = font.render(f"Score: {game.score}", True, (255, 255, 255))
            steps_text = font.render(f"Steps: {game.steps}", True, (255, 255, 255))
            cause_text = font.render(f"Cause: {death_cause}", True, (255, 0, 0))
            position_text = font_small.render(f"Collision at: ({head_x}, {head_y})", True, (255, 255, 0))
            continue_text = font_small.render("Press SPACE to continue or ESC to quit", True, (200, 200, 200))
            
            # Position texts
            y_offset = game.height // 2 - 90
            game.screen.blit(game_over_text, (game.width // 2 - game_over_text.get_width() // 2, y_offset))
            game.screen.blit(score_text, (game.width // 2 - score_text.get_width() // 2, y_offset + 40))
            game.screen.blit(steps_text, (game.width // 2 - steps_text.get_width() // 2, y_offset + 70))
            game.screen.blit(cause_text, (game.width // 2 - cause_text.get_width() // 2, y_offset + 100))
            game.screen.blit(position_text, (game.width // 2 - position_text.get_width() // 2, y_offset + 130))
            game.screen.blit(continue_text, (game.width // 2 - continue_text.get_width() // 2, y_offset + 160))
            
            pygame.display.flip()
            
            # Wait for user input
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game.close()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            waiting = False
                        elif event.key == pygame.K_ESCAPE:
                            game.close()
                            return
        
        print(f"Game {game_num + 1} - Score: {game.score}, Steps: {game.steps}")
        total_score += game.score
        total_steps += game.steps
        
        if visual and games > 1:
            time.sleep(1)  # Shorter pause between games since we already paused
        
        game.close()
    
    if games > 1:
        avg_score = total_score / games
        avg_steps = total_steps / games
        print(f"\nAverage over {games} games:")
        print(f"  Average Score: {avg_score:.2f}")
        print(f"  Average Steps: {avg_steps:.2f}")

def play_human():
    """Play game as human"""
    print("Playing Snake as human player")
    print("Use arrow keys to control the snake")
    print("Press ESC or close window to quit")
    
    game = SnakeGame()
    game.init_display()
    
    running = True
    while running and not game.game_over:
        action = game.handle_human_input()
        if action is None:
            running = False
            break
        
        state, reward, done = game.step(action)
        game.render()
        
        if done:
            print(f"Game Over! Final Score: {game.score}")
            print("Press any key to continue or close window to quit")
            
            # Wait for key press or window close
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        waiting = False
                        # Reset game for another round
                        game.reset()
    
    game.close()

def demo_random_ai():
    """Demo with random AI"""
    print("Running demo with random AI...")
    
    import random
    
    game = SnakeGame(speed=5)
    game.init_display()
    
    state = game.reset()
    
    while not game.game_over:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.close()
                return
        
        # Random action
        action = random.randint(0, 2)
        state, reward, done = game.step(action)
        game.render()
    
    print(f"Random AI - Score: {game.score}, Steps: {game.steps}")
    time.sleep(3)
    game.close()

def list_models():
    """List available trained models"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models directory found. Train a model first!")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and 'best_snake' in f]
    
    if not model_files:
        print("No trained models found. Train a model first!")
        return
    
    print("Available trained models:")
    for i, model_file in enumerate(sorted(model_files), 1):
        model_path = os.path.join(models_dir, model_file)
        size = os.path.getsize(model_path)
        mtime = time.ctime(os.path.getmtime(model_path))
        print(f"  {i}. {model_file} ({size} bytes, modified: {mtime})")

def main():
    parser = argparse.ArgumentParser(description="Snake Game with Genetic Algorithm AI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train AI using genetic algorithm')
    train_parser.add_argument('--population-size', type=int, default=100,
                            help='Population size (default: 100)')
    train_parser.add_argument('--generations', type=int, default=50,
                            help='Number of generations (default: 50)')
    train_parser.add_argument('--games-per-individual', type=int, default=3,
                            help='Games per individual for evaluation (default: 3)')
    train_parser.add_argument('--mutation-rate', type=float, default=0.1,
                            help='Mutation rate (default: 0.1)')
    train_parser.add_argument('--mutation-strength', type=float, default=0.5,
                            help='Mutation strength (default: 0.5)')
    train_parser.add_argument('--elite-percentage', type=float, default=0.2,
                            help='Elite percentage (default: 0.2)')
    train_parser.add_argument('--crossover-rate', type=float, default=0.8,
                            help='Crossover rate (default: 0.8)')
    train_parser.add_argument('--save-interval', type=int, default=10,
                            help='Save model every N generations (default: 10)')
    train_parser.add_argument('--no-multiprocessing', action='store_true',
                            help='Disable multiprocessing (for debugging)')
    
    # Play command
    play_parser = subparsers.add_parser('play', help='Play with trained AI')
    play_parser.add_argument('model', nargs='?', default='models/best_snake_final.pkl',
                           help='Path to trained model (default: models/best_snake_final.pkl)')
    play_parser.add_argument('--no-visual', action='store_true',
                           help='Run without visual display (faster)')
    play_parser.add_argument('--games', type=int, default=1,
                           help='Number of games to play (default: 1)')
    
    # Human command
    subparsers.add_parser('human', help='Play as human')
    
    # Demo command
    subparsers.add_parser('demo', help='Run demo with random AI')
    
    # List models command
    subparsers.add_parser('list', help='List available trained models')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_ai(args)
    elif args.command == 'play':
        play_with_ai(args.model, visual=not args.no_visual, games=args.games)
    elif args.command == 'human':
        play_human()
    elif args.command == 'demo':
        demo_random_ai()
    elif args.command == 'list':
        list_models()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
