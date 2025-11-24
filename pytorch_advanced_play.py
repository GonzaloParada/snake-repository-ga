#!/usr/bin/env python3
"""
Play Snake with Advanced PyTorch-trained AI
"""

import argparse
import time
import pygame
import math
from pytorch_advanced import AdvancedSnakeAI
from snake_game import SnakeGame

def play_advanced_pytorch_ai(model_path: str = "models/best_snake_advanced_final.pth", 
                            games: int = 1, 
                            visual: bool = True):
    """Play Snake with Advanced PyTorch AI"""
    
    print(f"Loading Advanced PyTorch AI model from: {model_path}")
    try:
        ai = AdvancedSnakeAI.load(model_path)
        print("🚀 Advanced PyTorch AI model loaded successfully!")
        print(f"Network parameters: {ai.get_total_params()}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    total_score = 0
    total_steps = 0
    
    for game_num in range(games):
        if games > 1:
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
            pytorch_text = font_small.render("🚀 Advanced PyTorch AI", True, (0, 255, 255))
            continue_text = font_small.render("Press SPACE to continue or ESC to quit", True, (200, 200, 200))
            
            # Position texts
            y_offset = game.height // 2 - 100
            game.screen.blit(game_over_text, (game.width // 2 - game_over_text.get_width() // 2, y_offset))
            game.screen.blit(score_text, (game.width // 2 - score_text.get_width() // 2, y_offset + 40))
            game.screen.blit(steps_text, (game.width // 2 - steps_text.get_width() // 2, y_offset + 70))
            game.screen.blit(cause_text, (game.width // 2 - cause_text.get_width() // 2, y_offset + 100))
            game.screen.blit(position_text, (game.width // 2 - position_text.get_width() // 2, y_offset + 130))
            game.screen.blit(pytorch_text, (game.width // 2 - pytorch_text.get_width() // 2, y_offset + 150))
            game.screen.blit(continue_text, (game.width // 2 - continue_text.get_width() // 2, y_offset + 180))
            
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
            time.sleep(1)
        
        game.close()
    
    if games > 1:
        avg_score = total_score / games
        avg_steps = total_steps / games
        print(f"\nAdvanced PyTorch AI Results:")
        print(f"  Average Score: {avg_score:.2f}")
        print(f"  Average Steps: {avg_steps:.2f}")
        print(f"🚀 Advanced PyTorch Performance Summary")
        
        # Compare with benchmarks
        print(f"\n📊 BENCHMARK COMPARISON:")
        print(f"  🚀 Advanced PyTorch: {avg_score:.2f}")
        print(f"  🥇 NumPy Champion: 28.00")
        print(f"  🔥 Basic PyTorch: 3.70")
        
        if avg_score > 28:
            print(f"🏆 SUCCESS! Advanced PyTorch BEAT NumPy by {avg_score - 28:.2f} points!")
        elif avg_score > 20:
            print(f"🎯 CLOSE! Need {28 - avg_score:.2f} more points to beat NumPy")
        elif avg_score > 10:
            print(f"📈 PROGRESS! Significant improvement over basic PyTorch")
        else:
            print(f"🔄 NEEDS WORK! Still behind basic PyTorch")


def main():
    parser = argparse.ArgumentParser(description='Play Snake with Advanced PyTorch AI')
    parser.add_argument('model', nargs='?', default='models/best_snake_advanced_final.pth',
                       help='Path to Advanced PyTorch model file')
    parser.add_argument('--games', type=int, default=1,
                       help='Number of games to play')
    parser.add_argument('--no-visual', action='store_true',
                       help='Run without visual display (faster)')
    
    args = parser.parse_args()
    
    visual = not args.no_visual
    play_advanced_pytorch_ai(args.model, args.games, visual)


if __name__ == "__main__":
    main()
