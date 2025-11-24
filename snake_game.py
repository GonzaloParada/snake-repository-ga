import pygame
import random
import numpy as np
from enum import Enum
from typing import List, Tuple, Optional

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class SnakeGame:
    def __init__(self, width: int = 640, height: int = 480, block_size: int = 20, speed: int = 30, headless: bool = False):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.speed = speed
        self.headless = headless  # For training without display
        
        # Calculate grid dimensions
        self.grid_width = width // block_size
        self.grid_height = height // block_size
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        
        # Initialize pygame only if not headless
        if not self.headless:
            pygame.init()
        
        self.screen = None
        self.clock = None
        self.font = None
        
        # Game state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the game to initial state and return the initial state"""
        # Snake starts in the middle
        start_x = self.grid_width // 2
        start_y = self.grid_height // 2
        self.snake = [(start_x, start_y)]
        self.direction = Direction.RIGHT
        
        # Place food
        self.food = self._place_food()
        
        # Game state
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0  # Track steps since last food to prevent infinite loops
        self.max_steps_without_food = 200  # Allow 200 steps without eating before timeout
        self.game_over = False
        
        return self.get_state()
    
    def _place_food(self) -> Tuple[int, int]:
        """Place food at a random location not occupied by snake"""
        while True:
            x = random.randint(0, self.grid_width - 1)
            y = random.randint(0, self.grid_height - 1)
            if (x, y) not in self.snake:
                return (x, y)
    
    def get_state(self) -> np.ndarray:
        """Get the current state of the game as a feature vector for the AI"""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Direction vectors
        directions = {
            Direction.UP: (0, -1),
            Direction.RIGHT: (1, 0),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0)
        }
        
        # Current direction
        dir_x, dir_y = directions[self.direction]
        
        # Check danger in each direction (straight, left, right)
        danger_straight = self._is_collision(head_x + dir_x, head_y + dir_y)
        
        # Left direction (relative to current direction)
        left_dir = Direction((self.direction.value - 1) % 4)
        left_x, left_y = directions[left_dir]
        danger_left = self._is_collision(head_x + left_x, head_y + left_y)
        
        # Right direction (relative to current direction)
        right_dir = Direction((self.direction.value + 1) % 4)
        right_x, right_y = directions[right_dir]
        danger_right = self._is_collision(head_x + right_x, head_y + right_y)
        
        # Direction booleans
        dir_up = self.direction == Direction.UP
        dir_right = self.direction == Direction.RIGHT
        dir_down = self.direction == Direction.DOWN
        dir_left = self.direction == Direction.LEFT
        
        # Food direction
        food_up = food_y < head_y
        food_down = food_y > head_y
        food_left = food_x < head_x
        food_right = food_x > head_x
        
        state = np.array([
            # Danger
            danger_straight,
            danger_left,
            danger_right,
            
            # Current direction
            dir_up,
            dir_right,
            dir_down,
            dir_left,
            
            # Food direction
            food_up,
            food_down,
            food_left,
            food_right
        ], dtype=np.float32)
        
        return state
    
    def _is_collision(self, x: int, y: int) -> bool:
        """Check if position (x, y) would result in collision"""
        # Wall collision
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True
        
        # Self collision
        if (x, y) in self.snake:
            return True
        
        return False
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Take a step in the game
        action: 0 = straight, 1 = turn right, 2 = turn left
        Returns: (new_state, reward, done)
        """
        self.steps += 1
        self.steps_since_food += 1
        
        # Update direction based on action
        if action == 1:  # Turn right
            self.direction = Direction((self.direction.value + 1) % 4)
        elif action == 2:  # Turn left
            self.direction = Direction((self.direction.value - 1) % 4)
        # action == 0 means go straight (no direction change)
        
        # Move snake
        head_x, head_y = self.snake[0]
        directions = {
            Direction.UP: (0, -1),
            Direction.RIGHT: (1, 0),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0)
        }
        dx, dy = directions[self.direction]
        new_head = (head_x + dx, head_y + dy)
        
        # Check collision
        if self._is_collision(new_head[0], new_head[1]):
            self.game_over = True
            return self.get_state(), -10, True  # Negative reward for collision
        
        # Add new head
        self.snake.insert(0, new_head)
        
        # Check if food eaten
        reward = 0
        if new_head == self.food:
            self.score += 1
            reward = 10  # Positive reward for eating food
            self.food = self._place_food()
            self.steps_since_food = 0  # Reset steps since food
        else:
            # Remove tail if no food eaten
            self.snake.pop()
            reward = 0  # Small positive reward for staying alive
        
        # Check if too many steps without progress
        if self.steps_since_food > self.max_steps_without_food:
            self.game_over = True
            return self.get_state(), -5, True
        
        return self.get_state(), reward, self.game_over
    
    def init_display(self):
        """Initialize the display for visual gameplay"""
        if self.screen is None and not self.headless:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Snake Game - Genetic Algorithm")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
    
    def render(self):
        """Render the game visually"""
        if self.screen is None or self.headless:
            return
        
        self.screen.fill(self.BLACK)
        
        # Draw snake with distinctive head
        for i, segment in enumerate(self.snake):
            x, y = segment
            rect = pygame.Rect(x * self.block_size, y * self.block_size, 
                             self.block_size, self.block_size)
            
            if i == 0:  # Head
                # Draw head with different color and eyes
                pygame.draw.rect(self.screen, (0, 200, 0), rect)  # Bright green head
                pygame.draw.rect(self.screen, (0, 255, 0), rect, 2)  # Bright border
                
                # Draw eyes based on direction
                eye_size = 3
                eye_offset = self.block_size // 4
                
                if self.direction == Direction.UP:
                    left_eye = (x * self.block_size + eye_offset, y * self.block_size + eye_offset)
                    right_eye = (x * self.block_size + self.block_size - eye_offset, y * self.block_size + eye_offset)
                elif self.direction == Direction.DOWN:
                    left_eye = (x * self.block_size + eye_offset, y * self.block_size + self.block_size - eye_offset)
                    right_eye = (x * self.block_size + self.block_size - eye_offset, y * self.block_size + self.block_size - eye_offset)
                elif self.direction == Direction.LEFT:
                    left_eye = (x * self.block_size + eye_offset, y * self.block_size + eye_offset)
                    right_eye = (x * self.block_size + eye_offset, y * self.block_size + self.block_size - eye_offset)
                else:  # RIGHT
                    left_eye = (x * self.block_size + self.block_size - eye_offset, y * self.block_size + eye_offset)
                    right_eye = (x * self.block_size + self.block_size - eye_offset, y * self.block_size + self.block_size - eye_offset)
                
                # Draw eyes
                pygame.draw.circle(self.screen, self.WHITE, left_eye, eye_size)
                pygame.draw.circle(self.screen, self.WHITE, right_eye, eye_size)
                pygame.draw.circle(self.screen, self.BLACK, left_eye, eye_size - 1)
                pygame.draw.circle(self.screen, self.BLACK, right_eye, eye_size - 1)
                
            else:  # Body
                # Draw body segments with gradient effect
                body_color = (0, 150 - min(i * 5, 50), 0)  # Darker green for body, gradient effect
                pygame.draw.rect(self.screen, body_color, rect)
                pygame.draw.rect(self.screen, (0, 100, 0), rect, 1)  # Dark border
        
        # Draw food with animation
        food_x, food_y = self.food
        food_rect = pygame.Rect(food_x * self.block_size, food_y * self.block_size, 
                               self.block_size, self.block_size)
        
        # Pulsing food effect
        import time
        pulse = int(abs(time.time() * 5) % 2)
        food_color = (255, 100 + pulse * 50, 100 + pulse * 50)
        
        pygame.draw.rect(self.screen, food_color, food_rect)
        pygame.draw.rect(self.screen, self.RED, food_rect, 2)
        
        # Draw a small cross in the food
        center_x = food_x * self.block_size + self.block_size // 2
        center_y = food_y * self.block_size + self.block_size // 2
        cross_size = self.block_size // 4
        pygame.draw.line(self.screen, self.WHITE, 
                        (center_x - cross_size, center_y), 
                        (center_x + cross_size, center_y), 2)
        pygame.draw.line(self.screen, self.WHITE, 
                        (center_x, center_y - cross_size), 
                        (center_x, center_y + cross_size), 2)
        
        # Draw score and steps
        if self.font:
            score_text = self.font.render(f"Score: {self.score}", True, self.WHITE)
            steps_text = self.font.render(f"Steps: {self.steps}", True, self.WHITE)
            self.screen.blit(score_text, (10, 10))
            self.screen.blit(steps_text, (10, 35))
        
        pygame.display.flip()
        if self.clock:
            self.clock.tick(self.speed)
    
    def handle_human_input(self) -> Optional[int]:
        """Handle human keyboard input and return action"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.direction = Direction.DOWN
                elif event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
        return 0  # Continue straight (direction already set by input)
    
    def close(self):
        """Close the game display"""
        if not self.headless and self.screen is not None:
            pygame.quit()
        self.screen = None
        self.clock = None
        self.font = None

if __name__ == "__main__":
    # Test the game with human input
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
            pygame.time.wait(2000)
            running = False
    
    game.close()
