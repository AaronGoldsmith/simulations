# This version discourages "looping behavior"
import pygame
import numpy as np
import random
import math

from utils import is_wall_ahead, wall_in_view
from const import *

# Initialize pygame
pygame.init()

class Snake:
    def __init__(self, initial_positions, generation):
        # Assign random position ensuring no overlap
        self.position = pygame.math.Vector2(self.get_random_position(initial_positions))
        self.angle = random.uniform(0, 360)  # Angle in degrees

        # Initialize velocity in the direction of the angle
        rad_angle = math.radians(self.angle)
        self.velocity = pygame.math.Vector2(math.cos(rad_angle), math.sin(rad_angle)) * SNAKE_SPEED

        self.length = 3  # Start with length 3 (head + 2 segments)
        self.positions = [self.position.copy()]  # List to store the body segments
        self.alive = True
        self.fitness = 0
        self.generation = generation
        self.brain = self.create_brain()
        self.steps_since_last_food = 0  # Steps since the snake last ate
         
         # Initialize action history
        self.action_history = []
        self.max_history_length = 20

    def get_random_position(self, existing_positions):
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.randint(SNAKE_SIZE * 2, WIDTH - SNAKE_SIZE * 2)
            y = random.randint(SNAKE_SIZE * 2, HEIGHT - SNAKE_SIZE * 2)
            overlap = False
            for pos in existing_positions:
                if abs(x - pos[0]) < SNAKE_SIZE * 4 and abs(y - pos[1]) < SNAKE_SIZE * 4:
                    overlap = True
                    break
            if not overlap:
                existing_positions.append((x, y))
                return x, y
        existing_positions.append((x, y))
        return x, y

    def create_brain(self):
        # Initialize neural network with random weights
        return {
            'w1': np.random.randn(INPUT_SIZE, HIDDEN_SIZE),
            'b1': np.random.randn(HIDDEN_SIZE),
            'w2': np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE),
            'b2': np.random.randn(OUTPUT_SIZE),
        }

    def predict(self, inputs):
        # Forward pass through the network
        hidden = np.tanh(np.dot(inputs, self.brain['w1']) + self.brain['b1'])
        output = np.tanh(np.dot(hidden, self.brain['w2']) + self.brain['b2'])
        return output

    def decide(self, inputs):
        output = self.predict(inputs)
        steering = output[0]  # Range [-1, 1]
        return steering

    def mutate(self):
        # Mutate the neural network weights with some probability
        for key in self.brain.keys():
            if np.random.rand() < MUTATION_RATE:
                mutation_array = np.random.randn(*self.brain[key].shape) * 0.1
                self.brain[key] += mutation_array

    def apply_behavior(self, steering_input):
        # Rotate snake based on steering input
        self.angle += steering_input * ROTATION_SPEED
        self.angle %= 360  # Keep angle within [0, 360]
        
        # Log the steering input to action history
        self.action_history.append(steering_input)
        if len(self.action_history) > self.max_history_length:
            self.action_history.pop(0)

    def evaluate_fitness(self):
        """
        Evaluate and update the snake's fitness score based on its action history.
        Penalizes repetitive turning to discourage looping behaviors.
        """
        # Count consecutive turns in the same direction
        consecutive_turns = 0
        last_turn_direction = None  # 'left', 'right', or None
        
        # Analyze the action history from the most recent actions
        for steering_input in reversed(self.action_history):
            if steering_input < 0:
                current_turn = 'left'
            elif steering_input > 0:
                current_turn = 'right'
            else:
                current_turn = 'straight'
            
            if current_turn == last_turn_direction and current_turn != 'straight':
                consecutive_turns += 1
            else:
                break  # Stop counting when the direction changes or the snake goes straight
            
            last_turn_direction = current_turn
        
        
        if consecutive_turns > TURN_THRESHOLD:
            # Calculate excess turns beyond the threshold
            excess_turns = consecutive_turns - TURN_THRESHOLD
            # Apply a fitness penalty proportional to the excess turns
            self.fitness -= PENALTY_VALUE * excess_turns

    def update(self, food_list, snakes):
        if not self.alive:
            # We don't want to update or recalculate fitness after the snake dies
            return

        # Update velocity based on angle
        rad_angle = math.radians(self.angle)
        self.velocity = pygame.math.Vector2(math.cos(rad_angle), math.sin(rad_angle)) * SNAKE_SPEED

        # Update position
        self.position += self.velocity

        # Increment steps since last food
        self.steps_since_last_food += 1

        # Check for starvation
        steps_allowed_without_food = (MAX_STEPS_WITHOUT_FOOD + self.generation
              if DECREASE_FOOD_RELIANCE else MAX_STEPS_WITHOUT_FOOD)
        
        if self.steps_since_last_food > steps_allowed_without_food:
            self.alive = False
            self.fitness -= 50  # Penalize for starving
            return

        # Add new position if the snake has moved enough
        if len(self.positions) == 0 or self.position.distance_to(self.positions[0]) > SEGMENT_SPACING:
            self.positions.insert(0, self.position.copy())

        # Keep the snake's length
        while len(self.positions) > self.length:
            self.positions.pop()

        # Boundary checking
        if self.position.x < 0 or self.position.x > WIDTH or self.position.y < 0 or self.position.y > HEIGHT:
            self.alive = False
            self.fitness -= 50  # Penalize for dying
            return

        # Increment fitness for surviving
        self.fitness += 0.1  # Smaller increment to balance fitness
        self.evaluate_fitness()
        
        # self.evaluate_fitness() # not implemented

    def grow(self):
        self.length += 1  # Increase length
        self.fitness += 10  # Reward for eating food
        self.steps_since_last_food = 0  # Reset starvation counter

    def check_collision(self, food_list):
        for food in food_list:
            if self.position.distance_to(food.position) < SNAKE_SIZE:
                self.grow()
                food_list.remove(food)  # Remove food from the list
                break  # Only one food can be eaten per update

    def check_collision_with_snakes(self, other_snakes):
        for other_snake in other_snakes:
            for segment in other_snake.positions:
                if self.position.distance_to(segment) < SNAKE_SIZE / 2:
                    self.alive = False
                    self.fitness -= 50  # Penalize for colliding with another snake
                    return

    def draw(self, screen):
        # Draw the snake's body segments
        for i, pos in enumerate(self.positions):
            # Fade the color for older segments
            color_intensity = max(50, 255 - i * 15)
            segment_color = (0, color_intensity, 0)
            pygame.draw.circle(screen, segment_color, (int(pos.x), int(pos.y)), SNAKE_SIZE // 2)

class Food:
    def __init__(self):
        self.position = pygame.math.Vector2(0, 0)
        self.respawn()

    def respawn(self):
        self.position.x = random.randint(SNAKE_SIZE * 2, WIDTH - SNAKE_SIZE * 2)
        self.position.y = random.randint(SNAKE_SIZE * 2, HEIGHT - SNAKE_SIZE * 2)

    def draw(self, screen):
        pygame.draw.circle(screen, RED, (int(self.position.x), int(self.position.y)), SNAKE_SIZE // 2)

def crossover(parent1, parent2, generation):
    child = Snake([], generation)
    child.brain = {}
    for key in parent1.brain.keys():
        child.brain[key] = np.copy(parent1.brain[key])
        mask = np.random.rand(*child.brain[key].shape) > 0.5
        child.brain[key][mask] = parent2.brain[key][mask]
    return child

def evolve_snakes(snakes, generation):
    snakes.sort(key=lambda x: x.fitness, reverse=True)
    elites = snakes[:int(ELITE_PERCENTAGE * POPULATION_SIZE)]

    new_snakes = []
    existing_positions = []
    while len(new_snakes) < POPULATION_SIZE:
        parent1, parent2 = random.sample(elites, 2)
        child = crossover(parent1, parent2, generation)
        child.mutate()
        child.position = pygame.math.Vector2(child.get_random_position(existing_positions))
        child.positions = [child.position.copy() for _ in range(child.length)]
        # Initialize angle and velocity
        child.angle = random.uniform(0, 360)
        rad_angle = math.radians(child.angle)
        child.velocity = pygame.math.Vector2(math.cos(rad_angle), math.sin(rad_angle)) * SNAKE_SPEED
        child.alive = True
        child.steps_since_last_food = 0  # Reset starvation counter
        new_snakes.append(child)

    return new_snakes

def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI Snake Genetic Algorithm")
    clock = pygame.time.Clock()

    generation = 1
    existing_positions = []
    snakes = [Snake(existing_positions,generation) for _ in range(POPULATION_SIZE)]

    # Initialize with 20 food items
    food_list = [Food() for _ in range(INITIAL_FOOD_COUNT)]

    font = pygame.font.SysFont(None, 24)

    running = True
    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Spawn new food if below the maximum allowed
        if len(food_list) < MAX_FOOD_ITEMS:
            food_list.append(Food())

        # Update and draw food
        for food in food_list:
            food.draw(screen)

        # Update and draw snakes
        alive_snakes = 0
        for snake in snakes:
            if snake.alive:
                alive_snakes += 1

                # Inputs for the snake
                if food_list:
                    closest_food = min(food_list, key=lambda f: snake.position.distance_to(f.position))
                    distance_to_food = snake.position.distance_to(closest_food.position)
                    angle_to_food = math.degrees(math.atan2(closest_food.position.y - snake.position.y,
                                                          closest_food.position.x - snake.position.x))
                    angle_diff_food = (angle_to_food - snake.angle + 360) % 360
                    if angle_diff_food > 180:
                        angle_diff_food -= 360
                else:
                    distance_to_food = math.hypot(WIDTH, HEIGHT)
                    angle_diff_food = 0

                # Find the closest other snake
                other_snakes = [s for s in snakes if s != snake and s.alive]
                if other_snakes:
                    closest_snake = min(other_snakes, key=lambda s: snake.position.distance_to(s.position))
                    distance_to_snake = snake.position.distance_to(closest_snake.position)
                    angle_to_snake = math.degrees(math.atan2(closest_snake.position.y - snake.position.y,
                                                        closest_snake.position.x - snake.position.x))
                    angle_diff_snake = (angle_to_snake - snake.angle + 360) % 360
                    if angle_diff_snake > 180:
                        angle_diff_snake -= 360
                    norm_distance_to_snake = distance_to_snake / math.hypot(WIDTH, HEIGHT)
                    norm_angle_diff_snake = angle_diff_snake / 180
                else:
                    norm_distance_to_snake = 1.0
                    norm_angle_diff_snake = 0.0

                wall_ahead = wall_in_view(snake) # normalized value
                # wall_ahead = is_wall_ahead(snake, threshold=50)

                # TODO: add an input for 
                # Normalize inputs
                inputs = np.array([
                    distance_to_food / math.hypot(WIDTH, HEIGHT),  # Normalized distance to food
                    angle_diff_food / 180,                        # Normalized angle difference to food
                    math.sin(math.radians(snake.angle)),
                    math.cos(math.radians(snake.angle)),
                    snake.length / 20.0,                           # Normalized length
                    norm_distance_to_snake,                        # Normalized distance to closest snake
                    norm_angle_diff_snake,                         # Normalized angle difference to closest snake
                    wall_ahead                                  # Normalized input for obstacle ahead
                ])

                # Get steering input from the neural network
                steering_input = snake.decide(inputs)

                # Apply behavior and update
                snake.apply_behavior(steering_input)
                snake.update(food_list, snakes)

                # Check collisions
                snake.check_collision(food_list)
                snake.check_collision_with_snakes(other_snakes)

                # Draw snake
                snake.draw(screen)

        # Display generation and population stats
        generation_text = font.render(f"Generation: {generation}", True, BLACK)
        alive_text = font.render(f"Alive: {alive_snakes}/{POPULATION_SIZE}", True, BLACK)
        screen.blit(generation_text, (10, 10))
        screen.blit(alive_text, (10, 30))

        pygame.display.flip()
        clock.tick(FPS)  # Control the game speed

        # If all snakes are dead, evolve the population
        if alive_snakes == 0:
            generation += 1
            print(f"Generation {generation} complete.")

            # Reset existing positions for new generation
            existing_positions = []
            snakes = evolve_snakes(snakes, generation)

            # Reset food list for the new generation
            food_list = [Food() for _ in range(INITIAL_FOOD_COUNT)]

    pygame.quit()

if __name__ == "__main__":
    main()
