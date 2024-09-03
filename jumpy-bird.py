import pygame
import numpy as np
import random

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
GROUND_LEVEL = HEIGHT - 50
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Bird parameters
BIRD_WIDTH = 30
BIRD_HEIGHT = 30
JUMP_VELOCITY = -7  # Reduced jump strength
GRAVITY = 0.5       # Gravity effect
JUMP_COOLDOWN = 15  # Cooldown time between jumps

# Pipe parameters
PIPE_WIDTH = 80
PIPE_HEIGHT = 500
PIPE_SPEED = 5
PIPE_GAP = 180  # Distance between upper and lower pipe
PIPE_FREQUENCY = 80 # Frequency pipes appear
INITIAL_PIPE_SPEED = PIPE_SPEED  # Store the initial pipe speed to reset after each generation

# Genetic algorithm parameters
POPULATION_SIZE = 100 
MUTATION_RATE = 0.2   
ELITE_PERCENTAGE = 0.2

# Neural network parameters
INPUT_SIZE = 3  # Distance to next pipe, delta y to the center of the pipe gap, bird's velocity
HIDDEN_SIZE = 5
OUTPUT_SIZE = 1  # Jump decision

# Global score to track points
global_score = 0
score_threshold = 100  # Points needed to increase pipe speed
speed_increment = 1  # Amount to increase speed each time the threshold is crossed

class Bird:
    def __init__(self):
        self.x = 100
        self.y = HEIGHT / 2
        self.velocity_y = 0
        self.alive = True
        self.fitness = 0
        self.jump_timer = 0  # Cooldown timer for jumps
        self.brain = self.create_brain()

    def create_brain(self):
        # Initialize a simple neural network with random weights
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

    def jump(self):
        if self.jump_timer == 0:
            self.velocity_y = JUMP_VELOCITY
            self.jump_timer = JUMP_COOLDOWN  # Reset cooldown

    def update(self):
        if not self.alive:
            return

        # Apply gravity to the bird's velocity
        self.velocity_y += GRAVITY
        self.y += self.velocity_y

        # Countdown jump cooldown timer
        if self.jump_timer > 0:
            self.jump_timer -= 1

        # Penalty for flying too high
        if self.y < 50:
            self.fitness -= 1  # Penalize if near the top of the screen

        # Reward for staying in mid-range altitude
        if 150 < self.y < HEIGHT - 150:
            self.fitness += 1  # Reward for staying in a mid-range altitude

        # Boundaries check
        if self.y < 0:
            self.y = 0
            self.velocity_y = 0
        if self.y > HEIGHT - BIRD_HEIGHT:
            self.y = HEIGHT - BIRD_HEIGHT
            self.alive = False

        self.fitness += 1  # General fitness for surviving

    def mutate(self):
        for key in self.brain.keys():
            if np.random.rand() < MUTATION_RATE:
                self.brain[key] += np.random.randn(*self.brain[key].shape)

class Pipe:
    def __init__(self):
        self.x = WIDTH
        self.height = random.randint(100, HEIGHT - 100 - PIPE_GAP)
        self.width = PIPE_WIDTH
        self.gap = PIPE_GAP

    def update(self):
        self.x -= PIPE_SPEED

def spawn_pipe():
    return Pipe()

def main():
    global PIPE_SPEED, global_score, score_threshold, INITIAL_PIPE_SPEED  
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Jumpy Bird Genetic Algorithm")
    clock = pygame.time.Clock()

    generation = 1
    birds = [Bird() for _ in range(POPULATION_SIZE)]
    pipes = []
    high_score = 0  # Highest fitness score achieved

    frame_count = 0
    spawn_counter = PIPE_FREQUENCY

    font = pygame.font.SysFont(None, 36)

    running = True
    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Spawn pipes with a fixed interval
        if frame_count >= spawn_counter:
            pipes.append(spawn_pipe())
            frame_count = 0

        num_dead_birds = sum(1 for bird in birds if not bird.alive)

        for bird in birds:
            if not bird.alive:
                continue

            # Determine the current pipe and calculate inputs
            if len(pipes) > 0:
                next_pipe = pipes[0]
                pipe_distance = next_pipe.x - bird.x
                pipe_center_y = next_pipe.height + PIPE_GAP / 2
                delta_y_to_pipe = bird.y - pipe_center_y

                # Update global score when a bird successfully passes a pipe
                if bird.x > next_pipe.x + PIPE_WIDTH and bird.alive:
                    global_score += 1  # Increase global score
                    high_score = max(global_score, high_score)  # Update high score if necessary

            else:
                pipe_distance = WIDTH
                pipe_center_y = HEIGHT / 2
                delta_y_to_pipe = bird.y - pipe_center_y

            # Collect inputs
            # inputs = np.array([pipe_distance, delta_y_to_pipe, bird.velocity_y, bird.y])
            inputs = np.array([pipe_distance, delta_y_to_pipe, bird.velocity_y])

            # Bird decides to jump or not
            if bird.predict(inputs) > 0:
                bird.jump()

            bird.update()

            # Check for collisions with pipes
            for pipe in pipes:
                if bird.x + BIRD_WIDTH > pipe.x and bird.x < pipe.x + PIPE_WIDTH:
                    if bird.y < pipe.height or bird.y + BIRD_HEIGHT > pipe.height + PIPE_GAP:
                        bird.alive = False

            pygame.draw.rect(screen, GREEN if bird.alive else RED, (bird.x, bird.y, BIRD_WIDTH, BIRD_HEIGHT))

        for pipe in pipes:
            pipe.update()
            pygame.draw.rect(screen, BLACK, (pipe.x, 0, pipe.width, pipe.height))  # Upper pipe
            pygame.draw.rect(screen, BLACK, (pipe.x, pipe.height + PIPE_GAP, pipe.width, HEIGHT))  # Lower pipe

        # Remove pipes that have moved off-screen and transition to the next pipe
        pipes = [pipe for pipe in pipes if pipe.x + PIPE_WIDTH > 0]

        # Increase pipe speed based on global score
        if global_score >= score_threshold:
            PIPE_SPEED += speed_increment
            score_threshold += 100  # Increase threshold for the next speed increment

        # Display generation, bird death information, and high score
        generation_text = font.render(f"Generation: {generation}", True, BLACK)
        death_text = font.render(f"Bird Population: {POPULATION_SIZE-num_dead_birds}/{POPULATION_SIZE}", True, BLACK)
        high_score_text = font.render(f"High Score: {high_score}", True, BLACK)
        screen.blit(generation_text, (10, 10))
        screen.blit(death_text, (10, 50))
        screen.blit(high_score_text, (10, 90))

        if all(not bird.alive for bird in birds):
            # Evolve birds
            birds = evolve_birds(birds)
            pipes = []
            frame_count = 0
            generation += 1

            # Reset global score for the new generation
            global_score = 0

            # Reset pipe speed to a value close to the initial speed
            PIPE_SPEED = INITIAL_PIPE_SPEED + generation // 5  # Increase a little bit with each generation

        pygame.display.flip()
        clock.tick(FPS)
        frame_count += 1

    pygame.quit()

def evolve_birds(birds):
    # Sort birds by fitness
    birds.sort(key=lambda b: b.fitness, reverse=True)
    survivors = birds[:int(ELITE_PERCENTAGE * POPULATION_SIZE)]

    # Crossover and mutation to create new birds
    new_birds = []
    while len(new_birds) < POPULATION_SIZE:
        parent1, parent2 = random.sample(survivors, 2)
        child = crossover(parent1, parent2)
        child.mutate()
        new_birds.append(child)

    return new_birds

def crossover(parent1, parent2):
    child = Bird()
    for key in parent1.brain.keys():
        if np.random.rand() > 0.5:
            child.brain[key] = parent1.brain[key].copy()
        else:
            child.brain[key] = parent2.brain[key].copy()
    return child

if __name__ == "__main__":
    main()
