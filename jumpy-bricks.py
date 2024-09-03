import pygame
import numpy as np
import random

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 400
GROUND_LEVEL = HEIGHT - 50
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Agent parameters
AGENT_WIDTH = 20
AGENT_HEIGHT = 40
AGENT_SPEED = 5
JUMP_VELOCITY = -15
GRAVITY = 1

# Obstacle parameters
OBSTACLE_WIDTH = 20
OBSTACLE_HEIGHT = 50
OBSTACLE_SPEED = 5
MIN_SPAWN_INTERVAL = 30  # minimum frames before a new obstacle can spawn
MAX_SPAWN_INTERVAL = 90  # maximum frames before a new obstacle can spawn

# Genetic algorithm parameters
POPULATION_SIZE = 20
MUTATION_RATE = 0.1
ELITE_PERCENTAGE = 0.2

# Neural network parameters
INPUT_SIZE = 3  # Distance to the next obstacle, y velocity, distance to obstacle after next
HIDDEN_SIZE = 5
OUTPUT_SIZE = 1  # Jump decision


class Agent:
    def __init__(self):
        self.x = 50
        self.y = GROUND_LEVEL - AGENT_HEIGHT
        self.velocity_y = 0
        self.alive = True
        self.fitness = 0
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
        if self.y == GROUND_LEVEL - AGENT_HEIGHT:
            self.velocity_y = JUMP_VELOCITY

    def update(self):
        if not self.alive:
            return

        self.velocity_y += GRAVITY
        self.y += self.velocity_y

        if self.y > GROUND_LEVEL - AGENT_HEIGHT:
            self.y = GROUND_LEVEL - AGENT_HEIGHT
            self.velocity_y = 0

        self.fitness += 1

    def mutate(self):
        for key in self.brain.keys():
            if np.random.rand() < MUTATION_RATE:
                self.brain[key] += np.random.randn(*self.brain[key].shape)


class Obstacle:
    def __init__(self):
        self.x = WIDTH
        self.y = GROUND_LEVEL - OBSTACLE_HEIGHT
        self.width = OBSTACLE_WIDTH
        self.height = OBSTACLE_HEIGHT

    def update(self):
        self.x -= OBSTACLE_SPEED


def spawn_obstacle():
    return Obstacle()


def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Genetic Algorithm Slider Game")
    clock = pygame.time.Clock()

    generation = 1
    agents = [Agent() for _ in range(POPULATION_SIZE)]
    obstacles = []

    frame_count = 0
    spawn_counter = random.randint(MIN_SPAWN_INTERVAL, MAX_SPAWN_INTERVAL)

    font = pygame.font.SysFont(None, 36)

    running = True
    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Spawn obstacles with variability
        if frame_count >= spawn_counter:
            obstacles.append(spawn_obstacle())
            frame_count = 0
            spawn_counter = random.randint(MIN_SPAWN_INTERVAL, MAX_SPAWN_INTERVAL)

        num_dead_agents = sum(1 for agent in agents if not agent.alive)

        for agent in agents:
            if not agent.alive:
                continue

            # Check distance to the next obstacle
            if obstacles:
                obstacle_distance = obstacles[0].x - agent.x
                if obstacle_distance < 0:
                    obstacle_distance = WIDTH
            else:
                obstacle_distance = WIDTH
            
            if len(obstacles) > 1:
                next_obstacle_distance = obstacles[1].x - agent.x
            else:
                next_obstacle_distance = 25

            # Agent decides to jump or not
            if agent.predict(np.array([obstacle_distance, next_obstacle_distance, agent.velocity_y])) > 0:
                agent.jump()

            agent.update()

            # Check for collisions with obstacles
            for obstacle in obstacles:
                if agent.x < obstacle.x + obstacle.width and agent.x + AGENT_WIDTH > obstacle.x and agent.y + AGENT_HEIGHT > obstacle.y:
                    agent.alive = False

            pygame.draw.rect(screen, GREEN if agent.alive else RED, (agent.x, agent.y, AGENT_WIDTH, AGENT_HEIGHT))

        for obstacle in obstacles:
            obstacle.update()
            pygame.draw.rect(screen, BLACK, (obstacle.x, obstacle.y, obstacle.width, obstacle.height))

        obstacles = [ob for ob in obstacles if ob.x > 0]

        # Display generation and agent death information
        generation_text = font.render(f"Generation: {generation}", True, BLACK)
        death_text = font.render(f"Agents crashed: {num_dead_agents}/{POPULATION_SIZE}", True, BLACK)
        screen.blit(generation_text, (10, 10))
        screen.blit(death_text, (10, 50))

        if all(not agent.alive for agent in agents):
            # Evolve agents
            agents = evolve_agents(agents)
            obstacles = []
            frame_count = 0
            generation += 1

        pygame.display.flip()
        clock.tick(FPS)
        frame_count += 1

    pygame.quit()


def evolve_agents(agents):
    # Sort agents by fitness
    agents.sort(key=lambda a: a.fitness, reverse=True)
    survivors = agents[:int(ELITE_PERCENTAGE * POPULATION_SIZE)]

    # Crossover and mutation to create new agents
    new_agents = []
    while len(new_agents) < POPULATION_SIZE:
        parent1, parent2 = random.sample(survivors, 2)
        child = crossover(parent1, parent2)
        child.mutate()
        new_agents.append(child)

    return new_agents


def crossover(parent1, parent2):
    child = Agent()
    for key in parent1.brain.keys():
        if np.random.rand() > 0.5:
            child.brain[key] = parent1.brain[key].copy()
        else:
            child.brain[key] = parent2.brain[key].copy()
    return child


if __name__ == "__main__":
    main()
