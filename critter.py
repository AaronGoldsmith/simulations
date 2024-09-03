import pygame
import numpy as np
import random

# Initialize Pygame
pygame.init()

# Window settings
WIDTH, HEIGHT = 800, 600
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Evolving Neural Network Critters")

# Colors
BLACK = (0, 0, 0)
LIGHT_GRAY = (200, 200, 200)
FOV_COLOR = (150, 150, 255)

# Critter settings
NUM_CRITTERS = 20
CRITTER_RADIUS = 20
FOV_ANGLE = 45  # Field of View angle in degrees
FOV_RADIUS = 150  # Radius of the field of view
TURN_ANGLE = 10  # Degrees to turn when avoiding an obstacle
MUTATION_RATE = 0.1  # Mutation rate for GA
LIFESPAN = 500  # Lifespan of each critter in frames

# Neural Network settings
INPUT_SIZE = 2  # Inputs: [obstacle in FOV (binary), distance to closest critter]
HIDDEN_SIZE = 4  # Hidden layer neurons
OUTPUT_SIZE = 2  # Outputs: [turning direction (-1, 0, 1), speed adjustment (-1, 0, 1)]

class Critter:
    def __init__(self, x, y, brain=None):
        self.position = np.array([x, y], dtype=float)
        self.color = BLACK
        self.angle = random.uniform(0, 360)  # Random initial orientation
        self.speed = random.uniform(1, 2)  # Random speed
        self.alive = True
        self.age = 0
        self.fitness = 0
        self.brain = brain if brain else self.create_brain()

    def create_brain(self):
        # Create a simple neural network with a hidden layer
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

    def update_behavior(self, turn_signal, speed_signal):
        # Update turning based on neural network output
        if turn_signal > 0.2:
            self.angle += TURN_ANGLE  # Turn right
        elif turn_signal < -0.2:
            self.angle -= TURN_ANGLE  # Turn left

        # Update speed based on neural network output
        if speed_signal > 0.2:
            self.speed = min(self.speed + 0.1, 3)  # Speed up
        elif speed_signal < -0.2:
            self.speed = max(self.speed - 0.1, 0.5)  # Slow down

    def move(self):
        # Move in the direction of the current angle
        direction = np.array([np.cos(np.radians(self.angle)), -np.sin(np.radians(self.angle))], dtype=float)
        self.position += direction * self.speed

        # Check if critter hits the wall
        if (self.position[0] <= CRITTER_RADIUS or self.position[0] >= WIDTH - CRITTER_RADIUS or
                self.position[1] <= CRITTER_RADIUS or self.position[1] >= HEIGHT - CRITTER_RADIUS):
            self.alive = False

    def update_color_based_on_proximity(self, distance_to_closest):
        # Change color intensity based on proximity to closest critter
        intensity = max(0, 1 - (distance_to_closest / FOV_RADIUS))
        blue_intensity = int(255 * intensity)
        self.color = (0, 0, blue_intensity)

    def draw(self):
        pygame.draw.circle(WINDOW, self.color, self.position.astype(int), CRITTER_RADIUS)

    def draw_fov(self):
        # Draw the field of view as a sector (arc) around the critter
        fov_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.arc(
            fov_surface,
            FOV_COLOR,
            (
                self.position[0] - FOV_RADIUS,
                self.position[1] - FOV_RADIUS,
                2 * FOV_RADIUS,
                2 * FOV_RADIUS,
            ),
            np.radians(self.angle - FOV_ANGLE / 2),
            np.radians(self.angle + FOV_ANGLE / 2),
            FOV_RADIUS
        )
        WINDOW.blit(fov_surface, (0, 0))

    def calculate_fitness(self):
        # Fitness is based on survival time and proximity to other critters
        self.fitness = self.age + (1 / (self.speed + 1))

def detect_obstacles_in_fov(critter, critters):
    """ Detects if there is an obstacle (wall or other critter) in the critter's FoV and calculates distance to closest critter. """
    x, y = critter.position
    obstacle_in_fov = 0
    closest_distance = float('inf')

    # Check if the critter is near the walls
    if x < FOV_RADIUS or x > WIDTH - FOV_RADIUS or y < FOV_RADIUS or y > HEIGHT - FOV_RADIUS:
        obstacle_in_fov = 1

    for other in critters:
        if other == critter or not other.alive:
            continue
        dist = np.linalg.norm(critter.position - other.position)
        if dist < closest_distance:
            closest_distance = dist
        if dist < FOV_RADIUS:
            angle_to_other = np.degrees(np.arctan2(-(other.position[1] - y), other.position[0] - x))
            angle_to_other = (angle_to_other + 360) % 360  # Normalize angle
            critter_angle = (critter.angle + 360) % 360  # Normalize angle
            lower_bound = (critter_angle - FOV_ANGLE / 2) % 360
            upper_bound = (critter_angle + FOV_ANGLE / 2) % 360

            # Check if other critter is in FoV
            if lower_bound <= angle_to_other <= upper_bound or (lower_bound > upper_bound and (angle_to_other >= lower_bound or angle_to_other <= upper_bound)):
                obstacle_in_fov = 1

    # Normalize distance to closest critter
    max_distance = max(WIDTH, HEIGHT)
    normalized_distance = min(closest_distance / max_distance, 1)

    return obstacle_in_fov, normalized_distance

def crossover(parent1, parent2):
    """ Create a child by combining the weights of two parents. """
    child_brain = {}
    for key in parent1.brain:
        child_brain[key] = np.where(np.random.rand(*parent1.brain[key].shape) < 0.5,
                                    parent1.brain[key], parent2.brain[key])
    return child_brain

def mutate(brain):
    """ Randomly mutate the weights of a brain. """
    for key in brain:
        mutation_mask = np.random.rand(*brain[key].shape) < MUTATION_RATE
        brain[key] += mutation_mask * np.random.randn(*brain[key].shape)

def next_generation(critters):
    """ Generate a new population of critters using crossover and mutation. """
    # Sort critters by fitness in descending order
    critters = sorted(critters, key=lambda c: c.fitness, reverse=True)
    num_parents = NUM_CRITTERS // 2
    new_critters = []

    for _ in range(NUM_CRITTERS):
        parent1, parent2 = random.choices(critters[:num_parents], k=2)
        child_brain = crossover(parent1, parent2)
        mutate(child_brain)
        new_critters.append(Critter(random.randint(100, 700), random.randint(100, 500), brain=child_brain))

    return new_critters

def main():
    clock = pygame.time.Clock()
    critters = [Critter(random.randint(100, 700), random.randint(100, 500)) for _ in range(NUM_CRITTERS)]
    generation = 0
    running = True

    while running:
        WINDOW.fill(LIGHT_GRAY)  # Set background to light gray

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        all_dead = all(not critter.alive for critter in critters)

        if all_dead:
            generation += 1
            print(f"Generation {generation} complete. Evolving...")
            critters = next_generation(critters)

        # Update and draw each critter
        for critter in critters:
            if critter.alive:
                obstacle_in_fov, distance_to_closest = detect_obstacles_in_fov(critter, critters)
                turn_signal, speed_signal = critter.predict([obstacle_in_fov, distance_to_closest])
                critter.update_behavior(turn_signal, speed_signal)
                critter.update_color_based_on_proximity(distance_to_closest)
                critter.move()
                critter.calculate_fitness()
                critter.draw_fov()
                critter.draw()
                critter.age += 1

        pygame.display.update()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
