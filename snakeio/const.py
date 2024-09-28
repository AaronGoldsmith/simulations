# Screen dimensions
WIDTH, HEIGHT = 800, 600
FPS = 60  # Frames per second

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (255, 0, 0)
BLUE = (0, 200, 200)

# Snake parameters
SNAKE_SIZE = 10
SNAKE_SPEED = 1.5 
ROTATION_SPEED = 15  # Degrees per frame

MAX_STEPS_WITHOUT_FOOD = 100  # How quickly starvation happens
DECREASE_FOOD_RELIANCE = True # Starvation rate decreases over time

DISCOURAGE_REPETITIVE_MOVEMENT = False # Include historical actions in fitness evaluation

# Define a threshold for consecutive turns
TURN_THRESHOLD = 5  # Maximum allowed consecutive turns in the same direction
PENALTY_VALUE = 10  # Fitness penalty per excess consecutive turn

# Body segment spacing on the snake
SEGMENT_SPACING = 7

# Genetic algorithm parameters
POPULATION_SIZE = 20
MUTATION_RATE = 0.3
ELITE_PERCENTAGE = 0.2

# Neural network parameters
INPUT_SIZE = 8  
HIDDEN_SIZE = 16
OUTPUT_SIZE = 1  #  Steering

# Food generation parameters
INITIAL_FOOD_COUNT = 20  
MAX_FOOD_ITEMS = 20 
