import math
import pygame

from const import WIDTH, HEIGHT

def is_wall_ahead(snake, threshold=50):
    rad_angle = math.radians(snake.angle)
    direction = pygame.math.Vector2(math.cos(rad_angle), math.sin(rad_angle))
    ahead_position = snake.position + direction * threshold

    # Check if ahead_position is within the screen boundaries
    if 0 <= ahead_position.x <= WIDTH and 0 <= ahead_position.y <= HEIGHT:
        return 0.1  # No immediate obstacle

    return 1  # Obstacle within threshold

def swall_in_view(snake, threshold=50, field_of_view=90, num_rays=10):
    """
    Checks if there's a wall ahead of the snake within the specified threshold distance
    by examining a sector (field of view) in the direction the snake is moving.

    Args:
        snake: The snake object containing position (pygame.math.Vector2) and angle (degrees).
        threshold (int, optional): The distance ahead to check for obstacles. Defaults to 50.
        field_of_view (int, optional): The total angle of the sector in degrees. Defaults to 90.
        num_rays (int, optional): The number of rays to cast within the sector. Defaults to 10.

    Returns:
        float: A normalized value between 0 and 1 where:
               - 1.0 indicates an obstacle is very close.
               - Values approaching 0.0 indicate no obstacle within the threshold.
    """
    rad_angle = math.radians(snake.angle)
    central_direction = pygame.math.Vector2(math.cos(rad_angle), math.sin(rad_angle))
    start_pos = snake.position

    # Define walls as lines: each wall is represented by two points (start, end)
    walls = {
        'left': (pygame.math.Vector2(0, 0), pygame.math.Vector2(0, HEIGHT)),
        'right': (pygame.math.Vector2(WIDTH, 0), pygame.math.Vector2(WIDTH, HEIGHT)),
        'top': (pygame.math.Vector2(0, 0), pygame.math.Vector2(WIDTH, 0)),
        'bottom': (pygame.math.Vector2(0, HEIGHT), pygame.math.Vector2(WIDTH, HEIGHT))
    }

    # Calculate the angle increment between rays
    half_fov = field_of_view / 2
    angle_increment = field_of_view / (num_rays - 1) if num_rays > 1 else 0

    closest_distance = threshold  # Initialize with the maximum threshold

    for i in range(num_rays):
        # Calculate the angle for this ray
        ray_angle = rad_angle - math.radians(half_fov) + math.radians(angle_increment) * i
        direction = pygame.math.Vector2(math.cos(ray_angle), math.sin(ray_angle))
        end_pos = start_pos + direction * threshold

        # Check intersection with all walls
        for wall_name, (wall_start, wall_end) in walls.items():
            intersection = get_line_intersection(start_pos, end_pos, wall_start, wall_end)
            if intersection:
                distance = (intersection - start_pos).length()
                if 0 <= distance <= closest_distance:
                    closest_distance = distance

    # Normalize the closest distance
    normalized_distance = 1.0 - (closest_distance / threshold)
    # Clamp the value between 0 and 1
    normalized_distance = max(0.0, min(normalized_distance, 1.0))

    return normalized_distance

def get_line_intersection(p1, p2, p3, p4):
    """
    Calculates the intersection point of two line segments (p1-p2 and p3-p4).
    Returns the point of intersection as pygame.math.Vector2 if it exists, otherwise None.
    """
    # Convert points to coordinates
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    x3, y3 = p3.x, p3.y
    x4, y4 = p4.x, p4.y

    # Calculate denominators
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None  # Parallel lines

    # Calculate numerators
    t_num = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
    u_num = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)

    t = t_num / denom
    u = u_num / denom

    # Check if intersection is within both line segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection_x = x1 + t * (x2 - x1)
        intersection_y = y1 + t * (y2 - y1)
        return pygame.math.Vector2(intersection_x, intersection_y)
    else:
        return None  # Intersection not within the line segments
    