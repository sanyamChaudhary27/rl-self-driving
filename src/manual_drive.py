import math

import pygame

from env.car import Car


WIDTH = 1000
HEIGHT = 700
FPS = 60

# Conversion between our physics world and the screen.
#
# Physics:
#     metres
#
# Pygame:
#     pixels
#
SCALE = 8


def world_to_screen(x, y):
    """
    Convert physics coordinates into Pygame coordinates.

    Physics:
        +y points upward

    Pygame:
        +y points downward
    """

    screen_x = WIDTH / 2 + x * SCALE
    screen_y = HEIGHT / 2 - y * SCALE

    return int(screen_x), int(screen_y)


def get_car_polygon(car):
    """
    Create four corners representing the car.
    """

    length = 4.5
    width = 1.8

    local_points = [
        (+length / 2, +width / 2),
        (+length / 2, -width / 2),
        (-length / 2, -width / 2),
        (-length / 2, +width / 2),
    ]

    points = []

    cos_h = math.cos(car.heading)
    sin_h = math.sin(car.heading)

    for local_x, local_y in local_points:

        # Rotate local point by car heading
        rotated_x = (
            local_x * cos_h
            - local_y * sin_h
        )

        rotated_y = (
            local_x * sin_h
            + local_y * cos_h
        )

        # Move it to the car's world position
        world_x = car.x + rotated_x
        world_y = car.y + rotated_y

        # Convert metres -> pixels
        points.append(
            world_to_screen(world_x, world_y)
        )

    return points


def main():

    pygame.init()

    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT)
    )

    pygame.display.set_caption(
        "Self-Driving RL Simulator"
    )

    clock = pygame.time.Clock()

    car = Car(
        x=-40.0,
        y=0.0,
        heading=0.0,
        velocity=10.0,
    )

    running = True

    while running:

        # ------------------------------------------
        # TIME
        # ------------------------------------------

        dt = clock.tick(FPS) / 1000.0

        # ------------------------------------------
        # EVENTS
        # ------------------------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                running = False

        # ------------------------------------------
        # KEYBOARD
        # ------------------------------------------

        keys = pygame.key.get_pressed()

        steering = 0.0

        if keys[pygame.K_LEFT]:
            steering = 1.0

        if keys[pygame.K_RIGHT]:
            steering = -1.0

        car.set_steering(steering)

        # ------------------------------------------
        # PHYSICS
        # ------------------------------------------

        car.update(dt)

        # ------------------------------------------
        # DRAW
        # ------------------------------------------

        screen.fill((35, 40, 45))

        car_points = get_car_polygon(car)

        pygame.draw.polygon(
            screen,
            (220, 80, 80),
            car_points,
        )

        # Draw a small circle showing the
        # car's mathematical centre.
        center = world_to_screen(
            car.x,
            car.y,
        )

        pygame.draw.circle(
            screen,
            (255, 255, 255),
            center,
            3,
        )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()