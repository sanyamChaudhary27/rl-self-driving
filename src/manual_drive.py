import math

import pygame

from env.car import Car

from env.road import Road


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

def world_to_screen(x, y, camera_x=0.0, camera_y=0.0):
    """
    Convert physics coordinates into Pygame coordinates.

    Physics:
        +y points upward

    Pygame:
        +y points downward
    """
    
    screen_x = WIDTH / 2 + (x - camera_x) * SCALE
    screen_y = HEIGHT / 2 - (y - camera_y) * SCALE

    return int(screen_x), int(screen_y)

def get_car_polygon(car, camera_x, camera_y):
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
        rotated_x = (
            local_x * cos_h
            - local_y * sin_h
        )

        rotated_y = (
            local_x * sin_h
            + local_y * cos_h
        )

        world_x = car.x + rotated_x
        world_y = car.y + rotated_y

        points.append(
            world_to_screen(
                world_x,
                world_y,
                camera_x,
                camera_y,
            )
        )

    return points

def main():
    pygame.init()
    font = pygame.font.Font(None, 28)

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

    road = Road()

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
        # CAMERA
        # ------------------------------------------

        camera_x = car.x
        camera_y = car.y

        # ------------------------------------------
        # DRAW
        # ------------------------------------------

        screen.fill((35, 40, 45))

        # ------------------------------------------
        # ROAD
        # ------------------------------------------

        center_points = []
        left_points = []
        right_points = []

        for x_pixel in range(WIDTH):

            world_x = (
                camera_x
                + (x_pixel - WIDTH / 2) / SCALE
            )

            center_y = road.center_y(world_x)

            left_y = center_y + road.half_width
            right_y = center_y - road.half_width

            center_points.append(
                world_to_screen(
                    world_x,
                    center_y,
                    camera_x,
                    camera_y,
                )
            )

            left_points.append(
                world_to_screen(
                    world_x,
                    left_y,
                    camera_x,
                    camera_y,
                )
            )

            right_points.append(
                world_to_screen(
                    world_x,
                    right_y,
                    camera_x,
                    camera_y,
                )
            )


        pygame.draw.lines(
            screen,
            (180, 180, 180),
            False,
            left_points,
            2,
        )

        pygame.draw.lines(
            screen,
            (180, 180, 180),
            False,
            right_points,
            2,
        )

        pygame.draw.lines(
            screen,
            (100, 100, 100),
            False,
            center_points,
            1,
        )

        car_points = get_car_polygon(car, camera_x, camera_y)

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
            camera_x,
            camera_y,
        )

        pygame.draw.circle(
            screen,
            (255, 255, 255),
            center,
            3,
        )

        # ------------------------------------------
        # TELEMETRY
        # ------------------------------------------

        lateral_error = road.lateral_error(
            car.x,
            car.y,
        )

        off_road = road.is_off_road(
            car.x,
            car.y,
        )

        heading_deg = math.degrees(
            car.heading
        )

        steering_deg = math.degrees(
            car.steering_angle
        )

        road_heading = road.heading(
            car.x
        )

        heading_error = (
            car.heading - road_heading
        )

        heading_error = math.atan2(
            math.sin(heading_error),
            math.cos(heading_error),
        )

        telemetry = [
            f"x: {car.x:.1f} m",
            f"y: {car.y:.1f} m",
            f"speed: {car.velocity:.1f} m/s",
            f"heading: {heading_deg:.1f} deg",
            f"steering: {steering_deg:.1f} deg",
            f"lateral error: {lateral_error:.2f} m",
            f"heading error: {math.degrees(heading_error):.1f} deg",
            f"off road: {off_road}",
            f"FPS: {clock.get_fps():.1f}",
        ]

        for i, text in enumerate(telemetry):
            surface = font.render(
                text,
                True,
                (230, 230, 230),
            )

            screen.blit(
                surface,
                (15, 15 + i * 25),
            )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()