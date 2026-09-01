import pygame
import time


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

WIDTH = 1000
HEIGHT = 700
FPS = 60


def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Self-Driving RL Simulator")

    clock = pygame.time.Clock()

    running = True

    while running:

        # ------------------------------------------
        # EVENTS
        # ------------------------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

        # ------------------------------------------
        # KEYBOARD STATE
        # ------------------------------------------

        keys = pygame.key.get_pressed()
        pressed = ""

        if keys[pygame.K_LEFT]:
            pressed = "LEFT"
            print("LEFT")

        if keys[pygame.K_RIGHT]:
            pressed = "RIGHT"
            print("RIGHT")

        if keys[pygame.K_UP]:
            pressed = "UP"
            print("UP")

        if keys[pygame.K_DOWN]:
            pressed = "DOWN"
            print("DOWN")

        # ------------------------------------------
        # DRAWING
        # ------------------------------------------

        screen.fill((35, 40, 45))

        pygame.display.flip()

        # ------------------------------------------
        # FPS
        # ------------------------------------------

        clock.tick(FPS)
        time.sleep(0.05)

    pygame.quit()


if __name__ == "__main__":
    main()