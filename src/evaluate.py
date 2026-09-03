import argparse
import math
import sys
from pathlib import Path

import pygame
import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent.dqn_agent import DQNAgent
from env.driving_env import DrivingEnv
from manual_drive import (
    WIDTH,
    HEIGHT,
    SCALE,
    world_to_screen,
    get_car_polygon,
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_dqn.pt"
)

EVAL_SEED = 1001

# 1.0 = real simulated speed
# 2.0 = twice as fast
PLAYBACK_SPEED = 2.0


ACTION_NAMES = {
    0: "HARD LEFT",
    1: "SLIGHT LEFT",
    2: "STRAIGHT",
    3: "SLIGHT RIGHT",
    4: "HARD RIGHT",
}


def load_agent(env, checkpoint_path=None):
    if checkpoint_path is None:
        v01_path = (
            PROJECT_ROOT
            / "checkpoints"
            / "v01_random_amplitude"
            / "best_dqn.pt"
        )
        if v01_path.exists():
            checkpoint_path = v01_path
        else:
            checkpoint_path = CHECKPOINT_PATH
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Please train the agent first using train.py."
        )

    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        device="cpu",
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    agent.online_network.load_state_dict(
        checkpoint["online_network"]
    )

    agent.target_network.load_state_dict(
        checkpoint["target_network"]
    )

    # Evaluation mode.
    agent.online_network.eval()
    agent.target_network.eval()

    print(
        "Loaded checkpoint from episode:",
        checkpoint["episode"] + 1,
    )

    return agent


def draw_road(
    screen,
    road,
    camera_x,
    camera_y,
):
    center_points = []
    left_points = []
    right_points = []

    for x_pixel in range(WIDTH):
        world_x = (
            camera_x
            + (x_pixel - WIDTH / 2) / SCALE
        )

        center_y = road.center_y(
            world_x
        )

        left_y = (
            center_y
            + road.half_width
        )

        right_y = (
            center_y
            - road.half_width
        )

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained DQN driver"
    )

    parser.add_argument(
        "--amplitude",
        type=float,
        default=8.0,
        help="Road sine-wave amplitude in metres",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint .pt file",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    env = DrivingEnv(
        road_amplitude=args.amplitude
    )

    agent = load_agent(env, checkpoint_path=args.checkpoint)

    pygame.init()

    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT)
    )

    pygame.display.set_caption(
        "DQN Self-Driving Evaluation"
    )

    font = pygame.font.Font(
        None,
        26,
    )

    clock = pygame.time.Clock()

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    state, info = env.reset(
        seed=EVAL_SEED
    )

    total_reward = 0.0

    terminated = False
    truncated = False

    running = True

    while running and not (
        terminated or truncated
    ):

        # ----------------------------------------------------
        # EVENTS
        # ----------------------------------------------------

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                running = False

        # ====================================================
        # NEURAL NETWORK DECISION
        # ====================================================

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = agent.online_network(
                state_tensor
            )

        action = torch.argmax(
            q_values,
            dim=1,
        ).item()

        # IMPORTANT:
        #
        # epsilon = 0
        #
        # No exploration.
        # This is the learned policy alone.

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward

        # ====================================================
        # CAMERA
        # ====================================================

        camera_x = env.car.x
        camera_y = env.car.y

        # ====================================================
        # DRAW
        # ====================================================

        screen.fill(
            (35, 40, 45)
        )

        draw_road(
            screen,
            env.road,
            camera_x,
            camera_y,
        )

        car_points = get_car_polygon(
            env.car,
            camera_x,
            camera_y,
        )

        pygame.draw.polygon(
            screen,
            (220, 80, 80),
            car_points,
        )

        center = world_to_screen(
            env.car.x,
            env.car.y,
            camera_x,
            camera_y,
        )

        pygame.draw.circle(
            screen,
            (255, 255, 255),
            center,
            3,
        )

        # ====================================================
        # TELEMETRY
        # ====================================================

        heading_error_deg = math.degrees(
            info["heading_error"]
        )

        q_values_list = (
            q_values
            .squeeze(0)
            .cpu()
            .tolist()
        )

        telemetry = [
            "DQN AUTONOMOUS MODE",
            f"Road amplitude: {args.amplitude:.1f} m",
            "",
            f"Step: {env.steps}/{env.max_steps}",
            f"Action: {ACTION_NAMES[action]}",
            f"Reward: {reward:.3f}",
            f"Total reward: {total_reward:.2f}",
            "",
            (
                "Lateral error: "
                f"{info['lateral_error']:.3f} m"
            ),
            (
                "Heading error: "
                f"{heading_error_deg:.2f} deg"
            ),
            "",
            "Q-values:",
            f"Hard L : {q_values_list[0]:.2f}",
            f"Slight L: {q_values_list[1]:.2f}",
            f"Straight: {q_values_list[2]:.2f}",
            f"Slight R: {q_values_list[3]:.2f}",
            f"Hard R : {q_values_list[4]:.2f}",
        ]

        for i, text in enumerate(
            telemetry
        ):
            surface = font.render(
                text,
                True,
                (230, 230, 230),
            )

            screen.blit(
                surface,
                (15, 15 + i * 23),
            )

        pygame.display.flip()

        state = next_state

        # env.dt = simulated seconds per step.
        #
        # 1 / dt gives real-time display FPS.
        display_fps = int(
            PLAYBACK_SPEED
            / env.dt
        )

        clock.tick(
            display_fps
        )

    pygame.quit()
    env.close()

    print()
    print("=" * 50)
    print("EVALUATION FINISHED")
    print("=" * 50)

    print(
        "Steps:",
        env.steps,
    )

    print(
        "Total reward:",
        round(total_reward, 2),
    )

    print(
        "Ending:",
        (
            "OFF ROAD"
            if terminated
            else "TIME LIMIT"
        ),
    )


if __name__ == "__main__":
    main()

