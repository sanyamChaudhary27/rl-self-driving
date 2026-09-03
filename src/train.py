import os
import random
import sys
from collections import deque
from pathlib import Path

# Ensure 'src' directory is in sys.path even when run from different working directories
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent.dqn_agent import DQNAgent
from env.driving_env import DrivingEnv


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

NUM_EPISODES = 300

LEARNING_RATE = 1e-3
GAMMA = 0.99

BATCH_SIZE = 64
REPLAY_CAPACITY = 50_000

# Don't learn immediately from a tiny replay buffer.
LEARNING_STARTS = 1000

TARGET_UPDATE_FREQUENCY = 1000

# Epsilon-greedy exploration
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 200

DEVICE = "cpu"

CHECKPOINT_DIR = "checkpoints/v01_random_amplitude"
RUN_DIR = "runs/dqn_random_amplitude"


# ============================================================
# EPSILON SCHEDULE
# ============================================================

def get_epsilon(episode):
    """
    Linearly decay epsilon.

    Episode 0:
        epsilon = 1.0

    Episode 200+:
        epsilon = 0.05
    """

    fraction = min(
        episode / EPSILON_DECAY_EPISODES,
        1.0,
    )

    epsilon = (
        EPSILON_START
        + fraction
        * (EPSILON_END - EPSILON_START)
    )

    return epsilon


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    agent,
    episode,
    path,
):
    torch.save(
        {
            "episode": episode,
            "online_network": (
                agent.online_network.state_dict()
            ),
            "target_network": (
                agent.target_network.state_dict()
            ),
            "optimizer": (
                agent.optimizer.state_dict()
            ),
            "training_steps": (
                agent.training_steps
            ),
        },
        path,
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------------

    env = DrivingEnv(
        randomize_amplitude=True,
        amplitude_range=(4.0, 24.0),
    )

    env.action_space.seed(SEED)

    # --------------------------------------------------------
    # AGENT
    # --------------------------------------------------------

    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        replay_capacity=REPLAY_CAPACITY,
        target_update_frequency=(
            TARGET_UPDATE_FREQUENCY
        ),
        device=DEVICE,
    )

    # --------------------------------------------------------
    # TENSORBOARD
    # --------------------------------------------------------

    writer = SummaryWriter(
        log_dir=RUN_DIR
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    recent_rewards = deque(
        maxlen=20
    )

    best_average_reward = float("-inf")

    global_step = 0

    print("=" * 65)
    print("DQN SELF-DRIVING TRAINING")
    print("=" * 65)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"State size: "
        f"{env.observation_space.shape[0]}"
    )

    print(
        f"Action size: "
        f"{env.action_space.n}"
    )

    print(
        f"Episodes: {NUM_EPISODES}"
    )

    print("=" * 65)

    # ========================================================
    # EPISODES
    # ========================================================

    for episode in range(NUM_EPISODES):

        state, info = env.reset(
            seed=SEED + episode
        )

        epsilon = get_epsilon(
            episode
        )

        episode_reward = 0.0
        episode_steps = 0

        losses = []
        lateral_errors = []

        terminated = False
        truncated = False

        # ====================================================
        # ONE EPISODE
        # ====================================================

        while not (
            terminated or truncated
        ):

            # ------------------------------------------------
            # SELECT ACTION
            # ------------------------------------------------

            action = agent.select_action(
                state=state,
                epsilon=epsilon,
            )

            # ------------------------------------------------
            # ENVIRONMENT STEP
            # ------------------------------------------------

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            # ------------------------------------------------
            # STORE EXPERIENCE
            # ------------------------------------------------

            agent.store_transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                terminated=terminated,
            )

            # ------------------------------------------------
            # LEARN
            # ------------------------------------------------

            if (
                len(agent.replay_buffer)
                >= LEARNING_STARTS
            ):

                loss = agent.learn()

                if loss is not None:
                    losses.append(loss)

                    writer.add_scalar(
                        "Training/Loss",
                        loss,
                        global_step,
                    )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            episode_reward += reward

            episode_steps += 1

            lateral_errors.append(
                abs(
                    info[
                        "lateral_error"
                    ]
                )
            )

            state = next_state

            global_step += 1

        # ====================================================
        # EPISODE STATISTICS
        # ====================================================

        recent_rewards.append(
            episode_reward
        )

        average_reward = float(
            np.mean(recent_rewards)
        )

        average_loss = (
            float(np.mean(losses))
            if losses
            else 0.0
        )

        average_lateral_error = (
            float(
                np.mean(
                    lateral_errors
                )
            )
            if lateral_errors
            else 0.0
        )

        # ----------------------------------------------------
        # TENSORBOARD
        # ----------------------------------------------------

        writer.add_scalar(
            "Episode/Reward",
            episode_reward,
            episode,
        )

        writer.add_scalar(
            "Episode/AverageReward20",
            average_reward,
            episode,
        )

        writer.add_scalar(
            "Episode/Length",
            episode_steps,
            episode,
        )

        writer.add_scalar(
            "Episode/Epsilon",
            epsilon,
            episode,
        )

        writer.add_scalar(
            "Episode/AverageLoss",
            average_loss,
            episode,
        )

        writer.add_scalar(
            "Driving/AverageLateralError",
            average_lateral_error,
            episode,
        )

        writer.add_scalar(
            "Training/ReplayBufferSize",
            len(agent.replay_buffer),
            episode,
        )

        writer.add_scalar(
            "Environment/RoadAmplitude",
            info["road_amplitude"],
            episode,
        )

        # ----------------------------------------------------
        # TERMINAL OUTPUT
        # ----------------------------------------------------

        ending = (
            "CRASH"
            if terminated
            else "TIME LIMIT"
        )

        print(
            f"Episode "
            f"{episode + 1:03d}/{NUM_EPISODES} | "
            f"amp={info['road_amplitude']:4.1f}m | "
            f"reward={episode_reward:8.2f} | "
            f"avg20={average_reward:8.2f} | "
            f"steps={episode_steps:3d} | "
            f"eps={epsilon:.3f} | "
            f"loss={average_loss:.4f} | "
            f"lat={average_lateral_error:.2f}m | "
            f"{ending}"
        )

        # ====================================================
        # BEST CHECKPOINT
        # ====================================================

        # Wait until moving average contains enough episodes.
        if (
            len(recent_rewards) == 20
            and average_reward
            > best_average_reward
        ):

            best_average_reward = (
                average_reward
            )

            path = os.path.join(
                CHECKPOINT_DIR,
                "best_dqn.pt",
            )

            save_checkpoint(
                agent=agent,
                episode=episode,
                path=path,
            )

            print(
                "  -> Saved new best model "
                f"(avg20={average_reward:.2f})"
            )

        # ====================================================
        # PERIODIC CHECKPOINT
        # ====================================================

        if (
            (episode + 1) % 50 == 0
        ):

            path = os.path.join(
                CHECKPOINT_DIR,
                f"dqn_episode_{episode + 1}.pt",
            )

            save_checkpoint(
                agent=agent,
                episode=episode,
                path=path,
            )

    # ========================================================
    # FINAL MODEL
    # ========================================================

    save_checkpoint(
        agent=agent,
        episode=NUM_EPISODES - 1,
        path=os.path.join(
            CHECKPOINT_DIR,
            "final_dqn.pt",
        ),
    )

    writer.close()
    env.close()

    print()
    print("=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)

    print(
        "Final model: "
        "checkpoints/final_dqn.pt"
    )

    print(
        "Best model: "
        "checkpoints/best_dqn.pt"
    )


if __name__ == "__main__":
    main()
