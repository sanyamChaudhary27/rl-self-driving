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
# VALIDATION SUITE
# ============================================================

VALIDATION_AMPLITUDES = [8.0, 12.0, 16.0, 20.0, 24.0]
VALIDATION_SEED = 1001
VALIDATION_FREQUENCY = 20


def evaluate_policy(
    agent,
    amplitudes=VALIDATION_AMPLITUDES,
    eval_seed=VALIDATION_SEED,
):
    """
    Freeze exploration (epsilon=0) and evaluate agent across a fixed
    battery of road amplitudes using a constant seed.
    """
    total_reward = 0.0
    all_lateral_errors = []
    successes = 0

    for amp in amplitudes:
        eval_env = DrivingEnv(
            road_amplitude=amp,
            randomize_amplitude=False,
        )
        state, info = eval_env.reset(seed=eval_seed)
        terminated = False
        truncated = False
        ep_reward = 0.0

        while not (terminated or truncated):
            action = agent.select_action(
                state=state,
                epsilon=0.0,
            )
            next_state, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            all_lateral_errors.append(abs(info["lateral_error"]))
            state = next_state

        total_reward += ep_reward
        if not terminated:
            successes += 1
        eval_env.close()

    mean_reward = total_reward / len(amplitudes)
    mean_lateral = float(np.mean(all_lateral_errors)) if all_lateral_errors else 0.0

    return successes, mean_reward, mean_lateral


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
    best_val_score = (-1, float("-inf"))

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
            f"{ending}",
            flush=True,
        )

        # ====================================================
        # PERIODIC VALIDATION & BEST MODEL SELECTION
        # ====================================================

        if (episode + 1) % VALIDATION_FREQUENCY == 0:
            val_successes, val_mean_reward, val_mean_lat = evaluate_policy(
                agent=agent,
                amplitudes=VALIDATION_AMPLITUDES,
                eval_seed=VALIDATION_SEED,
            )

            writer.add_scalar(
                "Validation/Successes",
                val_successes,
                episode,
            )
            writer.add_scalar(
                "Validation/MeanReward",
                val_mean_reward,
                episode,
            )
            writer.add_scalar(
                "Validation/MeanLateralError",
                val_mean_lat,
                episode,
            )

            print(
                f"  -> [Validation] suite [8..24]: "
                f"survived={val_successes}/{len(VALIDATION_AMPLITUDES)} | "
                f"mean_reward={val_mean_reward:8.2f} | "
                f"mean_lat={val_mean_lat:.2f}m",
                flush=True,
            )

            val_score = (val_successes, val_mean_reward)
            if val_score > best_val_score:
                best_val_score = val_score

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
                    "  -> Saved new best model across road family "
                    f"(survived={val_successes}/{len(VALIDATION_AMPLITUDES)}, reward={val_mean_reward:.2f})",
                    flush=True,
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
