# Self-Driving Car with Deep Reinforcement Learning

## 1. Project Goal

Build a self-driving car agent from scratch using Deep Reinforcement Learning and PyTorch.

The project will begin with a lightweight 2D simulator that can run comfortably on a CPU-only laptop.

The long-term progression is:

2D lane keeping → obstacle avoidance → traffic → continuous vehicle control → sensor-based driving → vision-based driving → realistic simulation.

The primary purpose of the project is not just to obtain a trained model, but to understand how Deep Reinforcement Learning works internally.

---

## 2. Learning Goals

By the end of the project, we should understand and implement:

- Environment/state/action/reward design
- Markov Decision Processes
- Q-values
- Bellman targets
- Temporal Difference learning
- Exploration vs exploitation
- Replay buffers
- Target networks
- Deep Q-Networks
- Reward engineering
- Training stability
- Evaluation
- Checkpointing
- Experiment tracking

Later:

- Policy gradients
- Actor-Critic methods
- PPO
- SAC
- Continuous control
- CNN-based observations
- Domain randomization

---

# 3. Version 0 — Lane Keeping

The first task is intentionally small.

The agent must drive a car along a procedurally generated 2D road without leaving the road.

The car initially travels at a fixed speed.

The agent controls steering only.

This lets us study RL without simultaneously solving acceleration, braking, traffic, perception, and realistic vehicle physics.

---

# 4. Environment

The world contains:

- A car
- A road
- A road center line
- Road boundaries
- Curves
- Checkpoints/progress
- An optional visual renderer

The environment simulation and visual renderer must remain separate.

This means we can train extremely quickly without drawing graphics, while still being able to visually watch evaluation episodes.

---

# 5. Observation / State

Version 0 gives the agent relatively clean information about the car.

Example observation:

    state = [
        lateral_error,
        heading_error,
        road_curvature,
        speed
    ]

Where:

### lateral_error

How far the car is from the center of the road.

Negative:

    left of center

Positive:

    right of center

### heading_error

Difference between:

    direction car is facing

and:

    direction road is heading

### road_curvature

Information about the upcoming shape of the road.

### speed

Initially speed may be fixed, but it remains useful once throttle is introduced.

All observations should eventually be normalized to convenient numeric ranges.

---

# 6. Action Space

Version 0 uses discrete steering.

The agent has five possible actions:

    0 = hard left
    1 = slight left
    2 = straight
    3 = slight right
    4 = hard right

For example:

    steering angles =
    [-25°, -10°, 0°, +10°, +25°]

This makes DQN applicable.

Later we will remove this limitation and use continuous actions:

    steering ∈ [-1, 1]
    throttle ∈ [0, 1]
    brake ∈ [0, 1]

That version will use algorithms such as PPO or SAC instead of basic DQN.

---

# 7. Reward

Reward design must encourage actual driving rather than strange shortcuts.

Initial reward components may include:

### Progress reward

Reward the car for moving forward along the road.

### Centering reward

Small reward for staying reasonably near the road center.

### Heading reward

Reward pointing in approximately the same direction as the road.

### Smoothness penalty

Penalize unnecessary violent steering changes.

### Off-road penalty

A large negative reward when the car leaves the road.

Example conceptual reward:

    reward =
        progress_reward
        + lane_reward
        + heading_reward
        - steering_penalty

If offroad:

    reward -= large_penalty

The exact numbers are experimental parameters, not universal truths.

Reward behavior must be tested visually.

---

# 8. Episode Termination

An episode terminates when:

- The car leaves the road
- The car crashes
- A maximum number of simulation steps is reached

Later episodes may also terminate when the destination is reached.

---

# 9. Vehicle Physics

We should not move the car using fake rules such as:

    x += action

Instead, Version 0 should use a simplified kinematic bicycle model.

Important variables include:

    x
    y
    heading
    velocity
    steering_angle
    wheelbase

The initial model does not need suspension, tire deformation, gears, engines, or aerodynamics.

Those would add computational complexity without helping us understand RL yet.

---

# 10. First Deep RL Algorithm

The first agent will use a Deep Q-Network.

Neural network:

    state
      ↓
    Linear
      ↓
    ReLU
      ↓
    Linear
      ↓
    ReLU
      ↓
    Linear
      ↓
    Q(action 0)
    Q(action 1)
    Q(action 2)
    Q(action 3)
    Q(action 4)

Initial architecture:

    input_size
        ↓
       64
        ↓
       64
        ↓
        5

The final five values are Q-values.

For example:

    Q(s) = [
        2.1,
        3.7,
        5.9,
        4.4,
        1.2
    ]

The greedy action would be:

    action = 2

because it has the highest predicted long-term return.

---

# 11. DQN Components We Will Implement

We will NOT use a ready-made RL implementation.

We will write:

- Q-network
- Target Q-network
- Replay buffer
- Epsilon-greedy exploration
- Bellman target calculation
- Gradient update
- Target-network synchronization
- Checkpoint saving/loading
- Evaluation loop

PyTorch will only provide the deep-learning primitives.

---

# 12. Visualization

We need two different forms of visualization.

## Simulator UI

Pygame will display:

- Road
- Road boundaries
- Car
- Car orientation
- Current steering direction
- Episode
- Reward
- Epsilon
- Speed
- Optional sensor rays

The simulator must also support running without graphics.

Training mode:

    rendering OFF

Evaluation mode:

    rendering ON

During long training sessions, we may occasionally run a visual evaluation episode.

This avoids wasting CPU time rendering thousands of frames that a human cannot watch anyway.

## Training Dashboard

TensorBoard will track values such as:

- Episode reward
- Rolling average reward
- Episode length
- DQN loss
- Epsilon
- Average Q-value
- Maximum Q-value
- Distance travelled
- Number of crashes

This lets us distinguish:

    "the neural network is learning"

from:

    "the car happened to survive once"

---

# 13. Manual Driving Mode

Before training any neural network, the simulator must support keyboard control.

Example:

    ←    steer left
    →    steer right

We will manually drive the car first.

This verifies:

- Physics feels reasonable
- Road generation works
- Collision detection works
- Reward behaves logically
- Observations contain useful information
- Renderer matches the simulation

Only after the environment works manually will we introduce DQN.

---

# 14. Project Structure

    self-driving-drl/
    │
    ├── PROJECT.md
    ├── README.md
    │
    ├── src/
    │   ├── env/
    │   │   ├── car.py
    │   │   ├── road.py
    │   │   ├── driving_env.py
    │   │   └── renderer.py
    │   │
    │   ├── agent/
    │   │   ├── network.py
    │   │   ├── replay_buffer.py
    │   │   └── dqn_agent.py
    │   │
    │   ├── train.py
    │   ├── evaluate.py
    │   └── manual_drive.py
    │
    ├── checkpoints/
    ├── runs/
    └── experiments/

Responsibilities should remain separated.

The environment should not know how DQN works.

DQN should not know how Pygame draws a road.

The renderer should not calculate rewards.

---

# 15. Development Stages

## Stage 0
Project design and environment setup.

## Stage 1
Build car physics.

## Stage 2
Build road generator.

## Stage 3
Build Pygame renderer.

## Stage 4
Drive the car manually.

## Stage 5
Convert simulator into a Gymnasium environment.

## Stage 6
Build PyTorch Q-network.

## Stage 7
Build replay buffer.

## Stage 8
Implement DQN training.

## Stage 9
Add TensorBoard logging.

## Stage 10
Train and visually evaluate the agent.

## Stage 11
Tune rewards and hyperparameters.

## Stage 12
Introduce more complicated roads and randomized starting conditions.

---

# 16. Definition of Success for Version 0

Version 0 is considered successful when:

1. A human can drive the simulator.
2. The environment can run without rendering.
3. DQN trains entirely using our PyTorch implementation.
4. Average reward clearly improves during training.
5. The trained agent can follow roads it did not see in exactly the same starting configuration.
6. Training metrics are visible in TensorBoard.
7. We can visually watch the trained agent drive in Pygame.
8. Models can be saved and loaded.

---

# 17. Future Versions

## V1 — Better Driving

Add:

- Continuous steering
- Throttle
- Braking
- Variable speed
- PPO/SAC

## V2 — Sensors

Replace privileged state information with:

- Ray sensors
- Distance sensors
- LIDAR-like observations

## V3 — Traffic

Add:

- Other cars
- Moving obstacles
- Intersections
- Traffic lights
- Overtaking

## V4 — Vision

Replace engineered observations with camera frames.

Architecture:

    image
      ↓
     CNN
      ↓
    feature vector
      ↓
    RL policy
      ↓
    vehicle control

## V5 — Advanced Simulation

Add:

- Sensor noise
- Different road surfaces
- Weather effects
- Vehicle parameter randomization
- Domain randomization

Eventually migrate to a larger simulator when suitable hardware or cloud compute is available.