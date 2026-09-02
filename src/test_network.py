import torch

from agent.network import QNetwork
from agent.replay_buffer import ReplayBuffer
from env.driving_env import DrivingEnv


env = DrivingEnv()

state_size = env.observation_space.shape[0]
action_size = env.action_space.n


print("State size:", state_size)
print("Action size:", action_size)


network = QNetwork(
    state_size=state_size,
    action_size=action_size,
)


print()
print(network)


obs, info = env.reset(
    seed=42
)


print()
print("Environment observation:")
print(obs)


state = torch.tensor(
    obs,
    dtype=torch.float32,
)


print()
print("Tensor:")
print(state)


with torch.no_grad():

    q_values = network(state)


print()
print("Q-values:")
print(q_values)


action = torch.argmax(
    q_values
).item()


print()
print("Chosen action:")
print(action)


# ------------------------------------------
# REPLAY BUFFER TEST
# ------------------------------------------

buffer = ReplayBuffer(
    capacity=1000
)


obs, info = env.reset(
    seed=42
)


for _ in range(100):

    action = env.action_space.sample()

    (
        next_obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    done = (
        terminated or truncated
    )

    buffer.add(
        state=obs,
        action=action,
        reward=reward,
        next_state=next_obs,
        done=terminated,
    )

    obs = next_obs

    if done:
        obs, info = env.reset()


print()
print(
    "Replay buffer size:",
    len(buffer),
)


batch = buffer.sample(
    batch_size=32,
)


(
    states,
    actions,
    rewards,
    next_states,
    dones,
) = batch


print()
print("Batch shapes:")

print(
    "states:",
    states.shape,
)

print(
    "actions:",
    actions.shape,
)

print(
    "rewards:",
    rewards.shape,
)

print(
    "next states:",
    next_states.shape,
)

print(
    "dones:",
    dones.shape,
)
