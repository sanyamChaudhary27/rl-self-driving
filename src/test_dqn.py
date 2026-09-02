from agent.dqn_agent import DQNAgent
from env.driving_env import DrivingEnv


env = DrivingEnv()

agent = DQNAgent(
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.n,
)


state, info = env.reset(
    seed=42
)


# ==============================================
# COLLECT RANDOM EXPERIENCE
# ==============================================

for _ in range(500):

    action = env.action_space.sample()

    (
        next_state,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    agent.store_transition(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        terminated=terminated,
    )

    state = next_state

    episode_done = (
        terminated or truncated
    )

    if episode_done:
        state, info = env.reset()


print(
    "Replay buffer:",
    len(agent.replay_buffer),
)


# ==============================================
# LEARN
# ==============================================

for step in range(10):

    loss = agent.learn()

    print(
        f"learning step {step}: "
        f"loss={loss:.6f}"
    )
