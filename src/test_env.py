from gymnasium.utils.env_checker import check_env
from env.driving_env import DrivingEnv

env = DrivingEnv()

print("Checking Gymnasium API...")

check_env(
    env,
    skip_render_check=True,
)

print("Environment passed!")

obs, info = env.reset(
    seed=42
)

print()
print("Initial observation:")
print(obs)

print()
print("Initial info:")
print(info)

print()
print("Taking random actions...")

for step in range(20):

    action = env.action_space.sample()

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    print(
        f"{step=:02d} | "
        f"{action=} | "
        f"reward={reward:6.3f} | "
        f"lateral={info['lateral_error']:6.3f}"
    )

    if terminated or truncated:
        print("Episode ended.")
        obs, info = env.reset()

