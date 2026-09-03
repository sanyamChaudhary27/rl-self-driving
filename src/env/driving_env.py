import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.car import Car
from env.road import Road


def wrap_angle(angle):
    """
    Wrap an angle into [-pi, pi].
    """
    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


class DrivingEnv(gym.Env):
    """
    First reinforcement-learning environment.

    Task:
        Keep the car on the road while moving in the correct direction.

    Observation:
        [
            normalized lateral error,
            normalized heading error,
            normalized road curvature,
        ]

    Action:
        0 = hard left
        1 = slight left
        2 = straight
        3 = slight right
        4 = hard right
    """

    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        road_amplitude=8.0,
        road_curve_scale=35.0,
        road_half_width=4.0,
    ):
        super().__init__()

        # ------------------------------------------
        # SIMULATION
        # ------------------------------------------

        self.dt = 0.1
        self.max_steps = 500

        self.road = Road(
            amplitude=road_amplitude,
            curve_scale=road_curve_scale,
            half_width=road_half_width,
        )
        self.car = None
        self.steps = 0
        self.previous_steering = 0.0

        # ------------------------------------------
        # ACTION SPACE
        # ------------------------------------------

        self.action_space = spaces.Discrete(5)

        # positive steering = left
        # negative steering = right
        self.steering_actions = np.array(
            [
                1.0,   # hard left
                0.5,   # slight left
                0.0,   # straight
                -0.5,  # slight right
                -1.0,  # hard right
            ],
            dtype=np.float32,
        )

        # ------------------------------------------
        # OBSERVATION SPACE
        # ------------------------------------------

        self.observation_space = spaces.Box(
            low=np.array(
                [-1.0, -1.0, -1.0],
                dtype=np.float32,
            ),
            high=np.array(
                [1.0, 1.0, 1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    # ==========================================================
    # OBSERVATION
    # ==========================================================

    def _get_observation(self):
        lateral_error = self.road.lateral_error(
            self.car.x,
            self.car.y,
        )

        road_heading = self.road.heading(
            self.car.x
        )

        heading_error = wrap_angle(
            self.car.heading - road_heading
        )

        curvature = self.road.curvature(
            self.car.x
        )

        # ------------------------------------------
        # NORMALIZATION
        # ------------------------------------------

        normalized_lateral_error = (
            lateral_error / self.road.half_width
        )

        normalized_heading_error = (
            heading_error / math.pi
        )

        # Curvature values are naturally quite small,
        # so scale them into a more useful range.
        normalized_curvature = curvature * 50.0

        observation = np.array(
            [
                np.clip(
                    normalized_lateral_error,
                    -1.0,
                    1.0,
                ),
                np.clip(
                    normalized_heading_error,
                    -1.0,
                    1.0,
                ),
                np.clip(
                    normalized_curvature,
                    -1.0,
                    1.0,
                ),
            ],
            dtype=np.float32,
        )

        return observation

    # ==========================================================
    # DEBUG INFORMATION
    # ==========================================================

    def _get_info(self):
        lateral_error = self.road.lateral_error(
            self.car.x,
            self.car.y,
        )

        heading_error = wrap_angle(
            self.car.heading - self.road.heading(self.car.x)
        )

        return {
            "x": self.car.x,
            "y": self.car.y,
            "lateral_error": lateral_error,
            "heading_error": heading_error,
            "off_road": self.road.is_off_road(
                self.car.x,
                self.car.y,
            ),
        }

    # ==========================================================
    # RESET
    # ==========================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        self.steps = 0
        self.previous_steering = 0.0

        # Start at a slightly different place
        # each episode.
        start_x = float(
            self.np_random.uniform(
                -20.0,
                20.0,
            )
        )

        road_y = self.road.center_y(
            start_x
        )

        road_heading = self.road.heading(
            start_x
        )

        # Small randomized starting errors.
        lateral_offset = float(
            self.np_random.uniform(
                -0.5,
                0.5,
            )
        )

        heading_offset = float(
            self.np_random.uniform(
                math.radians(-5),
                math.radians(5),
            )
        )

        self.car = Car(
            x=start_x,
            y=road_y + lateral_offset,
            heading=road_heading + heading_offset,
            velocity=10.0,
        )

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    # ==========================================================
    # STEP
    # ==========================================================

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action: {action}"
            )

        self.steps += 1

        steering = float(
            self.steering_actions[action]
        )

        # ------------------------------------------
        # APPLY ACTION
        # ------------------------------------------

        self.car.set_steering(
            steering
        )

        self.car.update(
            self.dt
        )

        # ------------------------------------------
        # MEASURE RESULT
        # ------------------------------------------

        lateral_error = self.road.lateral_error(
            self.car.x,
            self.car.y,
        )

        road_heading = self.road.heading(
            self.car.x
        )

        heading_error = wrap_angle(
            self.car.heading - road_heading
        )

        # ------------------------------------------
        # REWARD
        # ------------------------------------------

        # 1 when perfectly aligned.
        #
        # 0 when sideways.
        #
        # -1 when driving backwards.
        alignment_reward = math.cos(
            heading_error
        )

        # 1 at road center.
        #
        # Approaches 0 near road edge.
        center_reward = max(
            0.0,
            1.0 - abs(lateral_error) / self.road.half_width,
        )

        steering_change = (
            steering - self.previous_steering
        )

        smoothness_penalty = (
            steering_change ** 2
        )

        reward = (
            1.0 * alignment_reward
            + 0.5 * center_reward
            - 0.05 * smoothness_penalty
        )

        # ------------------------------------------
        # TERMINATION
        # ------------------------------------------

        terminated = self.road.is_off_road(
            self.car.x,
            self.car.y,
        )

        if terminated:
            reward -= 10.0

        truncated = (
            self.steps >= self.max_steps
        )

        self.previous_steering = steering

        observation = self._get_observation()
        info = self._get_info()

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

