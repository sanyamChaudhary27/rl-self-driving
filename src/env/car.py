import math


class Car:
    def __init__(
        self,
        x=0.0,
        y=0.0,
        heading=0.0,
        velocity=10.0,
        wheelbase=2.5,
    ):
        # Position in world coordinates (metres)
        self.x = x
        self.y = y

        # Direction the car is facing (radians)
        self.heading = heading

        # Forward speed (metres / second)
        self.velocity = velocity

        # Current front-wheel steering angle (radians)
        self.steering_angle = 0.0

        # Distance between front and rear axle (metres)
        self.wheelbase = wheelbase

        # Maximum steering angle: 30 degrees
        self.max_steering_angle = math.radians(30)


    def set_steering(self, steering):
        """
        steering must be between -1 and +1

        -1 = maximum left
         0 = straight
        +1 = maximum right
        """

        steering = max(-1.0, min(1.0, steering))

        self.steering_angle = (
            steering * self.max_steering_angle
        )


    def update(self, dt):
        """
        Move the car forward by dt seconds using
        a simplified kinematic bicycle model.
        """

        # -----------------------------------------
        # Position
        # -----------------------------------------

        self.x += (
            self.velocity
            * math.cos(self.heading)
            * dt
        )

        self.y += (
            self.velocity
            * math.sin(self.heading)
            * dt
        )

        # -----------------------------------------
        # Rotation
        # -----------------------------------------

        yaw_rate = (
            self.velocity
            / self.wheelbase
            * math.tan(self.steering_angle)
        )

        self.heading += yaw_rate * dt

        # Keep heading between -pi and +pi
        self.heading = math.atan2(
            math.sin(self.heading),
            math.cos(self.heading),
        )