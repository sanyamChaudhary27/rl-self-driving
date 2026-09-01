import math


class Road:
    def __init__(
        self,
        amplitude=8.0,
        curve_scale=35.0,
        half_width=4.0,
    ):
        self.amplitude = amplitude
        self.curve_scale = curve_scale
        self.half_width = half_width


    def center_y(self, x):
        """
        Y coordinate of the road center at world position x.
        """

        return (
            self.amplitude
            * math.sin(x / self.curve_scale)
        )


    def slope(self, x):
        """
        dy/dx of the road center line.
        """

        return (
            self.amplitude
            / self.curve_scale
            * math.cos(x / self.curve_scale)
        )


    def heading(self, x):
        """
        Direction the road is pointing at position x.
        """

        return math.atan(self.slope(x))


    def second_derivative(self, x):
        """
        d²y/dx².
        """

        return (
            -self.amplitude
            / (self.curve_scale ** 2)
            * math.sin(x / self.curve_scale)
        )


    def curvature(self, x):
        """
        Curvature:

        k = y'' / (1 + y'^2)^(3/2)
        """

        dy = self.slope(x)
        ddy = self.second_derivative(x)

        return ddy / ((1 + dy**2) ** 1.5)


    def lateral_error(self, x, y):
        """
        Approximate signed distance from road center.

        Negative = one side
        Positive = other side
        """

        return y - self.center_y(x)


    def is_off_road(self, x, y):
        return (
            abs(self.lateral_error(x, y))
            > self.half_width
        )