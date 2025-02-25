import numpy as np
from typing import Tuple


class StructuredMesh:
    def __init__(
        self,
        num_nodes_x: int,
        num_nodes_y: int,
        x_bounds: Tuple[float, float],
        y_bounds: Tuple[float, float],
        tolerance: float = 1e-6,
        max_iterations: int = 5000,
    ):
        self.num_nodes_x = num_nodes_x
        self.num_nodes_y = num_nodes_y
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        # Computational grid
        xi = np.linspace(0, 1, num_nodes_x)
        eta = np.linspace(0, 1, num_nodes_y)
        self.xi_grid, self.eta_grid = np.meshgrid(xi, eta)
        self.delta_xi = 1 / (num_nodes_x - 1)
        self.delta_eta = 1 / (num_nodes_y - 1)

        # Initialize physical coordinates
        self.x_coordinates = np.zeros_like(self.xi_grid)
        self.y_coordinates = np.zeros_like(self.eta_grid)

    def _upper_boundary(self, x):
        return self.y_max - 0.17 * np.sin((x - 2) * np.pi) if 2 < x < 3 else self.y_min

    def _lower_boundary(self, x):
        return 0.17 * np.sin((x - 2) * np.pi) if 2 < x < 3 else self.y_max

    def derivative_y(self, x):
        return np.pi * 0.17 * np.cos((x - 2) * np.pi)
