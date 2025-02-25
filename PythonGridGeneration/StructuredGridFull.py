import numpy as np
import matplotlib.pyplot as plt
from itertools import product
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

    def initialize_algebraic_grid(self):
        # Initial guess: linear interpolation
        self.x_coordinates = self.x_min + self.xi_grid * (self.x_max - self.x_min)
        self.y_coordinates = self.y_min + self.eta_grid * (self.y_max - self.y_min)

        for j, i in product(range(self.num_nodes_y), range(self.num_nodes_x)):
            x = self.x_coordinates[j, i]
            if 2 < x < 3:
                self.y_coordinates[j, i] = self._lower_boundary(x) + self.eta_grid[
                    j, i
                ] * (self._upper_boundary(x) - self._lower_boundary(x))

    def compute_fixed_point(
        self, i: int, j: int, max_iter: int = 1000
    ) -> Tuple[float, float]:

        def fixed_point_function(
            x_interior: float, y_interior: float, x_boundary_current: float
        ) -> float:
            return x_interior + self.derivative_y(x_boundary_current) * (
                y_interior - self._lower_boundary(x_boundary_current)
            )

        x_interior = self.x_coordinates[j + 1, i]
        y_interior = self.y_coordinates[j + 1, i]
        x_boundary_current = self.x_coordinates[j, i]

        for _ in range(max_iter):
            x_boundary_next = fixed_point_function(
                x_interior, y_interior, x_boundary_current
            )
            if abs(x_boundary_current - x_boundary_next) < self.tolerance:
                break
            x_boundary_current = x_boundary_next
        return x_boundary_next, self._lower_boundary(x_boundary_next)

    def apply_boundaries(self, x_coordinates, y_coordinates) -> None:
        # Fixed x boundaries
        x_coordinates[:, 0] = self.x_min
        x_coordinates[:, -1] = self.x_max

        # Curved boundaries
        for i in range(self.num_nodes_x):
            x = self.x_coordinates[0, i]
            if 2 < x < 3:
                y_coordinates[0, i] = self._lower_boundary(x)
                y_coordinates[-1, i] = self._upper_boundary(x)

        # Orthogonal boundary conditions
        for i in range(self.num_nodes_x):
            x = x_coordinates[0, i]
            if 2 < x < 3:
                x_coordinates[0, i], y_coordinates[0, i] = self.compute_fixed_point(
                    i, 0
                )
                x_coordinates[-1, i], y_coordinates[-1, i] = (
                    x_coordinates[0, i],
                    self.y_max - y_coordinates[0, i],
                )

    def calculate_laplace_coefficients(self, i: int, j: int) -> np.ndarray:
        """Calculate coefficients for the Laplace solver."""
        dx_eta = self.x_coordinates[j + 1, i] - self.x_coordinates[j - 1, i]
        dy_eta = self.y_coordinates[j + 1, i] - self.y_coordinates[j - 1, i]
        dx_xi = self.x_coordinates[j, i + 1] - self.x_coordinates[j, i - 1]
        dy_xi = self.y_coordinates[j, i + 1] - self.y_coordinates[j, i - 1]

        alpha = (dx_eta**2 + dy_eta**2) / (4 * self.delta_eta**2)
        beta = (dx_xi * dx_eta + dy_xi * dy_eta) / (4 * self.delta_xi * self.delta_eta)
        gamma = (dx_xi**2 + dy_xi**2) / (4 * self.delta_xi**2)

        denom = 2 * (alpha / self.delta_xi**2 + gamma / self.delta_eta**2)
        return (
            np.array(
                [
                    -beta / (2 * self.delta_eta * self.delta_xi),  # a1
                    gamma / self.delta_eta**2,  # a2
                    beta / (2 * self.delta_eta * self.delta_xi),  # a3
                    alpha / self.delta_xi**2,  # a4
                    alpha / self.delta_xi**2,  # a5
                    beta / (2 * self.delta_eta * self.delta_xi),  # a6
                    gamma / self.delta_eta**2,  # a7
                    -beta / (2 * self.delta_eta * self.delta_xi),  # a8
                ]
            )
            / denom
        )

    def solve_laplace(self):
        iterations = 0
        error_exceeds_tolerance = True
        x_grid_new = self.x_coordinates.copy()
        y_grid_new = self.y_coordinates.copy()

        while error_exceeds_tolerance and iterations < self.max_iterations:
            iterations += 1
            error_exceeds_tolerance = False
            max_error = 0.0

            # Update interior points
            for j, i in product(
                range(1, self.num_nodes_y - 1), range(1, self.num_nodes_x - 1)
            ):
                coeffs = self.calculate_laplace_coefficients(i, j)

                x_neighbors = np.array(
                    [
                        x_grid_new[j - 1, i - 1],
                        x_grid_new[j - 1, i],
                        x_grid_new[j - 1, i + 1],
                        x_grid_new[j, i - 1],
                        x_grid_new[j, i + 1],
                        x_grid_new[j + 1, i - 1],
                        x_grid_new[j + 1, i],
                        x_grid_new[j + 1, i + 1],
                    ]
                )
                y_neighbors = np.array(
                    [
                        y_grid_new[j - 1, i - 1],
                        y_grid_new[j - 1, i],
                        y_grid_new[j - 1, i + 1],
                        y_grid_new[j, i - 1],
                        y_grid_new[j, i + 1],
                        y_grid_new[j + 1, i - 1],
                        y_grid_new[j + 1, i],
                        y_grid_new[j + 1, i + 1],
                    ]
                )

                xNew = np.dot(coeffs, x_neighbors)
                yNew = np.dot(coeffs, y_neighbors)

                x_grid_new[j, i] = xNew
                y_grid_new[j, i] = yNew

            # Apply boundary conditions after each full iteration
            self.apply_boundaries(x_grid_new, y_grid_new)

            for j, i in product(range(self.num_nodes_y), range(self.num_nodes_x)):
                error = np.linalg.norm(
                    [
                        x_grid_new[j, i] - self.x_coordinates[j, i],
                        y_grid_new[j, i] - self.y_coordinates[j, i],
                    ]
                )
                max_error = max(max_error, error)
                if error > self.tolerance:
                    error_exceeds_tolerance = True

            self.x_coordinates = x_grid_new.copy()
            self.y_coordinates = y_grid_new.copy()

            if iterations % 100 == 0:
                print(f"Iteration {iterations}, Max Error: {max_error}")

    def plot_grid(self) -> None:
        plt.figure(figsize=(8, 6))
        for j in range(self.num_nodes_y):
            plt.plot(self.x_coordinates[j, :], self.y_coordinates[j, :], "k-", lw=0.5)
        for i in range(self.num_nodes_x):
            plt.plot(self.x_coordinates[:, i], self.y_coordinates[:, i], "k-", lw=0.5)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Structured Grid using Laplace Equation")
        plt.axis("equal")
        plt.show()


def main():
    myGrid = StructuredMesh(
        num_nodes_x=500, num_nodes_y=100, x_bounds=(0, 5), y_bounds=(0, 1)
    )
    myGrid.initialize_algebraic_grid()
    myGrid.apply_boundaries(myGrid.x_coordinates, myGrid.y_coordinates)
    myGrid.solve_laplace()
    myGrid.plot_grid()


if __name__ == "__main__":
    main()
