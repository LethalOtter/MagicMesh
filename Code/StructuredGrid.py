import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class StructuredMesh:
    def __init__(self, iMax, jMax, xBounds, yBounds, tol=1e-6, iter_lim=100):

        self.iMax = iMax
        self.jMax = jMax
        self.xMin, self.xMax = xBounds
        self.yMin, self.yMax = yBounds

        self.tol = tol
        self.iter_lim = iter_lim

        # Create computational grid
        xi = np.linspace(0, 1, self.iMax)
        eta = np.linspace(0, 1, self.jMax)

        self.xiGrid, self.etaGrid = np.meshgrid(xi, eta)

        self.deltaXi = 1 / (self.iMax - 1)
        self.deltaEta = 1 / (self.jMax - 1)

    def y_upper(self, x):
        return self.yMax - 0.17 * np.sin((x - 2) * np.pi)

    def y_lower(self, x):
        return 0.17 * np.sin((x - 2) * np.pi)

    def dy_dx_upper(self, x):
        return -0.17 * np.pi * np.cos((x - 2) * np.pi)

    def dy_dx_lower(self, x):
        return 0.17 * np.pi * np.cos((x - 2) * np.pi)

    def create_algebraic_grid(self):
        self.xGrid = self.xMin + self.xiGrid * (self.xMax - self.xMin)
        self.yGrid = self.yMin + self.etaGrid * (self.yMax - self.yMin)

        # Instantiate boundary conditions
        # Curve boundary condition

        for i in range(self.iMax):
            x = self.xGrid[0, i]
            if 2 < x < 3:
                self.yGrid[0, i] = self.y_lower(x)
                self.yGrid[-1, i] = self.y_upper(x)
        self.xGrid[:, 0] = self.xMin
        self.xGrid[:, -1] = self.xMax

        # Normal boundary condition

        for i in range(self.iMax):
            x = self.xGrid[0, i]

            # lower boundary
            if 2 < x < 3:
                dy_dx = self.dy_dx_lower(x)
                # normal direction magnitude != 1
                normal_line = np.array([-dy_dx, 1])
                # normal direction magnitude = 1
                normal = normal_line / np.linalg.norm(normal_line)

                vector = np.array(
                    [
                        self.xGrid[1, i] - self.xGrid[0, i],
                        self.yGrid[1, i] - self.yGrid[0, i],
                    ]
                )

                vector_norm = np.linalg.norm(vector)

                vector_rot = vector_norm * normal

                self.xGrid[1, i] = self.xGrid[0, i] + vector_rot[0]
                self.yGrid[1, i] = self.yGrid[0, i] + vector_rot[1]

    def calculate_coefficients(self, i, j):
        alpha = (
            (self.xGrid[j + 1, i] - self.xGrid[j - 1, i]) ** 2
            + (self.yGrid[j + 1, i] - self.yGrid[j - 1, i]) ** 2
        ) / (4 * self.deltaEta**2)

        beta = (
            (self.xGrid[j, i + 1] - self.xGrid[j, i - 1])
            * (self.xGrid[j + 1, i] - self.xGrid[j - 1, i])
            + (self.yGrid[j, i + 1] - self.yGrid[j, i - 1])
            * (self.yGrid[j + 1, i] - self.yGrid[j - 1, i])
        ) / (4 * self.deltaXi * self.deltaEta)

        gamma = (
            (self.xGrid[j, i + 1] - self.xGrid[j, i - 1]) ** 2
            + (self.yGrid[j, i + 1] - self.yGrid[j, i - 1]) ** 2
        ) / (4 * self.deltaXi**2)

        denom = 2 * gamma / (self.deltaEta**2) + 2 * alpha / (self.deltaXi**2)
        a1 = (-beta / (2 * self.deltaEta * self.deltaXi)) / denom
        a2 = (gamma / self.deltaEta**2) / denom
        a3 = (beta / (2 * self.deltaEta * self.deltaXi)) / denom
        a4 = (alpha / self.deltaXi**2) / denom
        a5 = (alpha / self.deltaXi**2) / denom
        a6 = (beta / (2 * self.deltaEta * self.deltaXi)) / denom
        a7 = (gamma / self.deltaEta**2) / denom
        a8 = (-beta / (2 * self.deltaEta * self.deltaXi)) / denom

        return np.array([a1, a2, a3, a4, a5, a6, a7, a8])

    def solve_grid(self):
        xGridNew = self.xGrid.copy()
        yGridNew = self.yGrid.copy()
        omega = 1  # Relaxation factor

        error_flag = True
        iterations = 0

        while error_flag and iterations < self.iter_lim:
            iterations += 1
            error_flag = False
            max_error = 0

            for j, i in product(range(1, self.jMax - 1), range(1, self.iMax - 1)):

                coeffs = self.calculate_coefficients(i, j)

                xAdjacent = np.array(
                    [
                        xGridNew[j - 1, i - 1],
                        xGridNew[j - 1, i],
                        xGridNew[j - 1, i + 1],
                        xGridNew[j, i - 1],
                        self.xGrid[j, i + 1],
                        self.xGrid[j + 1, i - 1],
                        self.xGrid[j + 1, i],
                        self.xGrid[j + 1, i + 1],
                    ]
                )

                yAdjacent = np.array(
                    [
                        yGridNew[j - 1, i - 1],
                        yGridNew[j - 1, i],
                        yGridNew[j - 1, i + 1],
                        yGridNew[j, i - 1],
                        self.yGrid[j, i + 1],
                        self.yGrid[j + 1, i - 1],
                        self.yGrid[j + 1, i],
                        self.yGrid[j + 1, i + 1],
                    ]
                )

                xNew = np.dot(coeffs, xAdjacent)
                yNew = np.dot(coeffs, yAdjacent)

                xGridNew[j, i] = omega * xNew + (1 - omega) * self.xGrid[j, i]
                yGridNew[j, i] = omega * yNew + (1 - omega) * self.yGrid[j, i]

                error = np.linalg.norm(
                    np.array(
                        [
                            xGridNew[j, i] - self.xGrid[j, i],
                            yGridNew[j, i] - self.yGrid[j, i],
                        ]
                    )
                )

                # Orthoganality for bottom

                xBase = self.xGrid[0, i]
                if j == 1 and 2 < xBase < 3:
                    dy_dx = self.dy_dx_lower(xBase)
                    normal = np.array([-dy_dx, 1]) / np.linalg.norm([-dy_dx, 1])
                    step = self.deltaEta * (self.y_upper(xBase) - self.y_lower(xBase))
                    xGridNew[1, i] = self.xGrid[0, i] + normal[0] * step
                    yGridNew[1, i] = self.yGrid[0, i] + normal[1] * step

                max_error = max(max_error, error)
                if error > self.tol:
                    error_flag = True

            self.xGrid = xGridNew.copy()
            self.yGrid = yGridNew.copy()

            if iterations % 100 == 0:
                print(f"Iteration {iterations}, Max Error: {max_error}")

    def plot_grid(self):
        plt.figure(figsize=(8, 6))
        for j in range(self.jMax):
            plt.plot(self.xGrid[j, :], self.yGrid[j, :], "k-", zorder=0, lw=0.5)
        for i in range(self.iMax):
            plt.plot(self.xGrid[:, i], self.yGrid[:, i], "k-", zorder=0, lw=0.5)
        # plt.scatter(self.xGrid, self.yGrid, color="black", s=5, zorder=1)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Structured Grid using Laplace Equation")
        plt.axis("equal")
        plt.show()


if __name__ == "__main__":
    myGrid = StructuredMesh(iMax=250, jMax=50, xBounds=(0, 5), yBounds=(0, 1))
    myGrid.create_algebraic_grid()
    myGrid.plot_grid()
    myGrid.solve_grid()
    myGrid.plot_grid()
