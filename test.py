import itertools
import numpy as np
import matplotlib.pyplot as plt


class StructuredGrid:
    def __init__(self, numINodes, numJNodes, xBounds, yBounds, tol=1e-6, max_iter=5000):
        self.numINodes = numINodes
        self.numJNodes = numJNodes

        self.xLeft, self.xRight = xBounds
        self.yLower, self.yUpper = yBounds
        self.tol = tol
        self.max_iter = max_iter

        # Computational Grid initialization
        self.xiGrid = np.zeros((numJNodes, numINodes), dtype=np.float64)
        self.etaGrid = np.zeros((numJNodes, numINodes), dtype=np.float64)

        # Algebraic Grid initialization
        self.xGrid = np.zeros((numJNodes, numINodes))
        self.yGrid = np.zeros((numJNodes, numINodes))

        # Coefficient Vector initialization
        self.Coeffs = np.array((0, 0))

        self.delta_eta = 1 / (numJNodes - 1)
        self.delta_xi = 1 / (numINodes - 1)

    def y_upper_boundary(self, x):
        return (
            self.yUpper - 0.17 * np.sin((x - 2) * np.pi) if 2 < x < 3 else self.yUpper
        )

    def y_lower_boundary(self, x):
        return 0.17 * np.sin((x - 2) * np.pi) if 2 < x < 3 else self.yLower

    def make_computational_grid(self):
        for j, i in itertools.product(range(self.numJNodes), range(self.numINodes)):
            self.etaGrid[j, i] = j / (self.numJNodes - 1)
            self.xiGrid[j, i] = i / (self.numINodes - 1)

    def make_algebraic_grid(self):
        for j, i in itertools.product(range(self.numJNodes), range(self.numINodes)):
            x = self.xLeft + self.xiGrid[j, i] * (self.xRight - self.xLeft)
            if j in [0, self.numJNodes - 1]:
                y_l_bound, y_u_bound = self.y_lower_boundary(x), self.y_upper_boundary(
                    x
                )
            else:
                y_l_bound, y_u_bound = self.yLower, self.yUpper

            self.yGrid[j, i] = y_l_bound + self.etaGrid[j, i] * (y_u_bound - y_l_bound)
            self.xGrid[j, i] = x

    def calculate_coefficients(self, j, i):
        dx_deta = (self.xGrid[j + 1, i] - self.xGrid[j - 1, i]) / (2 * self.delta_eta)
        dx_dxi = (self.xGrid[j, i + 1] - self.xGrid[j, i - 1]) / (2 * self.delta_xi)
        dy_deta = (self.yGrid[j + 1, i] - self.yGrid[j - 1, i]) / (2 * self.delta_eta)
        dy_dxi = (self.yGrid[j, i + 1] - self.yGrid[j, i - 1]) / (2 * self.delta_xi)

        alpha = dx_deta**2 + dy_deta**2
        beta = (dx_dxi * dx_deta) + (dy_dxi * dy_deta)
        gamma = dx_dxi**2 + dy_dxi**2

        # norm_delta_xi = self.delta_xi * (self.xRight - self.xLeft)
        # norm_delta_eta = self.delta_eta * (self.yUpper - self.yLower)

        # epsilon = 1e-4
        # denom = 2 * alpha * norm_delta_eta**2 + 2 * gamma * norm_delta_xi**2

        # A = (alpha * norm_delta_xi**2) / denom
        # B = (beta * norm_delta_xi * norm_delta_eta) / (denom * 4)
        # C = (gamma * norm_delta_xi**2) / denom

        denom = 2 * alpha * self.delta_eta**2 + 2 * gamma * self.delta_xi**2

        A = (alpha * self.delta_xi**2) / denom
        B = (beta * self.delta_xi * self.delta_eta) / (denom * 2)
        C = (gamma * self.delta_xi**2) / denom

        return -B, C, B, A, A, B, C, -B

    def GaussSeidel(self):
        error = True
        iteration = 0
        while error and iteration < self.max_iter:
            errors = []

            newYGrid = self.yGrid.copy()

            # Filtering out rows and columns containing boundary conditions
            for j, i in itertools.product(
                range(1, self.numJNodes - 1), range(1, self.numINodes - 1)
            ):
                Coeffs = self.calculate_coefficients(j, i)

                newY = (
                    Coeffs[0] * newYGrid[j - 1, i - 1]
                    + Coeffs[1] * newYGrid[j - 1, i]
                    + Coeffs[2] * newYGrid[j - 1, i + 1]
                    + Coeffs[3] * newYGrid[j, i - 1]
                    + Coeffs[4] * self.yGrid[j, i + 1]
                    + Coeffs[5] * self.yGrid[j + 1, i - 1]
                    + Coeffs[6] * self.yGrid[j + 1, i]
                    + Coeffs[7] * self.yGrid[j + 1, i + 1]
                )

                errors.append(np.abs(self.yGrid[j, i] - newY))

                newYGrid[j, i] = newY

            self.yGrid = newYGrid
            maxError = np.max(errors)
            if maxError < self.tol:
                error = False
            iteration += 1
            print(f"Iteration {iteration}, Max Error: {maxError}")

        if iteration == self.max_iter:
            print("Warning: Maximum number of iterations reached without convergence.")

    def plot_grid(self):
        plt.figure(figsize=(8, 6))
        for j in range(self.numJNodes):
            plt.plot(
                self.xGrid[j, :],
                self.yGrid[j, :],
                color="black",
                zorder=0,
                linewidth=0.5,
            )
        for i in range(self.numINodes):
            plt.plot(
                self.xGrid[:, i],
                self.yGrid[:, i],
                color="black",
                zorder=0,
                linewidth=0.5,
            )
        # plt.scatter(self.xGrid, self.yGrid, color="black", s=5, zorder=1)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Structured Grid using Laplace Equation")
        plt.axis("equal")
        plt.show()


if __name__ == "__main__":
    myGrid = StructuredGrid(
        numINodes=50, numJNodes=40, xBounds=(0, 5), yBounds=(0, 1), tol=1e-6
    )
    myGrid.make_computational_grid()
    myGrid.make_algebraic_grid()
    myGrid.GaussSeidel()
    myGrid.plot_grid()
