import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class StructuredMesh:
    def __init__(self, iMax, jMax, xBounds, yBounds, tol=1e-6, iter_lim=5000):
        self.iMax = iMax
        self.jMax = jMax
        self.xMin, self.xMax = xBounds
        self.yMin, self.yMax = yBounds
        self.tol = tol
        self.iter_lim = iter_lim

        # Computational grid
        xi = np.linspace(0, 1, self.iMax)
        eta = np.linspace(0, 1, self.jMax)
        self.xiGrid, self.etaGrid = np.meshgrid(xi, eta)
        self.deltaXi = 1 / (self.iMax - 1)
        self.deltaEta = 1 / (self.jMax - 1)

    def create_algebraic_grid(self):
        # Initial guess: linear interpolation
        self.xGrid = self.xMin + self.xiGrid * (self.xMax - self.xMin)
        self.yGrid = self.yMin + self.etaGrid * (self.yMax - self.yMin)

        for j, i in product(range(self.jMax), range(self.iMax)):
            x = self.xGrid[j, i]
            if 2 < x < 3:
                self.yGrid[j, i] = self.y_lower(x) + self.etaGrid[j, i] * (
                    self.y_upper(x) - self.y_lower(x)
                )

    def y_upper(self, x):
        return self.yMax - 0.17 * np.sin((x - 2) * np.pi) if 2 < x < 3 else self.yMin

    def y_lower(self, x):
        return 0.17 * np.sin((x - 2) * np.pi) if 2 < x < 3 else self.yMax

    def dy(self, x):
        return np.pi * 0.17 * np.cos((x - 2) * np.pi)

    def fixedPoint(self, i, j):

        def g(xp, yp, xk):
            return xp + self.dy(xk) * (yp - self.y_lower(xk))

        xp = self.xGrid[j + 1, i]
        yp = self.yGrid[j + 1, i]
        xk = self.xGrid[j, i]

        for _ in range(1000):
            xkk = g(xp, yp, xk)
            if xk - xkk < self.tol:
                break
            xk = xkk
        return xkk, self.y_lower(xkk)

    def apply_boundaries(self, xGrid, yGrid):
        # Curve boundary condition

        for i in range(self.iMax):
            x = xGrid[0, i]
            if 2 < x < 3:
                yGrid[0, i] = self.y_lower(x)
                yGrid[-1, i] = self.y_upper(x)
        xGrid[:, 0] = self.xMin
        xGrid[:, -1] = self.xMax

        # Ortho boundary condition

        for i in range(self.iMax):
            x = xGrid[0, i]
            if 2 < x < 3:
                xGrid[0, i], yGrid[0, i] = self.fixedPoint(i, 0)
                xGrid[self.jMax - 1, i], yGrid[self.jMax - 1, i] = (
                    xGrid[0, i],
                    self.yMax - yGrid[0, i],
                )

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

        denom = 2 * (alpha / self.deltaXi**2 + gamma / self.deltaEta**2)
        a1 = (-beta / (2 * self.deltaEta * self.deltaXi)) / denom
        a2 = (gamma / self.deltaEta**2) / denom
        a3 = (beta / (2 * self.deltaEta * self.deltaXi)) / denom
        a4 = (alpha / self.deltaXi**2) / denom
        a5 = (alpha / self.deltaXi**2) / denom
        a6 = (beta / (2 * self.deltaEta * self.deltaXi)) / denom
        a7 = (gamma / self.deltaEta**2) / denom
        a8 = (-beta / (2 * self.deltaEta * self.deltaXi)) / denom

        return np.array([a1, a2, a3, a4, a5, a6, a7, a8])

    def apply_laplace(self):
        xGridNew = self.xGrid.copy()
        yGridNew = self.yGrid.copy()
        omega = 1  # Under-relaxation factor

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
                        xGridNew[j, i + 1],
                        xGridNew[j + 1, i - 1],
                        xGridNew[j + 1, i],
                        xGridNew[j + 1, i + 1],
                    ]
                )
                yAdjacent = np.array(
                    [
                        yGridNew[j - 1, i - 1],
                        yGridNew[j - 1, i],
                        yGridNew[j - 1, i + 1],
                        yGridNew[j, i - 1],
                        yGridNew[j, i + 1],
                        yGridNew[j + 1, i - 1],
                        yGridNew[j + 1, i],
                        yGridNew[j + 1, i + 1],
                    ]
                )

                xNew = np.dot(coeffs, xAdjacent)
                yNew = np.dot(coeffs, yAdjacent)

                # Under-relaxation
                xGridNew[j, i] = xNew
                yGridNew[j, i] = yNew

            self.apply_boundaries(xGridNew, yGridNew)

            for j, i in product(range(self.jMax), range(self.iMax)):
                error = np.linalg.norm(
                    [
                        xGridNew[j, i] - self.xGrid[j, i],
                        yGridNew[j, i] - self.yGrid[j, i],
                    ]
                )
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
            plt.plot(self.xGrid[j, :], self.yGrid[j, :], "k-", lw=0.5)
        for i in range(self.iMax):
            plt.plot(self.xGrid[:, i], self.yGrid[:, i], "k-", lw=0.5)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Structured Grid using Laplace Equation")
        plt.axis("equal")
        plt.show()


if __name__ == "__main__":
    myGrid = StructuredMesh(iMax=100, jMax=20, xBounds=(0, 5), yBounds=(0, 1))
    myGrid.create_algebraic_grid()
    myGrid.apply_boundaries(myGrid.xGrid, myGrid.yGrid)
    myGrid.apply_laplace()
    myGrid.plot_grid()
