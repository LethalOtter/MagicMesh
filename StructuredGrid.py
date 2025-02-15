# %%

import numpy as np
import matplotlib.pyplot as plt


class StructuredGrid:
    """
    It is of importance to note that 'i' is associated with increments and indices pointing in the x or xi direction
    and 'j' is associated with increments and indices pointing in the y or eta direction. Arrays will be accessed using
    [j, i] since the j will relate to row increments and the i will relate to column increments.
    """

    def __init__(
        self,
        numJNodes,
        numINodes,
    ):
        self.numINodes = numINodes
        self.numJNodes = numJNodes
        self.numNodes = numINodes * numJNodes
        self.EtaGrid = None
        self.XiGrid = None
        self.xGrid = None
        self.yGrid = None
        self.Coeffs = None
        self.delta_eta = 1 / (numJNodes - 1)
        self.delta_xi = 1 / (numINodes - 1)

    def ComputationalGrid(self):

        def Xi(self, i):
            return i / (self.numINodes - 1)

        def Eta(self, j):
            return j / (self.numJNodes - 1)

        self.EtaGrid = []
        for j in range(self.numJNodes):
            for i in range(self.numINodes):
                self.EtaGrid.append(Eta(self, j))

        self.XiGrid = []
        for j in range(self.numJNodes):
            for i in range(self.numINodes):
                self.XiGrid.append(Xi(self, i))

        return self.EtaGrid, self.XiGrid

    def MakeAlgebraicGrid(self, xLeft, xRight, yLower, yUpper):
        def yConversion(eta):
            return yLower + eta * (yUpper - yLower)

        def xConversion(xi):
            return xLeft + xi * (xRight - xLeft)

        self.yGrid = [yConversion(eta) for eta in self.EtaGrid]

        self.xGrid = [xConversion(xi) for xi in self.XiGrid]

        # y Boundary conditions
        for l in range(self.numINodes):
            if self.xGrid[l] <= 2 or self.xGrid[l] >= 3:
                continue

            self.yGrid[l] = 0.17 * np.sin((self.xGrid[l] - 2) * np.pi)
            self.yGrid[self.numINodes * (self.numJNodes - 1) + l] = 1 - 0.17 * np.sin(
                (self.xGrid[l] - 2) * np.pi
            )

        return self.yGrid, self.xGrid

    def CalculateCoefficients(self, l):
        dx_deta = (self.xGrid[l + self.numINodes] - self.xGrid[l - self.numINodes]) / (
            2 * self.delta_eta
        )

        dx_dxi = (self.xGrid[l + 1] - self.xGrid[l - 1]) / (2 * self.delta_xi)

        dy_deta = (self.yGrid[l + self.numINodes] - self.yGrid[l - self.numINodes]) / (
            2 * self.delta_eta
        )

        dy_dxi = (self.yGrid[l + 1] - self.yGrid[l - 1]) / (2 * self.delta_xi)

        alpha = dx_deta**2 + dy_deta**2
        beta = (dx_dxi * dx_deta) + (dy_dxi * dy_deta)
        gamma = dx_dxi**2 + dy_dxi**2

        A = (alpha * self.delta_xi**2) / (
            2 * alpha * self.delta_eta**2 + 2 * gamma * self.delta_xi**2
        )
        B = (beta * self.delta_xi * self.delta_eta) / (
            8 * alpha * self.delta_eta**2 + 8 * gamma * self.delta_xi**2
        )
        C = (gamma * self.delta_xi**2) / (
            2 * alpha * self.delta_eta**2 + 2 * gamma * self.delta_xi**2
        )

        # print(f"{alpha=}")
        # print(f"{beta=}")
        # print(f"{gamma=}")

        return -B, C, B, A, A, B, C, -B

    def GaussSeidel(self, tol):
        tol
        error = True
        while error:
            errors = []

            newYGrid = self.yGrid.copy()

            # Filtering out rows and columns containing boundary conditions
            for l, y in enumerate(self.yGrid):
                if l // self.numINodes < 1:  # Skipping bottom row
                    continue
                if l // self.numINodes == self.numJNodes - 1:  # Skipping top row
                    continue
                if l % self.numINodes < 1:  # Skipping left column
                    continue
                if l % self.numINodes == self.numINodes - 1:  # Skipping right column
                    continue

                Coeffs = self.CalculateCoefficients(l)

                newY = (
                    Coeffs[0] * newYGrid[l - self.numINodes - 1]
                    + Coeffs[1] * newYGrid[l - self.numINodes]
                    + Coeffs[2] * newYGrid[l - self.numINodes + 1]
                    + Coeffs[3] * newYGrid[l - 1]
                    + Coeffs[4] * self.yGrid[l + 1]
                    + Coeffs[5] * self.yGrid[l + self.numINodes - 1]
                    + Coeffs[6] * self.yGrid[l + self.numINodes]
                    + Coeffs[7] * self.yGrid[l + self.numINodes + 1]
                )

                errors.append(np.abs(self.yGrid[l] - newY))

                newYGrid[l] = newY

            self.yGrid = newYGrid
            maxError = max(errors)
            if maxError < tol:
                error = False

        return


if __name__ == "__main__":
    tolerance = 1e-6
    myGrid = StructuredGrid(30, 30)
    etaGrid, xiGrid = myGrid.ComputationalGrid()
    myGrid.MakeAlgebraicGrid(2, 3, 0, 1)

    # plt.figure(1)
    # plt.scatter(myGrid.xGrid, myGrid.yGrid)
    # plt.title("Iteration Zero")
    # plt.axis("equal")

    myGrid.GaussSeidel(tolerance)

    plt.figure(2)
    plt.scatter(myGrid.xGrid, myGrid.yGrid)
    # plt.axis("equal")

    plt.show()
