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
        self.delta_eta = 1 / numJNodes
        self.delta_xi = 1 / numINodes

    def ComputationalGrid(self):

        def Xi(self, i):
            return i / (self.numINodes - 1)

        def Eta(self, j):
            return j / (self.numJNodes - 1)

        self.EtaGrid = [
            [Eta(self, j) for i in range(self.numINodes)] for j in range(self.numJNodes)
        ]

        self.XiGrid = [
            [Xi(self, i) for i in range(self.numINodes)] for j in range(self.numJNodes)
        ]

        return self.EtaGrid, self.XiGrid

    def AlgebraicGrid(self, xLeft, xRight, yLower, yUpper):
        def yConversion(eta):
            return yLower + eta * (yUpper - yLower)

        def xConversion(xi):
            return xLeft + xi * (xRight - xLeft)

        self.yGrid = [[yConversion(eta) for eta in row] for row in self.EtaGrid]

        self.xGrid = [[xConversion(xi) for xi in row] for row in self.XiGrid]

        return self.yGrid, self.xGrid

    def LaplaceGrid(self):

        yGridNew = np.zeros_like(self.yGrid)
        xGridNew = np.zeros_like(self.xGrid)

        def dy_deta(j, i, yGrid):
            return (yGrid[j + 1, i] - yGrid[j - 1, i]) / (2 * self.delta_eta)

        def dy_dxi(j, i, yGrid):
            return (yGrid[j, i + 1] - yGrid[j, i - 1]) / (2 * self.delta_xi)

        def dx_deta(j, i, xGrid):
            return (xGrid[j + 1, i] - xGrid[j - 1, i]) / (2 * self.delta_eta)

        def dx_dxi(j, i, xGrid):
            return (xGrid[j, i + 1] - xGrid[j, i - 1]) / (2 * self.delta_xi)

        def alpha(j, i):
            return dx_deta(j, i) ** 2 + dy_deta(j, i) ** 2

        def beta(j, i):
            return (dx_dxi(j, i) * dx_deta(j, i)) + (dy_dxi(j, i) * dy_deta(j, i))

        def gammaa(j, i):
            return dx_dxi(j, i) ** 2 + dy_dxi(j, i) ** 2

        # define 'a' coefficients
        # iterate through array
        # crash out


if __name__ == "__main__":
    myGrid = StructuredGrid(5, 6)
    etaGrid, xiGrid = myGrid.ComputationalGrid()
    yGrid, xGrid = myGrid.AlgebraicGrid(0, 5, 0, 6)

    plt.figure(1)
    plt.scatter(xGrid, yGrid)

    plt.figure(1)
    plt.scatter(xiGrid, etaGrid)

    plt.show()
