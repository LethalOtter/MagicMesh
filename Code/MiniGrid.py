import numpy as np
import matplotlib.pyplot as plt


def y(x):
    return 0.17 * np.sin((x - 2) * np.pi)


def dy(x):
    return 0.17 * np.pi * np.cos((x - 2) * np.pi)


def g(xp, yp, xk):
    return xp + dy(xk) * (yp - y(xk))


def fixedPoint(i, j):
    xp = x_grid[j + 1, i]
    yp = y_grid[j + 1, i]
    xk = x_grid[j, i]
    for _ in range(1000):
        xkk = g(xp, yp, xk)
        if xk - xkk < 1e-10:
            break
        xk = xkk
    return xkk, y(xkk)


x_fine = np.linspace(2, 3, 100)
iMax = 5
jMax = 5
x_vals = np.linspace(2, 3, iMax)
y_vals = np.linspace(0, 1, jMax)
x_grid, y_grid = np.meshgrid(x_vals, y_vals)

for i in range(iMax):
    y_grid[0, i] = y(x_grid[0, i])

for i in range(iMax):
    x_grid[0, i], y_grid[0, i] = fixedPoint(i, 0)


### PLOTTING ###
plt.figure(figsize=(8, 6))
for j in range(jMax):
    plt.plot(x_grid[j, :], y_grid[j, :], "k-", zorder=0, lw=0.5)
for i in range(iMax):
    plt.plot(x_grid[:, i], y_grid[:, i], "k-", zorder=0, lw=0.5)
plt.plot(x_fine, y(x_fine))
# plt.scatter(x_grid, y_grid, color="black", s=5, zorder=1)
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.show()
