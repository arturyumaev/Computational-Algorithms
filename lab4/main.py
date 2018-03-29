import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def f(x):
    return np.sin(x)


def get_table(a, b, ndots):
    table = np.zeros((3, ndots)) # 3 is x, y, w
    weights = np.ones((ndots))
    x = np.linspace(a, b, ndots)
    y = f(x)

    # Fill the table by x vector
    for i in range(x.shape[0]):
        table[0][i] = x[i]

    # Fill the table by y vector
    for i in range(y.shape[0]):
        table[1][i] = y[i]

    # Fill the table by weights5
    for i in range(weights.shape[0]):
        table[2][i] = weights[i]

    return table


def phi(x, k, y=False, with_y=False):
    # a - numpy vector of coefficients
    # x - numpy vector
    # k - degree
    # Ф(x, a) = sum(a_k * ф_k(x))_i for i=(0, ..., n)
    if (with_y == False):
        return np.sum(x ** k)
    else:
        return np.sum((x ** k) * y)


def get_slae(x, y, k):
    # Create matrix A and b
    A = np.zeros((k + 1, k + 1))
    b = np.zeros((k + 1, 1))

    # A formation
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i][j] = phi(x, i + j)

    # b formation
    for i in range(b.shape[0]):
        b[i] = phi(x, i, y=y, with_y=True)

    return A, b


def solve_slae(A, b):
    # Returns a0, ..., ak coeficients
    return np.linalg.solve(A, b)


def plot_approximated(x, y, x_fitted, y_fitted, legend=['y=f(x)', 'Approximated Ф(x)~f(x)']):
    plt.title('Function approximation')
    plt.grid(True)
    plt.legend(legend)
    plt.plot(x, y, x_fitted, y_fitted, "r")
    plt.show()


def change_weight(table):
    w_idx = int(input("Input weight index in the table: "))
    new_w = float(input("Input weight: "))
    table[2][w_idx] = new_w

    # Show new table
    print_table(table)
    
    return table


def print_table(table):
    print(pd.DataFrame(table.T, columns=['x', 'y=sin(x)', 'w']))


def set_params():
    a = float(input("Input a: ") or -10)
    b = float(input("Input b: ") or 10)
    ndots = int(input("Input ndots: ") or 10)
    degree = int(input("Input Pn(x) degree: ") or 3)

    return a, b, ndots, degree


def compute_polynom(x, coefs):
    fitted_value = coefs[0]
    for i in range(1, len(coefs)):
        fitted_value = coefs[i] * (x ** i)

    return fitted_value


def get_fitted_space(x, coefs, ndots):
    y = np.array([compute_polynom(i, coefs) for i in x])

    return x, y


# Set main configs
a, b, ndots, degree = set_params()
table = get_table(a, b, ndots)
print_table(table)
table = change_weight(table)

# System solving
x = table[0]
y = table[1]
A, B = get_slae(x, y, degree)
coefficients = solve_slae(A, B)
x_fitted, y_fitted = get_fitted_space(x, coefficients, ndots)

# Plotting results
plot_approximated(x, y, x_fitted, y_fitted)
"""
if __name__ == "__main__":
    main()
"""
