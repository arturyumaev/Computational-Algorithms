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

def plot_approximated(x, y, legend):
    plt.grid(True)
    plt.legend(legend)
    plt.plot(x, y, "r")
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

def fact(n):
    if (n == 0):
        return 1
    return n * fact(n - 1)

def diffs(x, x_vector):
    m = len(x_vector)

    return (1 / fact(m - 1)) * (f(x) ** (m - 1))

def main():
    a, b, ndots, degree = set_params()
    table = get_table(a, b, ndots)
    print_table(table)
    table = change_weight(table)

if __name__ == "__main__":
    main()
