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

    # Fill the table by weights
    for i in range(weights.shape[0]):
        table[2][i] = weights[i]

    return table

def print_table(table):
    print(pd.DataFrame(table.T, columns=['x', 'y=sin(x)', 'w']))

def set_params():
    a = float(input("Input a: ") or -10)
    b = float(input("Input b: ") or 10)
    ndots = int(input("Input ndots: ") or 10)

    return a, b, ndots

def main():
    a, b, ndots = set_params()
    table = get_table(a, b, ndots)
    print_table(table)

if __name__ == "__main__":
    main()
