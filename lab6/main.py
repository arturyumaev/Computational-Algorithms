import numpy as np
import math
from matplotlib import pyplot as plt

def input_alpha():
    return float(input("Input aplha: "))

def input_eps():
    return float(input("Input epsilon: "))

def input_nodes():
    nodes = int((input("Input count of nodes: ")))
    if (nodes < 1):
        nodes = 1
    if (nodes > 40000):
        nodes = 40000
    return nodes

def input_degree():
    return int(input("Input degree: "))

def input_params():
    a = input_alpha()
    eps = input_eps()
    n_nodes = input_nodes()
    degree = input_degree()

    return a, eps, n_nodes, degree

def compute_Pn_coefs():
    pass

def compute_Pn():
    pass

# Lejandre Polynom
def Pn(x, n):
    if (n == 0):
        return 1

    if (n == 1):
        return x

    if (n > 1):
        first = (2 - 1 / n) * x * Pn(x, n - 1)
        second = (1 - 1 / n) * Pn(x, n - 2)
        return first - second

def sqtr(x):
    return x ** (1 / 2)

def sqr(x):
    return x * x

def f(t):
    return 1 / sqrt(2 * np.pi) * exp(-sqr(t) / 2)

def f_test(x):
    return np.sin(x)

def half_divide_method(a, b, eps):
    x = (a + b) / 2
    while math.fabs(f(x)) >= eps:
        x = (a + b) / 2
        a, b = (a, x) if f(a) * f(x) < 0 else (x, b)
    return (a + b) / 2


plt.grid(True)
plt.plot(np.linspace(-1, 1, 500), Pn(np.linspace(-1, 1, 500), 30))
plt.show()
