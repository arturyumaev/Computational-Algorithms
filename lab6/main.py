import numpy as np
import math
from matplotlib import pyplot as plt

#  a                  n
#  /          b - a  ___
#  | f(t)dt = -----  \
#  /            2    /__ A_i f(t_i)
#  b                 i=1


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

def sqrt(x):
    return x ** (1 / 2)

def sqr(x):
    return x * x

def sin(x):
    return np.sin(x)

def f(t):
    return 1 / sqrt(2 * np.pi) * exp(-sqr(t) / 2)

def f_test(x):
    return np.sin(x)

def half_divide_method(a, b, eps, Pn_degree):
    c = 0
    x = (a + b) / 2
    while math.fabs(Pn(x, Pn_degree)) >= eps:
        c += 1
        x = (a + b) / 2
        a, b = (a, x) if Pn(a, Pn_degree) * Pn(x, Pn_degree) < 0 else (x, b)

        # There is no root in area
        if (c >= 100):
            return "NO_ROOT"

    return (a + b) / 2

def _find_roots(n_nodes, h, eps, a=-1, b=1):
    roots = []
    start = a
    end = b

    while (start <= end):
        root_result = half_divide_method(start, start + h, eps, n_nodes)
        start += h

        if (root_result != "NO_ROOT"):
            roots.append(root_result)

    if (len(roots) != n_nodes):
        return 'ROOTS_NOT_FOUND'
    else:
        return roots

def find_all_roots(n_nodes, eps=1e-5):
    h = 1 / n_nodes

    while (_find_roots(n_nodes, h, eps) == 'ROOTS_NOT_FOUND'):
        h /= 2

    roots = _find_roots(n_nodes, h, eps)

    return roots

a = find_all_roots(15)
#plt.grid(True)
#plt.plot(np.linspace(-1, 1, 500), Pn(np.linspace(-1, 1, 500), 3))
#plt.show()
