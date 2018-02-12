from matplotlib import pyplot as plt
import numpy as np

class Approximator(object):

    def __init__(self):
        pass


    def read_input(self):
        self.degree = int(input("Input Pn(X) degree: "))
        self.x0 = float(input("Input x0: "))
    

    def plot(self, x, y, y_approximated):

        # Representation parameters
        plt.title("Approximation of Dirac function in segment [{0}, {1}]".format(min(x), max(x)))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)

        # Plotting original function
        plt.plot(x, y, "ob")

        plt.legend(["Dirac function"])
        #plt.plot(x, y_approximated, "r")


    def get_parameter_alpha(self, x, y):
        if len(x) == 2:
            return (y[0] - y[1]) / (x[0] - x[1])
        else:
            return (f(x[:-1], y[:-1]) - f(x[1:], y[1:])) / (x[0] - x[-1])


    def approximate(self, x, x0, degree):
        pass


    def interpolate(self):
        pass


    def find_root(self, a, b):
        pass


    def get_x_segment(self, a=-15, b=15, n_points=200):

        # Getting linear space of x values from a to b, splitted by n_points
        return np.linspace(a, b, n_points)


    def get_y_segment(self, x):
        return self.f(x)


    def f(self, x):

        # Dirac function
        return np.sin(x) / x


    def run(self):
        self.read_input()
        x = self.get_x_segment()
        y = self.get_y_segment(x)
        self.plot(x, y, 1)
        plt.show()


#i = Approximator()
#i.run()


def f(x, y):
    if len(x) == 1 or len(x) == 0:
        return 1
    if len(x) == 2:
        return (y[0] - y[1]) / (x[0] - x[1])
    else:
        return (f(x[:-1], y[:-1]) - f(x[1:], y[1:])) / (x[0] - x[-1])

def get(x0, xe, x, y, X):
    s = 0
    for i in range(x0, xe + 1):
        s += (X - x0) * f(x[x.index(x0):x.index(xe) + 1], y[x.index(x0):x.index(xe) + 1])
    return s
