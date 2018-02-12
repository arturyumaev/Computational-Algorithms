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




    def approximate(self, x, y, x0, degree):

        def dot_product(x0, x):
            product = 1
            for x_i in x:
                product *= (x0 - x_i)
            return product

        def polynof_coef(x, y):
            if len(x) == 1 or len(x) == 0:
                return 1
            if len(x) == 2:
                return (y[0] - y[1]) / (x[0] - x[1])
            else:
                return (polynof_coef(x[:-1], y[:-1]) - polynof_coef(x[1:], y[1:])) / (x[0] - x[-1])

        def calculate_approximation(x, y, x0, n):
            sigma = y[0]
            for k in range(1, n + 1):
                sigma += dot_product(x0, x[:k]) * polynof_coef(x[:k + 1], y[:k + 1])
            return sigma


        fitted_value = calculate_approximation(x, y, x0, degree)

        return fitted_value



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

    def dot(self, x, x_vector):
        product = 1

        for x_i in x_vector:
            product *= (x - x_i)

        return product


    def run(self):
        self.read_input()
        x = self.get_x_segment()
        y = self.get_y_segment(x)
        self.plot(x, y, 1)
        plt.show()


i = Approximator()
print(i.approximate([0, 1, 2, 3], [0, 0.5, 0.866, 1], 1.5, 3))
i.run()





    
