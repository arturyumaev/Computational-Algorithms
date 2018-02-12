# Artur Yumaev IU7-45 Computation Algorithms lab1 Dirac function approximation

from matplotlib import pyplot as plt
import numpy as np

class Approximator(object):

    def __init__(self):
        pass


    def read_input(self):
        self.degree = int(input("Input Pn(X) degree: "))
        self.x0 = float(input("Input x0: "))

        return self.degree, self.x0
    

    def plot(self, x, y, x_approx, y_approx, x0, y0):

        # Representation parameters
        plt.title("Approximation of Dirac function in segment [{0}, {1}, {2}]".format(min(x),
                                                                                 max(x),
                                                                                 self.segment_n_points))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)

        # Plotting original function
        plt.plot(x, y, "o-", color="#A9A9A9", lw=1)
        plt.plot(x_approx, y_approx, "r")
        plt.plot(x0, y0, "og")
        
        # Plotting an annotation approximated value
        plt.annotate('Predicted value: {:4.5f}'.format(y0),
                     xy=(x0, y0),
                     xytext=(max(x * 0.6), max(y) * 0.3),
                     arrowprops=dict(arrowstyle="-|>",
                                     connectionstyle="arc3,rad=-0.2",
                                     fc="w"))

        plt.legend(["Dirac function", "Approximated function", "Predicted value"])


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


    def get_x_segment(self, n_points, a=-15, b=15):

        # Getting linear space of x values from a to b, splitted by n_points
        return np.linspace(a, b, n_points)


    def get_y_segment(self, x):
        return self.f(x)


    def f(self, x):

        # Dirac function
        return np.sin(x) / x


    def get_x0_nearest(self, x0, x, N):

        if N % 2 != 0:
            N += 1

        # Searching for N neartest of x0
        x_prev = 0
        x_next = 0
        N_nearest = []
        
        for i in range(0, len(x) - 1):
            if x0 > x[i] and x0 < x[i + 1]:
                x_prev = x[i]
                x_next = x[i + 1]

                tmp_c = 0
                t = 0
                j = -1
                while t != N // 2 and j != 0:
                    j = i + tmp_c
                    N_nearest.append(x[j])
                    tmp_c -= 1
                    t += 1
                N_nearest = [k for k in reversed(N_nearest)]
                
                tmp_c = 0
                t = 0
                j = -1
                while t != N // 2 and j != 0:
                    j = i + tmp_c
                    N_nearest.append(x[j + 1])
                    tmp_c += 1
                    t += 1
                break # 2 nearest values was found

        return N_nearest


    def dot(self, x, x_vector):
        product = 1

        for x_i in x_vector:
            product *= (x - x_i)

        return product


    def run(self):

        # Getting original Dirac function
        self.segment_n_points = 100
        self.N_nearest = self.segment_n_points // 5


        X = self.get_x_segment(n_points=self.segment_n_points)
        y = self.get_y_segment(X)
        
        # Input parameters
        degree, x0 = self.read_input()

        # Getting N nearest of x0 points
        N_nearest_list = self.get_x0_nearest(x0, X, self.N_nearest)
        p = np.array(N_nearest_list)
        q = self.get_y_segment(p)
        o = []
        for i in N_nearest_list:
            o.append(self.approximate(p, q, i, degree))

        print(N_nearest_list)
        # Approximation on current segment
        self.plot(X, y, p, o, x0, self.approximate(p, q, x0, degree))
        plt.show()


i = Approximator()
i.run()





    
