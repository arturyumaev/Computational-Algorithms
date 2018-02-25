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
    

    def plot(self, x, y, x_approx, y_approx, x0, y0, MSE, MAE):

        # Representation parameters
        plt.title("Approximation of sin(x)*x function in segment [{}, {}, {}], MSE:{:4.5f}, real value:{:4.3f},".format(min(x),
                                                                                                                max(x),
                                                                                                                self.segment_n_points,
                                                                                                                MSE,
                                                                                                                self.f(x0)))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)

        # Plotting original function
        plt.plot(x, y, "o-", color="#A9A9A9", lw=1)
        plt.plot(x_approx, y_approx, "r")
        plt.plot(x0, y0, "og")
        
        # Plotting an annotation of approximated value
        plt.annotate('Predicted value: {:4.5f}'.format(y0),
                     xy=(x0, y0),
                     xytext=(max(x * 0.6), max(y) * 0.3),
                     arrowprops=dict(arrowstyle="-|>",
                                     connectionstyle="arc3,rad=-0.2",
                                     fc="w"))

        # Plotting a legend
        plt.legend(["Dirac function", "Approximated function", "Predicted value"])


    def approximate(self, x, y, x0, degree):

        # Approximation module

        def dot_product(x0, x):

            # For given x0 and vector x returs value like:
            # x  - [x_1, x_2, ..., x_N]
            # x0 - scalar
            # return (x0 - x_1) * (x0 - x_2) * ... * (x0 - x_N)
            
            product = 1

            for x_i in x:
                product *= (x0 - x_i)

            return product

        def polynof_coef(x, y):
            
            # Alpha parameters:
            #
            # For 2: y(x_i, x_j) = (y_i - y_j) / (x_i - x_j)
            # For 3: y(x_i, x_j, x_k) = (y(x_i, x_j) - y(x_j, x_k)) / (x_i - x_k)
            # ...
            # For N: y(x_1, ..., x_N) = ((x_1, ..., x_N-1) - y(x_2, ..., x_N)) / (x_1 - x_N)

            if ((len(x) == 1) or (len(x) == 0)):
                return 1

            if (len(x) == 2):
                return (y[0] - y[1]) / (x[0] - x[1])
            else:
                return (polynof_coef(x[:-1], y[:-1]) - polynof_coef(x[1:], y[1:])) / (x[0] - x[-1])

        def calculate_approximation(x, y, x0, n):

            # Returns an approximation like:
            #
            # Pn(x) = y0 + (x-x0)y(x0,x1) + (x-x0)(x-x1)y(x0,x1,x2) + (x-x0)(x-x1)(x-x2)y(x0,x1,x2,x3) ...
            
            sigma = y[0]
            for k in range(1, n + 1):
                sigma += dot_product(x0, x[:k]) * polynof_coef(x[:k + 1], y[:k + 1])

            return sigma


        fitted_value = calculate_approximation(x, y, x0, degree)

        return fitted_value


    def interpolate(self):
        pass



    def get_x_segment(self, n_points, a=-15, b=15):

        # Getting linear space of x values from a to b, splitted by n_points
        return np.linspace(a, b, n_points)


    def get_y_segment(self, x):
        return self.f(x)


    def f(self, x):

        # Dirac function
        return -(np.cos(x) - x)


    def compute_MSE_MAE(self, x, N_nearest_list, y_true_global, y_pred):

        # Mean Squared Error and Mean Absolute Error computation
        
        start_i = np.where(x == N_nearest_list[0])
        start_i = start_i[0][0] # Numpy unboxing

        MSE = 0
        MAE = 0

        for i in range(len(y_pred)):
            MSE +=    (y_true_global[i + start_i] - y_pred[i])**2
            MAE += abs(y_true_global[i + start_i] - y_pred[i])

        return MSE, MAE


    def get_x0_nearest(self, x0, x, N):

        if N % 2 != 0:
            N += 1

        if N == 0:
            N = 2

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


    def run(self):

        flag = int(input("Need to find a root?[1/0]: "))
        if flag == 1:
            a = float(input("Input a: "))
            b = float(input("Input b: "))
            if a > b:
                a, b = b, a
        result_root = []

        # Getting original Dirac function
        self.segment_n_points = 50
        

        X = self.get_x_segment(n_points=self.segment_n_points)
        y = self.get_y_segment(X)

        if flag == 1:
            aa, bb = self.find_root(a, b, X, y)
            
        # Input parameters
        degree, x0 = self.read_input()

        self.N_nearest = degree - 1

        # Getting N nearest of x0 points
        N_nearest_list = np.array(self.get_x0_nearest(x0, X, self.N_nearest))
        y_on_N_nearest = self.get_y_segment(N_nearest_list)

        result = []
        for n in N_nearest_list:
            result.append(self.approximate(N_nearest_list, y_on_N_nearest, n, degree))
            if flag == 1:
                result_root.append(self.approximate(bb, aa, 0, degree))

        if flag == 1:
            print("ROOT:", self.approximate(bb, aa, 0, degree))

        MSE, MAE = self.compute_MSE_MAE(X, N_nearest_list, y, result)

        # Approximation on current segment
        self.plot(X,
                  y,
                  N_nearest_list,
                  result,
                  x0,
                  self.approximate(N_nearest_list,
                                   y_on_N_nearest,
                                   x0,
                                   degree),
                  MSE,
                  MAE)
        
        plt.show()


    def find_root(self, a, b, X, y):
        
        x_s = 0
        x_e = 0
        
        for x_i in X:
            if a > x_i:
                x_s = x_i
            if b >= x_i:
                x_e = x_i

        X = list(X)
        ind_x_s = X.index(x_s)
        ind_x_e = X.index(x_e)

        X_a = []
        Y_a = []

        for i in range(ind_x_s, ind_x_e + 1):
            X_a.append(X[i])
            Y_a.append(self.f(X[i]))

        return X_a, Y_a

        


def main():
    i = Approximator()
    i.run()

if __name__ == '__main__':
    main()
