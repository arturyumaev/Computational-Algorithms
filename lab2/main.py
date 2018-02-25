import numpy as np
import pandas as pd


def f(x, y):
    z = x**2 + y**2
    return z


def get_matrix():
        
    m = 6
    n = 6

    A = np.array([[0]*6 for i in range(6)])

    for x in range(len(A)):
        for y in range(len(A[x])):
            A[x][y] = f(x, y)

    A = pd.DataFrame(A)

    return A
    

def main():

    A = get_matrix()

    PnX = int(input("Input Pn(x): "))
    PnY = int(input("Input Pn(y): "))

    print(A)

if __name__ == '__main__':
    main()
