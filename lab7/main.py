# Шибанова Дарья, ИУ7-42
# Лабораторная работа №7
# Для функции y(x) = a0 * x / (a1 + a2 * x)
# построить таблицу, включающую в себя столбцы x, y, односторонние
# производные, центральные производные, краевые производные 
# при порядке точности o(h*h), значения второй формулы Рунге
# для односторонних производных, выравнивающие переменные.

from math import *
import numpy as np
from prettytable import PrettyTable
from itertools import combinations

# Исходная функция
def f(x):
    a0 = 1
    a1 = 2
    a2 = 4
    return a0 * x / (a1 + a2 * x)

# Первая производная для точного значения функции:
def f1(x):
    #a0 = 1
    #a1 = 2
    #a2 = 4
    return (-4 * x) / ((4 * x + 2) * (4 * x + 2)) + 1 / (4 * x + 2)

# Функция получения наборов x и y:
def get_x_y(x1, x2, h, f):
    x = []
    y = []
    for xi in np.arange(x1, x2 + h, h):
        x.append(xi)
        y.append(f(xi))
    return x, y

# Разделённая разность
def div_diff(xs, ys):
    if len(xs) == 1:
        return ys[0]
    else:
        return (div_diff(xs[:-1], ys[:-1]) - div_diff(xs[1:], ys[1:])) / (xs[0] - xs[-1])

# Функция полиномиального разложения (полином Ньютона):
def pol(x, y, x_find):
    z = [x_find - xi for xi in x]
    y_find = div_diff(x[:2], y[:2])
    for i in range(1, len(z)):
        it = combinations(z[:i+1], i)
        summ = 0
        for k in it:
            if k is not None:
                p = 1
                for i in range(len(k)):
                    p *= k[i]
                summ += p
        y_find += (summ * div_diff(x[:i+2], y[:i+2]))
    return y_find
    
# Левосторонние производные:
def left_side(y, h):
    return [(y[i] - y[i-1]) / h for i in range(1, len(y))]

# Правосторонние производные:
def right_side(y, h):
    return [(y[i+1] - y[i]) / h for i in range(len(y) - 1)]

# Центральные производные:
def central(y, h):
    return [(y[i+1] - y[i-1]) / 2 / h for i in range(1, len(y) - 1)]

# Формула Рунге второй степени:
def Runge(y, h):
    result = []

    # Для первых двух точек используем правые прозводные
    for i in range(2):
        first_ksi = (y[i+1] - y[i]) / h
        second_ksi = (y[i+2] - y[i]) / 2 / h
        result.append(first_ksi + (first_ksi - second_ksi))
    # Для центральных точек - центральные производные
    for i in range(2, len(y) - 2):
        first_ksi = (y[i+1] - y[i-1]) / 2 / h
        second_ksi = (y[i+2] - y[i-2]) / 4 / h
        result.append(first_ksi + (first_ksi - second_ksi) / 3)
    # Для двух последних - левые производные
    for i in range(len(y) - 2, len(y)):
        first_ksi = (y[i] - y[i-1]) / h
        second_ksi = (y[i] - y[i-2]) / 2 / h
        result.append(first_ksi + (first_ksi - second_ksi))

    return result

# Левосторонняя производная повышенной точности (h*h):
def left_side_high(y, h):
    res = (-3 * y[0] + 4 * y[1] - y[2]) / 2 / h
    return res

# Правосторонняя производная повышенной точности (h*h):
def right_side_high(y, h):
    res = (3 * y[-1] - 4 * y[-2] + y[-3]) / 2 / h
    return res

# Выравнивающие переменные:
def level_vars(x, y, h):
    new_y = [1/i for i in y]
    new_x = [1/i for i in x]
    l = Runge(new_y, h)
    l_new = []
    for i in range(len(l)):
        l_new.append(l[i]/new_y[i]/new_y[i]*new_x[i]*new_x[i])
    return l_new

print('По умолчанию стоит отрезок [0;5] с шагом 1\n')
print('Использовать промежуток и шаг по умолчанию? (1 - да, 0 - нет)\n')
choice = int(input())
while ((choice != 1) and (choice != 0)):
    print('Некорректный ввод. Пожалуйста, введите заново:\n')
    choice = int(input())
if choice == 1:
    x1 = 0
    x2 = 5
    h = 1
else:
    # Ввод границ и шага:
    x1, x2 = map(float, input('Введите границы интервала \
для построения таблицы: ').split())
    h = float(input('Введите шаг таблицы: '))

x, y = get_x_y(x1, x2, h, f)

left = [''] + left_side(y, h)
right = right_side(y, h) + ['']
centre = [''] + central(y, h) + ['']

exrt = ['' for i in range(len(x))]
exrt[0] = left_side_high(y, h)
exrt[-1] = right_side_high(y, h)

r = Runge(y, h)

lev = level_vars(x, y, h)

table = PrettyTable()
table.add_column("X", x)
table.add_column("Y", y)
table.add_column("Левостороняя", left)
table.add_column("Правостороняя", right)
table.add_column("Центральная производная", centre)
table.add_column("Повышенная точность", exrt)
table.add_column("2ая формула Рунге", r)
table.add_column("Выравнивающие переменные", lev)
print(table)

x_find = float(input('Введите  Х:'))
print("\nРеальный результат:", f1(x_find))
print("\nРезультат полиномиального разложения:", pol(x, y, x_find))
print("\nПогрешность полиномиального разложения:", abs(f1(x_find) - pol(x, y, x_find)))
