# import numpy as np
# from scipy.integrate import odeint
# import matplotlib.pyplot as plt
#
# def damped_driven_oscillator(y, t, n, p, m, F_func_args):
#     """
#     Определяет дифференциальное уравнение затухающего и возбуждаемого осциллятора второго порядка.
#     y -- вектор состояния [x, v]
#     t -- время
#     n -- коэффициент затухания
#     p -- частота осциллятора
#     m -- масса
#     F_func_args -- аргументы для функции, описывающей внешнюю силу
#     """
#     x, v = y
#     F = F_sin(*F_func_args, t)
#     dydt = [v, -(2*n*v + p**2*x) + F/m]
#     return dydt
#
# def F_sin(A, w, t):
#     """
#     Описывает функцию внешней силы как синусоидальную зависимость от времени.
#     A -- амплитуда
#     w -- частота
#     t -- время
#     """
#     return A * np.sin(w * t)
#
# # Начальные условия
# x0 = 0.0  # начальное смещение
# v0 = 0.0  # начальная скорость
# y0 = [x0, v0]
#
# # Время
# T = np.linspace(0, 10, 1000)
#
# # Параметры модели
# n = 100.0  # коэффициент затухания
# p = 10.0  # частота осциллятора
# A = 10.0  # амплитуда внешней силы
# w = 100.0  # частота внешней силы
# m = 1.0
#
# # Аргументы для функции внешней силы
# F_func_args = (A, w)
#
# # Решение дифференциального уравнения
# sol = odeint(damped_driven_oscillator, y0, T, args=(n, p, m, F_func_args))
#
# # График
# plt.plot(T, sol[:, 0], 'b', label='x(t)')
# plt.xlabel('Время')
# plt.ylabel('Положение')
# plt.title('Затухающие и возбужденные колебания с внешней силой')
# plt.legend(loc='best')
# plt.grid()
# plt.show()
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def F_sin(A, w, t):
    return A * np.sin(w * t)

def model(x, t, m, b, k, A, w):
    x1, x2 = x[0], x[1]
    dx1dt = x2
    dx2dt = (F_sin(A, w, t) - b * x2 - k * x1) / m
    return [dx1dt, dx2dt]

A = 10
w = 100
T = np.linspace(0, 10, 1000)
delta_t = T[1] - T[0]
m, b, c = 1, 3, 100
x0, dx0 = 6, 0
x_init = [x0, dx0]

# Решение дифференциального уравнения
x_solution = odeint(model, x_init, T, args=(m, b, c, A, w))

# Извлечение значений x и x' из массива решения
x_values = x_solution[:, 0]  # перемещение x
v_values = x_solution[:, 1]  # скорость x'

# Вывод графика перемещения от времени
plt.plot(T, x_values, label='Перемещение (x)')
plt.xlabel('Время t')
plt.ylabel('Перемещение x')
plt.title('График перемещения от времени')
plt.legend()
# plt.xlim(0.1, 1.1)
plt.grid()
plt.show()

# # Вывод графика скорости от времени
# plt.plot(T, v_values, label='Скорость (v)')
# plt.xlabel('Время t')
# plt.ylabel('Скорость v')
# plt.title('График скорости от времени')
# plt.legend()
# plt.show()
