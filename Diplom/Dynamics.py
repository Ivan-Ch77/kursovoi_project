import math  # Math functionality
import pprint

# from numba import njit
import numpy as np  # Numpy for working with arrays
import matplotlib.pyplot as plt  # Plotting functionality
import sympy as sym

# @njit
def sys_param(m, b, c):
    p_sys = np.zeros(4)
    p_sys[0] = m
    p_sys[1] = b/(2*m)  # n
    p_sys[2] = np.sqrt(c/m)  # p
    p_sys[3] = np.sqrt(c/m-(b/(2*m))**2)  # p0 = sqrt(p**2-n**2)
    print('m=', p_sys[0], 'n=', p_sys[1], 'p=', p_sys[2], 'p0=', p_sys[3])
    return (p_sys)

# def x_Duhamel(p_sys, t, x0, dx0, F_1, F_2):
#     a = (F_2-F_1)/t
#     b = F_1
#     m, n, p, p0 = p_sys[0], p_sys[1], p_sys[2], p_sys[3]
#     x_ch_n = (-np.exp(-n*t)*((2*a*n+b*p**2+a*p**2*t)*np.cos(p0*t)
#                           +(b*n*p**2+a*(2*n**2-p**2+n*p**2*t))*np.sin(p0*t)/p0)/(m*p**4)
#               +(2*a*n+b*p**2)/(m*p**4))
#     x_o_o = np.exp(-n*t)*(x0*np.cos(p0*t)+(dx0+n*x0)*np.sin(p0*t)/p0)
#     return (x_o_o+x_ch_n)

# @njit
def x_Duhamel_start(p_sys, t, x0, dx0, F):
    m, n, p0, p = p_sys[0], p_sys[1], p_sys[2], p_sys[3]
    x_ch_n = -np.exp(-n*t)*F*(np.cos(p0*t)+n*np.sin(p0*t)/p0)/(m*p**2) + F/(m*p**2)
    x_o_o = np.exp(-n*t)*(x0*np.cos(p0*t)+(dx0+n*x0)*np.sin(p0*t)/p0)
    return (x_o_o+x_ch_n)

# @njit
def dx_Duhamel_start(p_sys, t, x0, dx0, F):
    m, n, p0, p = p_sys[0], p_sys[1], p_sys[2], p_sys[3]
    dx_ch_n = F*np.exp(-n*t)*np.sin(p0*t)/(m*p0)
    dx_o_o = (np.exp(-n*t)*(dx0*p0*np.cos(p0*t)-(dx0*n+x0*p**2)*np.sin(p0*t))/p0)
    return (dx_ch_n+dx_o_o)

# @njit
def x_Duhamel(p_sys, t, x0, dx0, F_1, F_2):
    a = (F_2-F_1)/t
    b = F_1
    m, n, p, p0 = p_sys[0], p_sys[1], p_sys[2], p_sys[3]
    # x_ch_n = (-sym.exp(-n*t)*((2*a*n+b*p**2+a*p**2*t)*sym.cos(p0*t)
    #                       +(b*n*p**2+a*(2*n**2-p**2+n*p**2*t))*sym.sin(p0*t)/p0)/(m*p**4)
    #           +(2*a*n+b*p**2)/(m*p**4))
    # x_ch_n = (np.exp(-n*t)*((2*a*n-b*p**2)*np.cos(p0*t)
    #                          +(-b*n*p**2+a*(2*n**2-p**2))*np.sin(p0*t)/p0)/(m*p**4)
    #           +(-2*a*n+b*p**2+a*p**2*t)/(m*p**4))
    # x_ch_n = np.exp(-n*t)/(m*p0)*\
    #          (np.cos(p0*t)*(2*a*n-b*p**2)/p**4+
    #           np.sin(p0*t)/(p0*p**4)*(2*a*n**2-a*p**2-b*n*p**2))+\
    #          1/(m*p**4)*(b*p**2+a*(p**2*t-2*n))

    x_ch_n = (np.exp(-n*t)/(m*p0*p**4))*(np.sin(p0*t)*(2*a*n**2-a*p**2-b*n*p**2)+
                                         p0*np.cos(p0*t)*(2*a*n-b*p**2))+\
             b/(p**2*m)+(a*(p**2*t-2*n))/(p**4*m)  #  Моё
    # x_ch_n = (-np.exp(-n*t)*((2*a*n+b*p**2+a*p**2*t)*np.cos(p0*t)
    #                       +(b*n*p**2+a*(2*n**2-p**2+n*p**2*t))*np.sin(p0*t)/p0)/(m*p**4)
    #           +(2*a*n+b*p**2)/(m*p**4))
             # np.exp(n*t)*p0*(a*p**2*t-2*a*n+b*p**2)   # Марка
    x_o_o = np.exp(-n*t)*(x0*np.cos(p0*t)+(dx0+n*x0)*np.sin(p0*t)/p0)
    return (x_o_o+x_ch_n)

# @njit
def dx_Duhamel(p_sys, t, x0, dx0, F_1, F_2):
    a = (F_2-F_1)/t
    b = F_1
    m, n, p, p0 = p_sys[0], p_sys[1], p_sys[2], p_sys[3]
    # Моё
    # x_ch_n = (np.exp(-n*t)/(m*p0*p**2)*(np.sin(p0*t)*(b*p**2-a*n)-
    #                                     a*p0*np.cos(p0*t)+
    #                                     a*p0**2*np.exp(n*t)))
    # x_o_o = (np.exp(-n*t)/(p0))*(np.sin(p0*t)*(x0*p**2-dx0*n)+dx0*p0*np.cos(p0*t))

    # x_o_o = (np.exp(-n*t)/(p0)*( x0*p0*np.cos(p0*t)+(dx0+p*x0-n**2*x0/p)*np.sin(p0*t) ))  # Моё
    # x_o_o = -(np.exp(-n * t) / (p0) * ((2*n*dx0*p0 + p0*p**2*x0) * np.cos(p0 * t) + (dx0*p**2 - 2 * n**2 * dx0 - p ** 2 * n * x0) * np.sin(p0 * t)))  # Моё
    # x_o_o = (np.exp(-n*t)/(p0)*( dx0*p0*np.cos(p0*t)-(dx0*n+p**2*x0)*np.sin(p0*t) ))

    x_ch_n = (np.exp(-n*t)/(m*p0)*( (b+a*t)*np.sin(p0*t) ))  # верно должно быть(просто подынтегральное выражение в Дюамеле
    # x_o_o = (np.exp(-n*t)/(p0)*( dx0*p0*np.cos(p0*t)+(p**2*x0-dx0*n)*np.sin(p0*t) ))  # Моё
    x_o_o = (np.exp(-n * t) / (p0) * (dx0 * p0 * np.cos(p0 * t) - (dx0 * n + p ** 2 * x0) * np.sin(p0 * t)))  # Марка
    return (x_o_o+x_ch_n)
# sym.init_printing()  Нужно для Юпитера, чтобы красиво все выводилось


m = sym.Symbol('m')
p0 = sym.Symbol('p0')
p = sym.Symbol('p')
F1 = sym.Symbol('F1')
F2 = sym.Symbol('F2')
dt = sym.Symbol('dt')
tau = sym.Symbol('tau')
t = sym.Symbol('t')
x0 = sym.Symbol('x0')
dx0 = sym.Symbol('dx0')
n = sym.Symbol('n')
a = sym.Symbol('a')
b = sym.Symbol('b')





# print('''Случай значимого демпфирования (неконсервативная система)\nПодынтегральное выражение:''')
# print((F1 + ((F2-F1)/dt)*tau)*sym.sin(p0*(t-tau)))

# # Случай незначительного демпфирования, т.е. система консервативна
# f1 = sym.sin(p0*(t - tau))  # Подынтегральная функция в интеграле Дюамеля для x1(t)
#
# f2 = tau * sym.sin(p0*(t - tau))  # Подынтегральная функция в интеграле Дюамеля для x2(t)
# f12 = (F1 + ((F2-F1)/dt)*tau)*sym.sin(p0*(t-tau))
# defInt1 = sym.integrate(f1, (tau, 0, t))  # Решение интеграла(def - definition) 1
# defInt2 = sym.integrate(f2, (tau, 0, t))  # Решение интеграла(def - definition) 2
# defInt12 = sym.integrate(f12, (tau, 0, t))
#
# print('Посчитали интеграл Дюамеля 1 и упростили запись')
# print(sym.simplify(defInt1))
#
# print('Посчитали интеграл Дюамеля 2 и упростили запись')
# print(sym.simplify(defInt2))
#
# print('Посчитали интеграл Дюамеля 12 и упростили запись')
# print(sym.simplify(defInt12))
#
# Общее однородное
# # Другое решение может быть оно правильное (случай малого n => exp**(...) = 1)
# f3 = x0*sym.cos(p0*(t-tau))+((dx0+n*x0)/p0)*sym.sin(p0*(t-tau))
# defInt3 = sym.integrate(f3, (tau, 0, t))  # Решение интеграла(def - definition) 3
#
# sym.simplify(defInt3)


##################################
# print('''Случай значимого демпфирования (неконсервативная система)\nПодынтегральное выражение:''')
# print(sym.exp(-n*(t-tau))*(F1 + ((F2-F1)/dt)*tau)*sym.sin(p0*(t-tau)))
#
# f4 = sym.exp(-n*(t-tau))*(a*tau + b)*sym.sin(p0*(t-tau))
#
# defInt4 = sym.integrate(f4, (tau, 0, t))  # Решение исходного интеграла
#################################

# def F_sin(A, w, t):
#     return(A*sym.sin(w*t))

# @njit
# def F_sin(A, w, t):
#     return(A*np.sin(w*t))
#     # return 100

# A = 10
# w = 100
# T = np.linspace(0, 10, 50000)
# delta_t = T[1]-T[0]
# m, b, c = 1, 0, 100
# p_sys = sys_param(m, b, c)
# X = np.zeros(len(T))
# dX = np.zeros(len(T))
# x0, dx0 = 0, 0
# F_1 = A

# Integr = sym.exp(-n * t) * (x0 * sym.cos(p0 * t) + (dx0 + n * x0) * sym.sin(p0 * t) / p0) + (
#         sym.exp(-n * t) / (m * p0 * p ** 4)) * (
#                  sym.sin(p0 * t) * (2 * a * n ** 2 - a * p ** 2 - b * n * p ** 2) + p0 * (
#                  sym.cos(p0 * t) * (2 * a * n - b * p ** 2) + sym.exp(n * t) * p0 * (
#                  a * p ** 2 * t - 2 * a * n + b * p ** 2)))
# dx_dt = sym.diff(Integr, t)
# print(dx_dt)

# tau = sym.Symbol('tau')

# for i in range(len(T)-1):
#     F_1 = F_sin(A, w, T[i])
#     F_2 = F_sin(A, w, T[i+1])
#     dt = T[i+1] - T[i]
#     # t = T[-1]
#     X[i+1] = x_Duhamel(p_sys, delta_t, x0, dx0, F_1, F_2)
#     dX[i+1] = dx_Duhamel(p_sys, delta_t, x0, dx0, F_1, F_2)
#     x0 = X[i+1]
#     dx0 = dX[i+1]
# # pprint.pprint(X)
# plt.title("Перемещения")
# plt.xlabel("T")
# plt.ylabel("M")
# # plt.xlim(0.1, 1.1)
# # plt.ylim(-1.8, 1.8)
# plt.grid()
# plt.plot(T, X)
# plt.show()
# plt.title("Виброскорость")
# plt.xlabel("T")
# plt.ylabel("M/c")
# # plt.xlim(0.1, 0.7)
# # plt.ylim(-1.8, 1.8)
# plt.grid()
# plt.plot(T, dX)
# plt.show()
