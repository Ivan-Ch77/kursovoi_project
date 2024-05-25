from numba import njit
import numpy as np
from math import exp, cos, sin, sqrt
import matplotlib.pyplot as plt
from scipy.integrate import odeint

@njit
def sys_param(m, b, c):
    p_sys = np.zeros(4)
    p_sys[0] = m
    p_sys[1] = b/(2*m)  # nx
    p_sys[2] = sqrt(c/m-(b/(2*m))**2)  # p0
    p_sys[3] = sqrt(c/m)  # px
    return (p_sys)
@njit
def x_Duhamel_start(p_sys, t, x0, dx0, F):
    m, n, p0, p = p_sys[0], p_sys[1], p_sys[2], p_sys[3]
    x_ch_n = -exp(-n*t)*F*(cos(p0*t)+n*sin(p0*t)/p0)/(m*p**2) + F/(m*p**2)
    x_o_o = exp(-n*t)*(x0*cos(p0*t)+(dx0+n*x0)*sin(p0*t)/p0)
    return (x_o_o+x_ch_n)
@njit
def dx_Duhamel_start(p_sys, t, x0, dx0, F):
    m, n, p0, p = p_sys[0], p_sys[1], p_sys[2], p_sys[3]
    dx_ch_n = F*exp(-n*t)*sin(p0*t)/(m*p0)
    dx_o_o = (exp(-n*t)*(dx0*p0*cos(p0*t)-(dx0*n+x0*p**2)*sin(p0*t))/p0)
    return (dx_ch_n+dx_o_o)
@njit
def x_Duhamel(p_sys, t, x0, dx0, F_1, F_2):
    a = (F_2-F_1)/t
    b = F_1
    m, n, p0, p = p_sys[0], p_sys[1], p_sys[2], p_sys[3]
    x_ch_n = (-exp(-n*t)*((2*a*n+b*p**2+a*p**2*t)*cos(p0*t)
                          +(b*n*p**2+a*(2*n**2-p**2+n*p**2*t))*sin(p0*t)/p0)/(m*p**4)
              +(2*a*n+b*p**2)/(m*p**4))
    x_o_o = exp(-n*t)*(x0*cos(p0*t)+(dx0+n*x0)*sin(p0*t)/p0)
    return (x_o_o+x_ch_n)
@njit
def dx_Duhamel(p_sys, t, x0, dx0, F_1, F_2):
    a = (F_2-F_1)/t
    b = F_1
    m, n, p0, p = p_sys[0], p_sys[1], p_sys[2], p_sys[3]
    x_ch_n = (exp(-n*t)/(m*p0)*( (b+a*t)*sin(p0*t) ))
    x_o_o = (exp(-n*t)/(p0)*( dx0*p0*cos(p0*t)-(dx0*n+p**2*x0)*sin(p0*t) ))
    return (x_o_o+x_ch_n)

@njit
def F_sin(A, w, t):
    return(A*sin(w*t))
def main():
    A = 10
    w = 100
    T = np.linspace(0, 10, 1000)
    delta_t = T[1]-T[0]
    m, b, c = 1, 3, 100
    p_sys = sys_param(m, b, c)
    X = np.zeros(len(T))
    x0, dx0 = 6, 0
    F_1 = A
    for i in range(len(T)-1):
        F_1 = F_sin(A, w, T[i])
        F_2 = F_sin(A, w, T[i+1])
        X[i+1] = x_Duhamel(p_sys, delta_t, x0, dx0, F_1, F_2)
        x0 = x_Duhamel(p_sys, delta_t, x0, dx0, F_1, F_2)
        dx0 = dx_Duhamel(p_sys, delta_t, x0, dx0, F_1, F_2)
    plt.title("Перемещения")
    plt.xlabel("T")
    plt.ylabel("M")
    # plt.xlim(0.1, 1.1)
    plt.grid()
    plt.plot(T, X)
    plt.show()
    return()

main()
