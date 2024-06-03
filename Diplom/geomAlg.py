import pprint
# from main import n, D, a, Hr, omega, oomega, f, b, dx, b_workpiece
from Dynamics import x_Duhamel_start, dx_Duhamel_start, x_Duhamel, dx_Duhamel, sys_param
import numpy as np
import matplotlib.pyplot as plt
# from main import step
import Dynamics


# Определяем параметры фрезы и резания
n = 2  # Кол-во зубьев фрезы [безразм]
D = 6.0  # Диаметр фрезы [мм]
a = 4.0  # Ширина резания [мм]
Hr = 1.0  # Радиальная глубина резания [мм]
omega = 1684  # Скорость вращения шпинделя [об/мин]
f = 40.0  # Подача на зуб [мкм]
bx = 150.0  # Длина заготовки [мм]
dx = 0.005  # Разбиение детали dx [мм]

#  Угол вступления в первый контакт фрезы и заготовки
# alpha = np.arccos(1 - 2*Hr/D)  ??????

T = 1 / (omega / 60)  # Период одного оборота шпинделя
print('T = ', T)
oomega = 2 * np.pi / T  # Угловая скорость рад/с
print('oomega = ', oomega)


# Высота заготовки(для примера) [мм]
b_workpiece = 50.0

# Скорость подачи стола [мм/с]. В моем примере 12.8 мм/с
V_f = f * n * omega / (60 * (10**3))
print("V_f = ", V_f)

# Начальные координаты фрезы (ноль СК - это левый нижний угол заготовки)
tool_start_y = b_workpiece - Hr + D/2
# tool_start_x = - D/2 * np.sin(alpha)
# tool_start_x = -D/2
tool_start_x = 10
print("tool_start_x = ", tool_start_x)
print("tool_start_y = ", tool_start_y)

# Задаем z-буфер
# num_points = int(b / dx)  # Кол-во отрезков разбиения
# x_values = np.arange(num_points) * dx
# buffer_list = np.column_stack((x_values, np.full(num_points, b_workpiece)))
# buffer_list_test = np.column_stack((x_values, np.full(num_points, b_workpiece)))

# pprint.pprint(x_values)
# pprint.pprint(buffer_list)
# pprint.pprint(buffer_list_test)

# temporary_buffer = []  # Временный буфер для реализации запаздывания
#
# Часть с зацикливанием, можем взять любое dt и t и тем самым получим процесс фрезерования
step = 60/omega/200  # Шаг по времени (т.к. мы взяли 4800 об/мин -> 1/200 оборота время step)
# dt = 0.4  # Через сколько остановить выполнение программы
# t = 0  # Начальный момент времени(потом будет изменяться с каждым шагом)

# Будем считать, что толщина срезаемого слоя снимается для каждой режущей кромки т.е. будем так их хранить:
# thickness_list = {1: [...], 2: [...], ... }
# где 1, 2, ... - номер режущей кромки, а [...] - список толщин срезаемого слоя, в нем есть нулевые элементы
# thickness_list = {}


# for i in range(n):
#     thickness_list[i] = [0] * int(dt / step + 1)  # Задаем длину списка в словаре толщин для каждой режущей кромки
# pprint.pprint(thickness_list)
# finish = t + dt  # Конечный момент времени
# count_len_thikness_list = 0  # Ведем подсчет какой сейчас номер шага (всего шагов = dt / step)
# Т.к. у нас время дискретно => мы разбиваем t + dt на шаги по времени
# и будем для каждого такого шага искать толщину срезаемого слоя


# Будем считать, что окружные и радиальные силы снимаются для каждой режущей кромки т.е. будем так их хранить:
# fenom_list = {1: [[Fr, Ft],[Fr,Ft], [Fr,Ft],...], 2: [[Fr, Ft],[Fr,Ft], [Fr,Ft],...], ... }
# где 1, 2, ... - номер режущей кромки, а [...] - список толщин срезаемого слоя, в нем есть нулевые элементы
# fenom_list = {}
#
# for i in range(n):
#     fenom_list[i] = [[0, 0]] * int(dt / step + 1)  # Задаем длину списка в словаре толщин для каждой режущей кромки
# pprint.pprint(fenom_list)

# finish = t + dt  # Конечный момент времени
# count_len_thikness_list = 0  # Ведем подсчет какой сейчас номер шага (всего шагов = dt / step)
# # Т.к. у нас время дискретно => мы разбиваем t + dt на шаги по времени
# # и будем для каждого такого шага искать толщину срезаемого слоя
#
#
# # Запаздывание
# lag = 30
#
#
# # Феноменологическая модель
Krc = 558.0  # [Н/мм^2]
Ktc = 970.0  # [Н/мм^2]
Krb = 2.0  # [Н/мм^2]
Ktb = 7.7  # [Н/мм^2]

# A = 10
# w = 100
# T = np.linspace(0, finish, int(finish/step))
# delta_t = T[1]-T[0]  # step

#модальные характеристики
# m_x=2.3*10**6/((73*2*pi)**2)
# k_x = 2.3*10**6
# b_x=2*n_x*m_x

# m, b, c = 1, 0, 100 # Коммент
m = 2.3*10**6/((73*2*np.pi)**2)
c = 2.3*10**6
b = 2*m*0.0057*73*2*np.pi

p_sys = sys_param(m, b, c)

print(p_sys)


# def buf(points1, x1, y1, x2, y2):
#     '''Построение буфера и двух положений режущей кромки относительно z-буфера'''
#     X1, Y1 = zip(*points1)
#     # Строим вторую фигуру - соединенные точки
#     plt.plot(X1, Y1, label='Заготовка', linestyle='dashed')
#
#     plt.scatter(x1, y1)
#     plt.scatter(x2, y2)
#
#     # Добавляем подписи и легенду
#     plt.title('Моделирование фрезерования')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.legend()
#
#
#     plt.xlim(1.7, 2.1)
#     plt.ylim(46.5, 47.5)
#     # Показываем график
#     plt.show()

def plot_figures(t, buffer, tooth_cord):
    '''
    t - время для которого строим положение фрезы и состояние заготовки
    buffer - z-буфер
    '''
    # '''Здесь происходит пострение заготовки в момент времени t+dt и фрезы в моменты времени t и t+dt
    # points1 - список координат фрезы [[x1, y1] , [x2, y2], ..., [xn, yn]], где n - кол-во зубьев фрезы
    # points2 список координат z-буфера вида: [..., [xi, yi],...], его длина равна b_заготовки / dx_буфера'''
    points1 = tooth_cord
    points2 = buffer
    # Извлекаем координаты x и y из списков точек
    x1, y1 = zip(*points1)
    X2, Y2 = zip(*points2)
    x3, y3 = zip(*tooth_cord)

    # Строим первую фигуру - набор точек
    plt.scatter(x1, y1, label='Фреза t=0.4')
    plt.scatter(x3, y3, label='Фреза t=0')

    # Строим вторую фигуру - соединенные точки
    plt.plot(X2, Y2, label='Заготовка', linestyle='dashed')

    # Добавляем подписи и легенду
    plt.title('Моделирование фрезерования')
    plt.xlabel('X, мм')
    plt.ylabel('Z, мм')
    plt.legend()

    # Делаем одинаковый масштаб
    plt.axis('equal')

    plt.xlim(-5, 40)
    # Показываем график
    plt.show()


# def plot_situation(crossdot, cuttingpoint, millingcenter, i, m1, b1, m2, b2, buffer):
#
#     X2, Y2 = zip(*buffer)
#     # Строим вторую фигуру - соединенные точки
#     plt.plot(X2, Y2, label='Заготовка', linestyle='dashed')
#
#     # Координаты точек
#     x_coords = [crossdot[0], cuttingpoint[0], millingcenter[0]]
#     y_coords = [crossdot[1], cuttingpoint[1], millingcenter[1]]
#
#     # Задаем диапазон x для построения прямых
#     x_values = range(int(millingcenter[0])-2, int(crossdot[0])+2)
#
#     # Вычисляем y для каждой прямой
#     y_values1 = [m1 * x + b1 for x in x_values]
#     y_values2 = [m2 * x + b2 for x in x_values]
#
#     # Построение графика
#     plt.plot(x_values, y_values1, label='Line 1')
#     plt.plot(x_values, y_values2, label='Line 2')
#
#     # Цвета и подписи для каждой точки
#     colors = ['red', 'blue', 'green']
#     labels = ['ТчкПер', 'КРежКрки', 'координаты центра фрезы']
#
#     # Построение точек
#     plt.scatter(x_coords, y_coords, c=colors, label=labels)
#
#     # Координаты центра окружности
#     circle_center = [millingcenter[0], millingcenter[1]]
#     circle_radius = D/2
#
#     # Построение точек
#     plt.scatter(x_coords, y_coords, c=colors, label=labels)
#
#     # Добавление подписей к точкам
#     for label, x, y in zip(labels, x_coords, y_coords):
#         plt.text(x, y, label, fontsize=12, ha='right')
#
#     # Построение окружности
#     circle = plt.Circle((circle_center[0], circle_center[1]), circle_radius, color='orange', fill=False)
#     plt.gca().add_patch(circle)
#
#     # Настройка графика
#     plt.title(f'График точек с подписями и цветами для {i} режущей кромки')
#     # plt.xlabel('X-координата')
#     # plt.ylabel('Y-координата')
#     # plt.legend()
#
#     # # Ограничиваем вывод графика, чтобы детальнее его рассмотреть
#     plt.xlim(1, 5.5)
#     plt.ylim(47, 50.25)
#
#     # Отображение графика
#     plt.show()


def plot_force(step, fenom_list, number=-1):
    ''' Функция строит графики сил Fx и Fy от времени
    На вход она принимает: step - шаг разбиения по времени
                           fenom_list - словарь с феноменами для каждой режущей кромки
                           number - номер режущей кромки, для которой нужно вывести силы,
                           если number = -1 - выводим для всех кромок (отсчет идет от 0 кромки)'''
    # Создание списка значений x для построения графика
    if number != -1:
        x = [i * step for i in range(1, len(fenom_list[number]) + 1)]
    else:
        x = [i * step for i in range(1, len(fenom_list[0]) + 1)]
        Fx_data = [0] * int(len(fenom_list[0]))
        Fy_data = [0] * int(len(fenom_list[0]))
    # Построение графиков для Fx и Fy
    for key in fenom_list:
        if number == -1 or key == number:
            fenom = fenom_list[key]
            Fx = [item[0] for item in fenom]
            Fy = [item[1] for item in fenom]
            for i in range(len(Fx_data)):
                if Fx[i] != 0:
                    Fx_data[i] = Fx[i]
                if Fy[i] != 0:
                    Fy_data[i] = Fy[i]


    # Построение графика Fx
    plt.figure()
    plt.plot(x, Fx_data)
    plt.title('График силы Fx в зависимости от времени t')
    plt.xlabel('t, с')
    plt.ylabel('Сила, Н')
    # plt.xlim(0.17, 0.21)
    plt.show()

    # Построение графика Fy
    plt.figure()
    plt.plot(x, Fy_data)
    plt.title('График силы Fy в зависимости от времени t')
    plt.xlabel('t, с')
    plt.ylabel('Сила, Н')
    # plt.xlim(0.17, 0.21)
    plt.show()

    # Построение графиков для Fx и Fy
    for key in fenom_list:
        if number == -1 or key == number:
            fenom = fenom_list[key]
            Fx = [item[0] for item in fenom]
            plt.plot(x, Fx, label=f'Fx (кромка {key})')

    # Добавление заголовка и меток осей
    plt.title('График силы Fx в зависимости от времени t')
    plt.xlabel('t, с')
    plt.ylabel('Сила, Н')

    # plt.xlim(0.17, 0.21)

    # Добавление легенды
    plt.legend()

    # Отображение графика
    plt.show()

    # Построение графиков для Fx и Fy
    for key in fenom_list:
        if number == -1 or key == number:
            fenom = fenom_list[key]
            Fy = [item[1] for item in fenom]
            plt.plot(x, Fy, label=f'Fy (кромка {key})')

    # Добавление заголовка и меток осей
    plt.title('График силы Fz в зависимости от времени t')
    plt.xlabel('t, с')
    plt.ylabel('Сила, Н')

    # plt.xlim(0.17, 0.21)

    # Добавление легенды
    plt.legend()

    # Отображение графика
    plt.show()



# def fenomenol_force(step, fenom_list, number=-1):
#     ''' Функция строит график окружной и радиальной силы резания от времени
#     На вход она принимает: step - шаг разбиения по времени
#                            fenom_list - словарь со списком списков радиальных и окружных сил для каждой режущей кромки
#                            number - номер режущей кромки, для которой нужно вывести толщины,
#                            если number = -1 - выводим для всех кромок (отсчет идет от 0 кромки)'''
#     if number != -1:
#             x1 = [i * step for i in range(1, len(fenom_list[number]) + 1)]  # Fr
#             y1 = thickness_list[number][0]  # Fr
#             x2 = [i * step for i in range(1, len(fenom_list[number]) + 1)]  # Ft
#             y2 = thickness_list[number][1]  # Ft
#     else:
#         x = [i * step for i in range(1, len(fenom_list[0]) + 1)]
#         y1 = []  # Fr
#         y2 = []  # Ft
#         max_length = max(len(lst) for lst in thickness_list.values())
#
#         # Идем по индексам от 0 до max_length - 1
#         for i in range(max_length):
#             found = False
#             for key in fenom_list:
#                 if len(fenom_list[key]) > i and fenom_list[key][i][0] != 0:
#                     # if i == 1777:
#                     y1.append(thickness_list[key][i])
#                     found = True
#                     break
#                 elif
#             if not found:
#                 y.append(0)

def plot_thickness(step, thickness_list, number=-1):
    ''' Функция строит график толщины срезаемого слоя от времени
    На вход она принимает: step - шаг разбиения по времени
                           thickness_list - словарь со списком толщин для каждой режущей кромки
                           number - номер режущей кромки, для которой нужно вывести толщины,
                           если number = -1 - выводим для всех кромок (отсчет идет от 0 кромки)'''
    # Создание списка значений x и y для построения графика
    if number != -1:
            x = [i * step for i in range(1, len(thickness_list[number]) + 1)]
            y = thickness_list[number]
    else:
        x = [i * step for i in range(1, len(thickness_list[0]) + 1)]
        y = []
        max_length = max(len(lst) for lst in thickness_list.values())

        # Идем по индексам от 0 до max_length - 1
        for i in range(max_length):
            found = False
            for key in thickness_list:
                if len(thickness_list[key]) > i and thickness_list[key][i] != 0:
                # if i == 1777:
                    y.append(thickness_list[key][i])
                    found = True
                    break
            if not found:
                y.append(0)

    # Построение точечного графика
    plt.scatter(x, y, color='blue')

    # Соединение точек прямыми линиями
    plt.plot(x, y, color='red', linestyle='solid')

    # Добавление заголовка и меток осей
    plt.title('График толщины срезаемого слоя в зависимости от времени t')
    plt.xlabel('t, с')
    plt.ylabel('h, мм')

    # # Добавление легенды
    # plt.legend()

    # Ограничиваем вывод графика, чтобы детальнее его рассмотреть
    # plt.xlim(0.17, 0.175)
    # plt.ylim(0, 0.08)

    # Отображение графика
    plt.show()


def k_xy(x1, y1, x2, y2):
    return (y2 - y1)/(x2 - x1)

def b_xy(x1, y1, x2, y2):
    # b1 = (y1*x2 - y2*x1)/(x2-x1)
    k = k_xy(x1, y1, x2, y2)
    b_ = y1 - x1*k
    return b_

def xy(m1, b1, m2, b2):
    if m1 == m2:
        # Прямые параллельны, нет точки пересечения
        return None
    x = (b2 - b1) / (m1 - m2)
    y = (b1 * m2 - b2 * m1) / (m2 - m1)
    return x, y

def distance_between_points(x1, y1, x2, y2):
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def fenom_model(h):
    # Fr = Krc * a * h + Krb * a  # Такую формулу использовать?
    # Ft = Ktc * a * h + Ktb * a  # Такую формулу использовать?
    Fr = Krc * a * h  # Такую формулу использовать?
    Ft = Ktc * a * h  # Такую формулу использовать?
    return [Fr, Ft]

def decart_fenom_model(cosfi, sinfi, FrFt: list):
    Fr, Ft = FrFt[0], FrFt[1]
    Fx = Fr * cosfi + Ft * sinfi
    Fz = Ft * cosfi - Fr * sinfi
    return Fx, Fz

# def tooth_coord(t, xDuh_c, yDuh_c):
#     """Функция, которая на вход получает время t и возвращает список координат вершин зубьев фрезы через время t"""
#     angle_step = 2 * np.pi / n
#     coord = []
#     tool_x = tool_start_x + V_f*t - xDuh_c  # Координата x центра фрезы через время t
#     tool_y = tool_start_y - yDuh_c  # Координата y центра фрезы через время t
#     for i in range(n):
#         angle = i * angle_step + oomega * t
#         x = tool_x + D / 2 * np.cos(angle)
#         y = tool_y + D / 2 * np.sin(angle)
#         coord.append([x, y])
#     return coord

def find_thickness(t, buffer, count, fenlist, forces, thicklist, xyDuh, dxdyDuh):
    '''Функция поиска толщины срезаемого слоя
    buffer - z-буфер поверхности
    t - начальный момент времени. Нужен, чтобы построить прямую y = mx+b'''

    # Список для возврата tooth_coord(t)
    coord = []

    # Отладка по перемещениям x
    X = []

    # Начальные условия на момент шага t
    # x0_Duh, y0_Duh = xyDuh[0], xyDuh[1]
    # dx0_Duh, dy0_Duh = dxdyDuh[0], dxdyDuh[1]
    x0_Duh = xyDuh[0]
    dx0_Duh = dxdyDuh[0]
    #
    # # xDuh_c, yDuh_c = 0, 0
    # xDuh_c = 0
    #
    # # Сюда будем записывать суммарные перемещения по x и по y на конец предыдущего шага
    # F_sum0 = np.zeros(2)
    # Ft = F_sum0
    #
    # # Fi = np.sqrt(Ft[0] ** 2 + Ft[1] ** 2)
    # Fi = Ft[0]
    # Fi_1 = 2 * Fi + 1
    #
    # F_proc = 1
    eps = 0.01
    x0_Duh = xyDuh[0]
    dx0_Duh = dxdyDuh[0]
    Fxy = np.zeros(2)
    Ft = Fxy
    Fi = np.sqrt(Ft[0] ** 2 + Ft[1] ** 2)
    Fi_1 = 2 * Fi + 1
    F_proc = 1
    F_sum0 = np.zeros(2)

    iteration_number = 0
    # while abs(Fi_1 - Fi) / F_proc > eps:
    #     if iteration_number == 0:
    #         xc_Duhamel = x_Duhamel_start(p_sys, step, x0_Duh, dx0_Duh, Ft[0])
    #         # yc_Duhamel = x_Duhamel_start(p_sys, step, y0_Duh, dy0_Duh, Ft[1])
    #     else:
    #         xc_Duhamel = x_Duhamel(p_sys, step, x0_Duh, dx0_Duh, Ft[0], Fi_1)
    #         # yc_Duhamel = x_Duhamel(p_sys, step, y0_Duh, dy0_Duh, Ft[1], Fi_1)

    # while abs(Fi_1-Fi)/F_proc > eps:
    #     if iteration_number == 0:
    #         xc_Duhamel = x_Duhamel_start(p_sys, step, x0_Duh, dx0_Duh, Ft[0])
    #         # yc_Duhamel = x_Duhamel_start(p_sys, step, y0_Duh, dy0_Duh, Ft[1])
    #     else:
    #         xc_Duhamel = x_Duhamel(p_sys, step, x0_Duh, dx0_Duh, Ft[0], Fi_1)
    #         # yc_Duhamel = x_Duhamel(p_sys, step, y0_Duh, dy0_Duh, Ft[1], Fi_1)
    #
    #     # Задаем положение центра фрезы
    #     tool_x = tool_start_x +V_f*t - xc_Duhamel*10**3
    #     tool_y = tool_start_y + V_f * t
    # tool_y = tool_start_y + V_f * t - yc_Duhamel*10**3


    # Проходим по каждой режущей кромке
    for i in range(n):
        Fxy = fenlist[i][count]
        Ft = Fxy
        # Для i-ой режущей кромки определяем перемещения x y из интеграла Дюамеля
        if count == 0:
            Fi = np.sqrt(Ft[0] ** 2 + Ft[1] ** 2)
            Fi_1 = Fi
            xc_Duhamel = x_Duhamel_start(p_sys, step, x0_Duh, dx0_Duh, Ft[0])
            # yDuh_c = x_Duhamel_start(p_sys, step, x0_Duh, dx0_Duh, Ft[1])
        elif count >= 1:
            F_sum0[0] = fenlist[i][count - 1][0]
            F_sum0[1] = fenlist[i][count - 1][1]
            Ft[0] = F_sum0[0]  # Конец предыдущего шага
            Ft[1] = F_sum0[1]  # Конец предыдущего шага
            Fi_1 = np.sqrt(Ft[0]**2+Ft[1]**2)  # конец текущего шага
            xc_Duhamel = x_Duhamel(p_sys, step, x0_Duh, dx0_Duh, Ft[0], Fi_1)
            yDuh_c = x_Duhamel(p_sys, step, x0_Duh, dx0_Duh, Ft[1], Fi_1)
        print("xc_Duhamel=", xc_Duhamel)
        X.append(xc_Duhamel)
        # Находим координаты центра фрезы и положение i-ой режущей кромки
        angle_step = 2 * np.pi / n
        tool_x = tool_start_x + V_f * t - xc_Duhamel*10**3  # Координата x центра фрезы через время t
        tool_y = tool_start_y  # Координата y центра фрезы через время t
        angle = i * angle_step + oomega * t
        x_tooth = tool_x + D / 2 * np.cos(angle)
        y_tooth = tool_y + D / 2 * np.sin(angle)
        # y_tooth = tool_y + D / 2 * np.sin(angle) - yDuh_c*10**3
        coord.append([x_tooth, y_tooth])


        # x_tooth = tooth_coordinate[i][0]
        # y_tooth = tooth_coordinate[i][1]

        intersection_point = None

        # xi_new_old = tooth_coordinate1[i][0]
        # yi_new_old = tooth_coordinate1[i][1]

        j = int(x_tooth // dx)  # int всегда округляет вниз, так что это нам подходит
        xj = buffer[j][0]
        yj = buffer[j][1]
        xj1 = buffer[j+1][0]
        yj1 = buffer[j+1][1]

        # Ищем уравнение прямой, проходящей через эти точки (y = mx + b)
        try:
            mj = k_xy(xj, yj, xj1, yj1)
            bj = b_xy(xj, yj, xj1, yj1)
        except RuntimeWarning:
            print('RuntimeWarning')

        # Здесь нужно, чтобы yi_new была меньше y = mj*xi_new + bj для j и j+1 координат z-буфера
        if y_tooth < mj*x_tooth + bj < b_workpiece:
            # Определение коэф-ов для режущей кромки
            # tool_x = tool_start_x + V_f * t - xDuh_c  # Координата x центра фрезы через время t
            # tool_y = tool_start_y - yDuh_c  # Координата y центра фрезы через время t
            m_tooth = k_xy(x_tooth, y_tooth, tool_x, tool_y)
            b_tooth = b_xy(x_tooth, y_tooth, tool_x, tool_y)
            # Определение коэф-ов для z-буффера
            m_buffer = k_xy(buffer[j][0], buffer[j][1], buffer[j+1][0], buffer[j+1][1])
            b_buffer = b_xy(buffer[j][0], buffer[j][1], buffer[j+1][0], buffer[j+1][1])

            if m_buffer < 0:
                # print('ЧУШЬ!')
                continue

            intersection_point = xy(m_tooth, b_tooth, m_buffer, b_buffer)

            # Часть кода, котора исключает неточности, связанные с разбиением, т.к. иногда может возникнуть ситуация,
            # когда точка пересечения intersection_point и buffer[j][0] находятся в разных ячейках z-буфера
            # (т.е. точка пересечения лежит не в buffer[j], а в j-1 или еще дальше в зависимости от параметров резания)
            # , тогда нужно искать точку пересечения не с j-ой поверхностью заготовки, а с j - k поверхностью заготовки
            k = j - int(tool_x // dx)
            if intersection_point[0] < buffer[j][0]:
                k1 = 0
                while True:
                    k1 += 1
                    # m_buffer = (buffer[j - k][1] - buffer[j - k-1][1]) / (buffer[j-k][0] - buffer[j - k-1][0])
                    # b_buffer = buffer[j - k-1][1] - m_buffer * buffer[j - k-1][0]
                    m_buffer = k_xy(buffer[j-k1][0], buffer[j-k1][1], buffer[j-k1+1][0], buffer[j-k1+1][1])
                    b_buffer = b_xy(buffer[j-k1][0], buffer[j-k1][1], buffer[j-k1+1][0], buffer[j-k1+1][1])

                    intersection_point = xy(m_tooth, b_tooth, m_buffer, b_buffer)

                    if intersection_point[0] >= buffer[j - k1][0] or k1 > k:
                        break

            point1 = intersection_point  # Точка пересечения
            # point2 = [x_tooth, y_tooth]  # Координаты режущей кромки
            # point3 = [xi_new_old, yi_new_old]
            # point4 = [tool_x, tool_y]  # Координаты центра фрезы

            # X = [point1[0], point2[0], point3[0], point4[0]]
            # Y = [point1[1], point2[1], point3[1], point4[1]]

            if intersection_point is not None:
                distance = distance_between_points(point1[0], point1[1], x_tooth, y_tooth)
                # print(distance)

                #  Обрубаем то что больше f
                # if distance * 1000 > f:
                #     distance = f / 1000

                # Записываем для i-ой режущей кромки толщину срезаемого слоя в словарь thickness_list
                thicklist[i][count] = distance
                print(distance)
                FrFt = fenom_model(distance)
                cosfi = 2 * (x_tooth - tool_x) / D
                sinfi = 2 * (y_tooth - tool_y) / D
                Fx, Fy = decart_fenom_model(cosfi, sinfi, FrFt)
                fenlist[i][count] = [Fx, Fy]  # Находим силы Fx, Fy и записываем их в итоговый словарь списков [Fx, Fy]
                F_sum0[0] += Fx
                # F_sum0[1] += Fy
                F_sum0[1] += 0

                # plot_points(X, Y, buffer_list)

                # if distance * 1000 > 1.2*f and 0.111 < t < 0.112 and t//step == 1777:
                #     print(f'{distance * 1000} мкм > {f} мкм ----------- ОШИБКА!!!')
                #     print('Точка пересечения', point1)
                #     print('Координаты режущей кромки', point2)
                #     print('Координаты центра фрезы', point4)
                #     # print(tooth_coordinate)
                #     print(np.sqrt((point2[0] - point4[0])**2 + (point2[1] - point4[1]) ** 2))
                #     print('distance', distance)
                #     print(i)
                #     print(t)
                #     print('Номер шага: ', t // step)
                    # if 0.111 < t < 0.113:
                    #     pass
                    # plot_situation(point1, point2, point4, i, m_tooth, b_tooth, m_buffer, b_buffer, buffer)
                #     # zds = True

            else:
                print("Прямые параллельны, невозможно найти расстояние")
        else:
            distance = 0
            thicklist[i][count] = distance
            FrFt = fenom_model(distance)
            cosfi = 2 * (x_tooth - tool_x) / D
            sinfi = 2 * (y_tooth - tool_y) / D
            Fx, Fy = decart_fenom_model(cosfi, sinfi, FrFt)
            fenlist[i][count] = [Fx, Fy]  # Находим силы Fr, Ft и записываем их
            # fenlist[i][count] = fenlist(distance)  # Находим силы Fr, Ft и записываем их в dict
            # print('Зуб фрезы не в заготовке')
    if iteration_number == 1:
        dx0_Duh = dx_Duhamel_start(p_sys, step, x0_Duh, dx0_Duh, Ft[0])
        # dy0_Duh = dx_Duhamel_start(p_sys, step, y0_Duh, dy0_Duh, Ft[1])
    else:
        dx0_Duh = dx_Duhamel(p_sys, step, x0_Duh, dx0_Duh, Ft[0], F_sum0[0])
        # dy0_Duh = dx_Duhamel(p_sys, step, y0_Duh, dy0_Duh, Ft[1], F_sum0[1])

    yDuh_c = 0
    result_xyDuh = [xc_Duhamel, yDuh_c]
    dy0_Duh = 0
    result_dxdyDuh = [dx0_Duh, dy0_Duh]
    forces[count] = F_sum0
    return thicklist, fenlist, coord, result_xyDuh, result_dxdyDuh, forces


def part_of_cutting(temp_buf0, temp_buf1, buffer):
    """Функция, которая отвечает за резание заготовки.
        На вход получает buffer - z-буфер
                         temp_buf0 - положение зубьев фрезы в момент времени t - lag*step
                         temp_buf1 - положение зубьев фрезы в момент времени t - lag*step + 1
                         lag - кол-во шагов запаздывания"""

    # Один раз пробегаюсь по всем режущим кромкам
    for i in range(n):

        # Определяем координаты режущих кромок в двух положениях
        xi = temp_buf0[i][0]
        yi = temp_buf0[i][1]
        xi_new = temp_buf1[i][0]
        yi_new = temp_buf1[i][1]

        try:
            m = k_xy(xi,yi,xi_new,yi_new)
            b1 = b_xy(xi,yi,xi_new,yi_new)
        except RuntimeWarning:
            print('RuntimeWarning')
        except ZeroDivisionError:
            print(xi, yi, xi_new, yi_new)

        # Находим ячейку, в которой в момент времени t - (lag-1)*step остановилась фреза
        j = int(xi_new // dx)
        j0 = int(xi // dx)
        # Определяем координаты x,y, которые отвечают координатам начала ячейки
        # Изменяем координаты y точек буфера, лежащих между xi и xi_new
        for i in range(int(j - j0)):
            x = buffer[j][0]
            y = buffer[j][1]
            if xi <= x <= xi_new and y > m*x + b1:
                # if t // step == 1777:
                #     buf(buffer_list, xi, yi, xi_new, yi_new)
                buffer[j] = (x, m*x + b1)
                # if t // step == 1777:
                #     buf(buffer_list, xi, yi, xi_new, yi_new)
            j -= 1
    return buffer

# while True:
#     if t >= finish:
#         plot_figures(t)  # Строим заготовку и фрезу
#         break
#     else:
#
#         # Ищем толщину срезаемого слоя для z-буфера и времени t
#         find_thickness(t)
#
#         temporary_buffer.append(tooth_coord(t))
#
#         if len(temporary_buffer) == lag:  # Когда кол-во элементов в хранилище станет равно запаздыванию
#             # Изменяем координаты z-буфера в соответствии с 0 и 1 значениями, хранящимися во временном хранилище
#             part_of_cutting(temporary_buffer[0], temporary_buffer[1])
#             # print(temporary_buffer[0], temporary_buffer[1])
#             del temporary_buffer[0]  # Удаляем 0 элемент, тем самым сдвигаем все элементы хранилища влево на один
#
#             # if t > 0.3 and zds:
#                     # print(t / step)
#                     # break
#
#         # Добавляем step ко времени, переходим к следующему шагу
#         t += step
#         count_len_thikness_list += 1  # Необходимо, чтобы в словаре thickness_list пройти от 0 до dt / step + 1
#         if 99 * step < t < 100*step:
#             print('dsdwawd', fenom_model(t))

# print(fenom_list)
# plot_thickness(step, thickness_list, -1)  # Построение графика толщины срезаемого слоя
# plot_force(step, fenom_list, -1)
# plot_thickness(step, thickness_list, 0)  # Построение графика толщины срезаемого слоя
# plot_thickness(step, thickness_list, 1)  # Построение графика толщины срезаемого слоя
# plot_thickness(step, thickness_list, 2)  # Построение графика толщины срезаемого слоя
# plot_thickness(step, thickness_list, 3)  # Построение графика толщины срезаемого слоя
