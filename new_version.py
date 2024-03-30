# import numpy as np
# import matplotlib.pyplot as plt
# # import matplotlib.patches as patches
# # from ipywidgets import interact
# # from matplotlib.widgets import Slider
#
# # Определяем параметры фрезы и резания
# n = 4  # Кол-во зубьев фрезы [безразм]
# D = 6.0  # Диаметр фрезы [мм]
# a = 4.0  # Ширина резания [мм]
# Hr = 3.0  # Радиальная глубина резания [мм]
# omega = 4800  # Скорость вращения шпинделя [об/мин]
# f = 40.0  # Подача на зуб [мкм]
# b = 150  # Длина заготовки [мм]
# # dt = 10  # Время, которое прошло с момента запуска фрезы [с]
# dx = 0.2  # Разбиение детали dx [мм]
#
# # t0 = 0  # Начало испытаний(время для слайдера)
#
# #  Угол вступления в первый контакт фрезы и заготовки
# alpha = np.arccos(1 - 2*Hr/D)
#
# T = 1 / (omega / 60)  # Период одного оборота шпинделя
# oomega = 2 * np.pi / T  # Угловая скорость рад/с
#
# # Высота заготовки(для примера) [мм]
# b_workpiece = 50.0
#
# # Скорость подачи стола [мм/c]. В моем примере 12.8 мм/c
# V_f = f * omega * n / (60 * (10**3))
# print("V_f = ", V_f)
#
# # Начальные координаты фрезы (ноль СК - это левый нижний угол заготовки)
# tool_start_y = b_workpiece - Hr + D/2
# tool_start_x = - D/2 * np.sin(alpha)
#
# num_points = int(b / dx)  # Кол-во отрезков разбиения
# x_values = np.arange(num_points) * dx
# buffer_list = np.column_stack((x_values, np.full(num_points, b_workpiece)))
#
# # Часть с зацикливанием, можем взять любое dt и t и тем самым получим процесс фрезерования
# step = 0.000125 / 4  # Шаг по времени (т.к. мы взяли 4800 об/мин -> 1/400 оборота время step)
# dt = 0.4  # Через сколько остановить выполнение программы(здесь это время полного оборота)
# t = 1  # Начальный момент времени(потом будет изменяться с каждым шагом)
# t_cutting = t  # Время, отвечающее за запаздывание, в дальнейшем t и t_cutting будут отличаться на lag*step
# lag_score = 0  # Счетчик, отвечающий за то, на сколько шагов step отличаются t и t_cutting
# thickness_list = {}
#
# # temporary_buffer = []  # Временный буфер для реализации запаздывания, сюда записываем координаты режущих кромок
#
#
# def checking_the_cutting_tool(t, dt, i, z_buffer_cell):
#     global buffer_list
#
#     # Получаем массивы длины n из списка множеств вида (x, y)
#     tooth_coordinate = tooth_coord(t)  # Положение зубьев фрезы в момент времени t
#     tooth_coordinate_new = tooth_coord(t+dt)  # Положение зубьев фрезы через время dt
#
#     list_xiyi = []  # Структура будет содержать списки из 2х значений: (xi, yi)
#     list_xiyi_new = []  # Структура будет содержать списки из 2х значений: (xi_new, yi_new)
#     # Случай, когда и в момент времени t и в момент времени t+dt фреза внутри заготовки
#     if (0 <= tooth_coordinate[i][0] <= b and 0 <= tooth_coordinate[i][1] <= b_workpiece) and \
#             (0 <= tooth_coordinate_new[i][0] <= b and 0 <= tooth_coordinate_new[i][1] <= b_workpiece):
#         list_xiyi.append((tooth_coordinate[i][0], tooth_coordinate[i][1]))
#         list_xiyi_new.append((tooth_coordinate_new[i][0], tooth_coordinate_new[i][1]))
#     # Случай, когда в момент времени t фреза внутри заготовки, а в момент времени t+dt - нет
#     elif (0 <= tooth_coordinate[i][0] <= b and 0 <= tooth_coordinate[i][1] <= b_workpiece) and not \
#             (0 <= tooth_coordinate_new[i][0] <= b and 0 <= tooth_coordinate_new[i][1] <= b_workpiece):
#         # Эту часть нужно модифицировать (создать общий буфер, в котором будут изменяться координаты)
#         # Из которого можно будет взять координаты xi_new_s, yi_new_s (с прошлых этапов фрезерования)
#         xi_new_s = tooth_coordinate_new[i][0]
#         yi_new_s = tooth_coordinate_new[i][1]
#         if tooth_coordinate_new[i][0] > b:
#             xi_new_s = buffer_list[z_buffer_cell][0]
#         if tooth_coordinate_new[i][1] > b_workpiece:
#             yi_new_s = b_workpiece
#         list_xiyi.append((tooth_coordinate[i][0], tooth_coordinate[i][1]))
#         list_xiyi_new.append((xi_new_s, yi_new_s))
#     elif not (0 <= tooth_coordinate[i][0] <= b and 0 <= tooth_coordinate[i][1] <= b_workpiece) and \
#             (0 <= tooth_coordinate_new[i][0] <= b and 0 <= tooth_coordinate_new[i][1] <= b_workpiece):
#         # Эту часть нужно модифицировать (создать общий буфер, в котором будут изменяться координаты)
#         # Из которого можно будет взять координаты xi_s, yi_s (с прошлых этапов фрезерования)
#         xi_s = tooth_coordinate[i][0]
#         yi_s = tooth_coordinate[i][1]
#         if tooth_coordinate[i][0] > b:
#             xi_s = b
#         if tooth_coordinate[i][1] > b_workpiece:
#             yi_s = b_workpiece
#         list_xiyi.append((xi_s, yi_s))
#         list_xiyi_new.append((tooth_coordinate_new[i][0], tooth_coordinate_new[i][1]))
#     else:
#         list_xiyi.append((tooth_coordinate[i][0], b_workpiece))
#         list_xiyi_new.append((tooth_coordinate_new[i][0], b_workpiece))
#     xi = list_xiyi[0][0]
#     yi = list_xiyi[0][1]
#     xi_new = list_xiyi_new[0][0]
#     yi_new = list_xiyi_new[0][1]
#
#     return xi, yi, xi_new, yi_new
#
#
# def tooth_coord(t):
#     """Функция, которая на вход получает время t и возвращает список координат вершин зубьев фрезы через время t"""
#     angle_step = 2 * np.pi / n
#     T = 1/(omega/60)  # Период одного оборота шпинделя
#     oomega = 2*np.pi / T  # Угловая скорость рад/с
#     coord = []
#     tool_x = tool_start_x + V_f*t  # Координата x центра фрезы через время t
#     tool_y = tool_start_y  # Координата y центра фрезы через время t
#     for i in range(n):
#         angle = i * angle_step + oomega * t
#         x = tool_x + D / 2 * np.cos(angle)
#         y = tool_y + D / 2 * np.sin(angle)
#         coord.append((x, y))
#     return coord
#
#
# def cutting(buffer, t, step):
#     """Функция, которая отвечает за резание заготовки.
#         На вход получает buffer - z-буфер
#                          t - начальный момент времени
#                          step - промежуток времени, который пройдет фреза за данный этап резания"""
#
# def plot_figures(points1, points2):
#     # Извлекаем координаты x и y из списков точек
#     x1, y1 = zip(*points1)
#     X2, Y2 = zip(*points2)
#     x3, y3 = zip(*tooth_coord(1))
#     x4, y4 = zip(*tooth_coord(1.0125))
#
#     # Строим первую фигуру - набор точек
#     plt.scatter(x1, y1, label='Фреза t=2.25')
#     plt.scatter(x3, y3, label='Фреза t=1')
#     # plt.scatter(x4, y4, label='Фреза t=0.9')
#
#     # Строим вторую фигуру - соединенные точки
#     plt.plot(X2, Y2, label='Заготовка', linestyle='dashed')
#
#     # Добавляем подписи и легенду
#     plt.title('Моделирование фрезерования')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.legend()
#
#     # Делаем одинаковый масштаб
#     plt.axis('equal')
#
#     plt.xlim(0, 40)
#     # Показываем график
#     plt.show()
#
# def plot_thickness(step, thickness_list, number=-1):
#     ''' Функция строит график толщины срезаемого слоя от времени
#     На вход она принимает: step - шаг разбиения по времени
#                            thickness_list - словарь со списком толщин для каждой режущей кромки
#                            number - номер режущей кромки, для которой нужно вывести толщины,
#                            если number = -1 - выводим для всех кромок (отсчет идет от 0 кромки)'''
#     # Создание списка значений x и y для построения графика
#     if number != -1:
#             x = [i * step for i in range(1, len(thickness_list[number]) + 1)]
#             y = thickness_list[number]
#     else:
#         x = [i * step for i in range(1, len(thickness_list[0]) + 1)]
#         y = []
#         max_length = max(len(lst) for lst in thickness_list.values())
#
#         # Идем по индексам от 0 до max_length - 1
#         for i in range(max_length):
#             found = False
#             for key in thickness_list:
#                 if len(thickness_list[key]) > i and thickness_list[key][i] != 0:
#                     y.append(thickness_list[key][i])
#                     found = True
#                     break
#             if not found:
#                 y.append(0)
#
#     # Построение точечного графика
#     plt.scatter(x, y, label='Точки', color='blue')
#
#     # Соединение точек прямыми линиями
#     plt.plot(x, y, label='Прямые', color='red', linestyle='solid')
#
#     # Добавление заголовка и меток осей
#     plt.title('График толщины срезаемого слоя в зависимости от времени t')
#     plt.xlabel('t')
#     plt.ylabel('h')
#
#     # Добавление легенды
#     plt.legend()
#     plt.xlim(0.31, 0.33)
#
#     # Отображение графика
#     plt.show()
#
# while True:
#     lag_score += 1
#     if lag_score == 100:
#         cutting(buffer_list, t_cutting, step)
#         t_cutting += step
#         lag_score -= 1
#     find_thickness(t)

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from ipywidgets import interact
# from matplotlib.widgets import Slider


def plot_figures(points1, points2):
    # Извлекаем координаты x и y из списков точек
    x1, y1 = zip(*points1)
    X2, Y2 = zip(*points2)
    x3, y3 = zip(*tooth_coord(1))
    x4, y4 = zip(*tooth_coord(1.0125))

    # Строим первую фигуру - набор точек
    plt.scatter(x1, y1, label='Фреза t=2.25')
    plt.scatter(x3, y3, label='Фреза t=1')
    # plt.scatter(x4, y4, label='Фреза t=0.9')

    # Строим вторую фигуру - соединенные точки
    plt.plot(X2, Y2, label='Заготовка', linestyle='dashed')

    # Добавляем подписи и легенду
    plt.title('Моделирование фрезерования')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    # Делаем одинаковый масштаб
    plt.axis('equal')

    plt.xlim(0, 40)
    # Показываем график
    plt.show()


def part_of_cutting(buffer, t, dt, count_len_thikness_list=0):
    """Функция, которая отвечает за резание заготовки.
        На вход получает buffer - z-буфер
                         t - начальный момент времени
                         dt - промежуток времени, который пройдет фреза за данный этап резания"""

    # global z_buffer_history  # Объявляем глобальную переменную для хранения истории z-буфера

    # Получаем массивы длины n из списка множеств вида (x, y)
    tooth_coordinate = tooth_coord(t)  # Положение зубьев фрезы в момент времени t
    tooth_coordinate_new = tooth_coord(t+dt)  # Положение зубьев фрезы через время dt

    list_xiyi = []  # Структура будет содержать списки из 2х значений: (xi, yi)
    list_xiyi_new = []  # Структура будет содержать списки из 2х значений: (xi_new, yi_new)
    for i in range(n):
        # Случай, когда и в момент времени t и в момент времени t+dt фреза внутри заготовки
        if (0 <= tooth_coordinate[i][0] <= b and 0 <= tooth_coordinate[i][1] <= b_workpiece) and \
                (0 <= tooth_coordinate_new[i][0] <= b and 0 <= tooth_coordinate_new[i][1] <= b_workpiece):
            list_xiyi.append((tooth_coordinate[i][0], tooth_coordinate[i][1]))
            list_xiyi_new.append((tooth_coordinate_new[i][0], tooth_coordinate_new[i][1]))
        # Случай, когда в момент времени t фреза внутри заготовки, а в момент времени t+dt - нет
        elif (0 <= tooth_coordinate[i][0] <= b and 0 <= tooth_coordinate[i][1] <= b_workpiece) and not \
                (0 <= tooth_coordinate_new[i][0] <= b and 0 <= tooth_coordinate_new[i][1] <= b_workpiece):
            # Эту часть нужно модифицировать (создать общий буфер, в котором будут изменяться координаты)
            # Из которого можно будет взять координаты xi_new_s, yi_new_s (с прошлых этапов фрезерования)
            xi_new_s = tooth_coordinate_new[i][0]
            yi_new_s = tooth_coordinate_new[i][1]
            if tooth_coordinate_new[i][0] > b:
                xi_new_s = b
            if tooth_coordinate_new[i][1] > b_workpiece:
                yi_new_s = b_workpiece
            list_xiyi.append((tooth_coordinate[i][0], tooth_coordinate[i][1]))
            list_xiyi_new.append((xi_new_s, yi_new_s))
        elif not (0 <= tooth_coordinate[i][0] <= b and 0 <= tooth_coordinate[i][1] <= b_workpiece) and \
                (0 <= tooth_coordinate_new[i][0] <= b and 0 <= tooth_coordinate_new[i][1] <= b_workpiece):
            # Эту часть нужно модифицировать (создать общий буфер, в котором будут изменяться координаты)
            # Из которого можно будет взять координаты xi_s, yi_s (с прошлых этапов фрезерования)
            xi_s = tooth_coordinate[i][0]
            yi_s = tooth_coordinate[i][1]
            if tooth_coordinate[i][0] > b:
                xi_s = b
            if tooth_coordinate[i][1] > b_workpiece:
                yi_s = b_workpiece
            list_xiyi.append((xi_s, yi_s))
            list_xiyi_new.append((tooth_coordinate_new[i][0], tooth_coordinate_new[i][1]))
        else:
            list_xiyi.append((tooth_coordinate[i][0], b_workpiece))
            list_xiyi_new.append((tooth_coordinate_new[i][0], b_workpiece))
        xi = list_xiyi[i][0]
        yi = list_xiyi[i][1]
        xi_new = list_xiyi_new[i][0]
        yi_new = list_xiyi_new[i][1]
        if yi == yi_new == b_workpiece:
            continue
        # Ищем уравнение прямой, проходящей через эти точки (y = mx + b)
        try:
            m = (yi_new - yi)/(xi_new - xi)
            b1 = yi - m*xi
        except RuntimeWarning:
            print('RuntimeWarning')


        for j, (x, y) in enumerate(buffer):
            if buffer[j-1][0] <= xi_new <= buffer[j][0] and yi_new < y < b_workpiece:
                # Определение коэф-ов для режущей кромки
                x_tooth = tooth_coord(t+dt)[i][0]
                y_tooth = tooth_coord(t + dt)[i][1]
                tool_x = tool_start_x + V_f * (t+dt)  # Координата x центра фрезы через время t
                tool_y = tool_start_y  # Координата y центра фрезы через время t
                m_tooth = (y_tooth - tool_y) / (x_tooth - tool_x)
                b_tooth = tool_y - m_tooth * tool_x
                # Определение коэф-ов для z-буффера
                m_buffer = (buffer[j][1] - buffer[j-1][1]) / (buffer[j][0] - buffer[j-1][0])
                b_buffer = buffer[j-1][1] - m_buffer * buffer[j-1][0]

                # # # Используйте историю z-буфера для получения предыдущих значений
                # m_buffer = 0
                # b_buffer = 0
                #
                # if len(temporary_buffer) == 200:
                #     m_buffer = temporary_buffer
                # if z_buffer_history[0] is not None:  # Проверка на наличие предыдущих значений в истории
                #     m_buffer = (z_buffer_history[0][j][1] - z_buffer_history[199][j][1]) / \
                #                (z_buffer_history[0][j][0] - z_buffer_history[199][j][0])
                #     b_buffer = z_buffer_history[199][j][1] - m_buffer * z_buffer_history[199][j][0]

                def find_intersection_point(m1, b1, m2, b2):
                    if m1 == m2:
                        # Прямые параллельны, нет точки пересечения
                        return None
                    else:
                        x = (b2 - b1) / (m1 - m2)
                        y = m1 * x + b1
                        return x, y

                def distance_between_points(point1, point2):
                    x1, y1 = point1
                    x2, y2 = point2
                    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    # print(distance)
                    return distance

                intersection_point = find_intersection_point(m_tooth, b_tooth, m_buffer, b_buffer)
                point1 = intersection_point
                point2 = (x_tooth, y_tooth)

                if intersection_point is not None:
                    distance = distance_between_points(point1, point2)
                    thickness_list[i][count_len_thikness_list] = distance
                else:
                    print("Прямые параллельны, невозможно найти расстояние")
            # Изменяем координаты y точек буфера, лежащих между xi и xi_new
            if xi <= x <= xi_new and y > m*x + b1:
                # Во временный буфер добавляем еще и номер j, чтобы отслеживать индекс буфера для которого изменяем x,y
                # temporary_buffer.append((j, (x, m*x + b1)))
                buffer[j] = (x, m*x + b1)
                # if len(temporary_buffer) == 7:
                #     buffer[temporary_buffer[0][0]] = temporary_buffer[0][1]
                #     temporary_buffer.pop(0)
                # print(temporary_buffer)
    # Проходимся по остатку временного буфера, чтобы окончательно заполнить buffer
    # for i in range(len(temporary_buffer)):
    #     buffer[temporary_buffer[i][0]] = temporary_buffer[i][1]
    return buffer


def tooth_coord(t):
    """Функция, которая на вход получает время t и возвращает список координат вершин зубьев фрезы через время t"""
    angle_step = 2 * np.pi / n
    coord = []
    tool_x = tool_start_x + V_f*t  # Координата x центра фрезы через время t
    tool_y = tool_start_y  # Координата y центра фрезы через время t
    for i in range(n):
        angle = i * angle_step + omega*t/60  # alpha + np.pi * 3 / 2  Это убрал
        x = tool_x + D / 2 * np.cos(angle)
        y = tool_y + D / 2 * np.sin(angle)
        coord.append((x, y))
    return coord


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
                    y.append(thickness_list[key][i])
                    found = True
                    break
            if not found:
                y.append(0)

    # Построение точечного графика
    plt.scatter(x, y, label='Точки', color='blue')

    # Соединение точек прямыми линиями
    plt.plot(x, y, label='Прямые', color='red', linestyle='solid')

    # Добавление заголовка и меток осей
    plt.title('График толщины срезаемого слоя в зависимости от времени t')
    plt.xlabel('t')
    plt.ylabel('h')

    # Добавление легенды
    plt.legend()
    # plt.xlim(0.31, 0.33)

    # Отображение графика
    plt.show()


# Определяем параметры фрезы и резания
n = 4  # Кол-во зубьев фрезы [безразм]
D = 6.0  # Диаметр фрезы [мм]
a = 4.0  # Ширина резания [мм]
Hr = 3.0  # Радиальная глубина резания [мм]
omega = 4800  # Скорость вращения шпинделя [об/мин]
f = 40.0  # Подача на зуб [мкм]
b = 150  # Длина заготовки [мм]
# dt = 10  # Время, которое прошло с момента запуска фрезы [с]
dx = 0.2  # Разбиение детали dx [мм]

# t0 = 0  # Начало испытаний(время для слайдера)

#  Угол вступления в первый контакт фрезы и заготовки
alpha = np.arccos(1 - 2*Hr/D)

# Высота заготовки(для примера) [мм]
b_workpiece = 50.0

# Скорость подачи стола [мм/c]. В моем примере 12.8 мм/c
V_f = f * omega * n / (60 * (10**3))
print("V_f = ", V_f)

# Начальные координаты фрезы (ноль СК - это левый нижний угол заготовки)
tool_start_y = b_workpiece - Hr + D/2
tool_start_x = - D/2 * np.sin(alpha)

num_points = int(b / dx)
x_values = np.arange(num_points) * dx
buffer_list = np.column_stack((x_values, np.full(num_points, b_workpiece)))

# thickness_list = [0] * len(buffer_list)

temporary_buffer = []  # Временный буфер для реализации запаздывания


# Часть с зацикливанием, можем взять любое dt и t и тем самым получим процесс фрезерования
step = 0.000125  # Шаг по времени (т.к. мы взяли 4800 об/мин -> 1/100 оборота время step)
dt = 0.4  # Через сколько остановить выполнение программы(здесь это время полного оборота)
t = 1  # Начальный момент времени(потом будет изменяться с каждым шагом)
buffer_list_test = buffer_list
thickness_list = {}

for i in range(n):
    thickness_list[i] = [0] * int(dt / step + 1)
print(len(thickness_list))

finish = t + dt  # Конечный момент времени
count_len_thikness_list = 0  # Ведем подсчет какой сейчас номер шага dt / step
# Т.к. у нас время дискретно => мы разбиваем t + dt на шаги по времени
# и будем для каждого такого шага искать толщину срезаемого слоя

while True:
    if t >= finish:
        plot_figures(tooth_coord(t), buffer_list)
        break
    else:
        part_of_cutting(buffer_list, t, step, count_len_thikness_list)
        t = t + step
        count_len_thikness_list += 1
        # print(t)
# print(thickness_list)
# print(len(thickness_list))
# print(''.join(str(thickness_list)))
plot_thickness(step, thickness_list, -1)
# thickness_of_the_cut_layer(part_of_cutting(buffer_list, t, step), 2.25)

