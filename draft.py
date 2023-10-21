import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Функция для рисования отрезка между двумя точками в 2D
def draw_line(x0, y0, x1, y1):
    points = []
    is_steep = abs(y1 - y0) > abs(x1 - x0)
    if is_steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    swapped = False
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        swapped = True
    dx = x1 - x0
    dy = abs(y1 - y0)
    error = dx / 2
    y = y0
    ystep = None
    if y0 < y1:
        ystep = 1
    else:
        ystep = -1
    for x in range(int(x0), int(x1) + 1):
        if is_steep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= dy
        if error < 0:
            y += ystep
            error += dx
    if swapped:
        points.reverse()
    return points

# Функция для рисования фрезы
def draw_mill(n, D, a):
    fig, ax = plt.subplots()

    # Рисуем фрезу как окружность
    center = (0, 0)
    mill = Circle(center, D / 2, fill=True, color='blue')
    ax.add_patch(mill)

    # Рисуем название фрезы
    ax.text(center[0], center[1], f'n={n}\nD={D} mm\na={a} mm', ha='center', va='center', color='black', fontsize=10)

    ax.set_aspect('equal')
    ax.set_xlim(-D / 2, D / 2)
    ax.set_ylim(-D / 2, D / 2)

# Функция для рисования параллелепипеда и линейной аппроксимации между точками разбиения
def draw_cuboid(width, height, length, dx, t):
    fig, ax = plt.subplots()

    # Размеры параллелепипеда
    w = width
    h = height
    l = length

    # Переворачиваем координаты вершин параллелепипеда для изменения направления осей
    vertices = [
        [0, 0, 0],
        [0, l, 0],
        [w, l, 0],
        [w, 0, 0],
        [0, 0, h],
        [0, l, h],
        [w, l, h],
        [w, 0, h]
    ]

    # Создаем список ребер
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    # Вычисляем, какая часть заготовки была выфрезерована к моменту времени t
    material_removed = (omega * np.pi * D * t) * (dx / f)

    # Рисуем ребра параллелепипеда
    for edge in edges:
        x0, y0, z0 = vertices[edge[0]]
        x1, y1, z1 = vertices[edge[1]]
        if x0 < material_removed and x1 < material_removed:
            continue
        x0 = max(x0, material_removed)
        x1 = max(x1, material_removed)
        points = draw_line(x0, z0, x1, z1)
        for point in points:
            ax.plot([x0, point[0]], [z0, point[1]], 'ko-', markersize=2)

    # Рисуем заготовку полностью
    for edge in edges:
        x0, y0, z0 = vertices[edge[0]]
        x1, y1, z1 = vertices[edge[1]]
        points = draw_line(x0, z0, x1, z1)
        for point in points:
            ax.plot([x0, point[0]], [z0, point[1]], 'ro-', markersize=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')

    plt.axis('equal')

# Задаем параметры фрезы и резания
n = int(input('Кол-во зубьев фрезы: '))
D = float(input('Диаметр фрезы[мм]: '))
a = float(input('Ширина резания[мм]: '))
Hr = float(input('Радиальная глубина резания[мм]: '))
omega = float(input('Скорость вращения шпинделя[об/мин]: '))
f = float(input('Подача[мкм]: '))
b = float(input('Длина заготовки[мм]: '))
dt = float(input('Время, прошедшее с запуска фрезы[с]: '))
dx = float(input('Разбиение детали на отрезки dx[мм]: '))

# Рисуем фрезу
draw_mill(n, D, a)

# Рисуем параллелепипед с учетом фрезерования
draw_cuboid(b, Hr, D, dx, dt)

plt.show()



# def define_a_quarter(omega, t, angle=0):
#     k = omega * t // (2 * np.pi)  # Сколько оборотов уже совершено
#     angle_new = angle - 2*np.pi*k  # Угол поворота [0,2pi]
#     if 0 <= angle_new <= np.pi/2:
#         quarter = 1
#     elif np.pi/2 <= angle_new <= np.pi:
#         quarter = 4
#     elif np.pi <= angle_new <= 3*np.pi/2:
#         quarter = 3
#     else:
#         quarter = 2
#
#     return quarter

# Функция для вычисления координат вершин фрезы в зависимости от времени t
# alpha - изначальный поворот фрезы относительно вертикали
# def calculate_tool_path(t, alpha=0):
    # Расчет угла поворота фрезы в зависимости от времени t
    # angular_velocity = 2 * np.pi * n  # Угловая скорость вращения фрезы (полный оборот в секунду)
    # angle = angular_velocity * t

    # Начальные параметры
    # angle = omega * t + alpha  # Текущий угол
    # tool_x = tool_start_x + V_f * t
    # tool_y = tool_start_y
    # Расчет координат вершин зубьев фрезы
    # tool_vertices = []
    # for i in range(n):
    #     tooth_angle = angle + 2 * np.pi * i / n
        # x_loc = D / 2 * np.cos(tooth_angle)  # Локальный x относительно центра фрезы
        # y_loc = D / 2 * np.sin(tooth_angle)  # Локальный y относительно центра фрезы
        # tool_vertices.append((x_loc, y_loc))  # Список координат вершин зубьев относительно центра фрезы
        # В глобальной СК, где центр - это нижняя левая часть заготовки
        # quarter = define_a_quarter(omega, t, tooth_angle)  # Возвращает список, где [0] - четверть, а [1:2] - коэф-ты прибавления/вычитания X, Y
        # if quarter[0] == 1:
        #     x_tooth = tool_x + np.sin(tooth_angle) * D/2
        #     y_tooth = tool_y + np.cos(tooth_angle) * D/2
        # elif quarter[0] == 2:
        #     x_tooth = tool_x - np.sin(tooth_angle) * D / 2
        #     y_tooth = tool_y + np.cos(tooth_angle) * D / 2
        # elif quarter[0] == 3:
        #     x_tooth = tool_x + np.sin(tooth_angle) * D / 2
        #     y_tooth = tool_y + np.cos(tooth_angle) * D / 2
        # elif quarter[0] == 4:
        #     x_tooth = tool_x + np.sin(tooth_angle) * D / 2
        #     y_tooth = tool_y + np.cos(tooth_angle) * D / 2

    # return tool_vertices



# def calculate_disk_coordinates(t, omega, V, points):
#     x0, y0 = tool_start_x, tool_start_y  # Начальные координаты диска (центр)
#     x_result = []
#     y_result = []
#
#     for point in points:
#         # Вычисляем угол, на который вращается диск за время t
#         angle_after_t = omega * t - (np.pi - alpha)
#
#         # Вычисляем координаты точки на диске, учитывая вращение
#         x_point = x0 + point[0] * np.cos(angle) - point[1] * np.sin(angle)
#         y_point = y0 + point[0] * np.sin(angle) + point[1] * np.cos(angle)
#
#         # Учитываем движение диска
#         x_point += V * t
#         x_result.append(x_point)
#         y_result.append(y_point)
#
#     return x_result, y_result



# Функция для отображения текущего состояния резания
# def plot_tool_path(t, alpha=0):
#     plt.figure(figsize=(8, 6))
#
#     # Отобразите заготовку (прямоугольник в примере)
#     plt.plot([0, b, b, 0, 0], [0, 0, b_workpiece, b_workpiece, 0], 'b-', label='Workpiece')
#
#     # Получите координаты вершин фрезы
#     tool_vertices = calculate_tool_path(t, alpha)
#
#     # Отобразите фрезу (зубья фрезы)
#     tool_x, tool_y = zip(*tool_vertices)
#     plt.plot(tool_x, tool_y, 'r-', label='Cutter')
#
#     plt.axis('equal')
#     plt.title(f"Time: {t} s")
#     plt.legend()
#     plt.show()


# n = int(input('Кол-во зубьев фрезы: '))  # Кол-во зубьев фрезы [безразм]
# D = float(input('Диаметр фрезы[мм]: '))  # Диаметр фрезы [мм]
# a = float(input('Ширина резания[мм]: '))  # Ширина резания [мм]
# Hr = float(input('Радиальная глубина резания[мм]: '))  # Радиальная глубина резания [мм]
# omega = float(input('Скорость вращения шпинделя[об/мин]: '))  # Скорость вращения шпинделя [об/мин]
# f = float(input('Подача[мкм]: '))  # Подача [мкм]
# b = float(input('Длина заготовки[мм]: '))  # Длина заготовки
# dt = float(input('Время, прошедшее с запуска фрезы[с]: '))  # Время, которое прошло с момента запуска фрезы
# dx = float(input('Разбиение детали на отрезки dx[мм]: '))  # Разбиение детали dx
