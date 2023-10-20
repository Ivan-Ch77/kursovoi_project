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
