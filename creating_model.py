import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ipywidgets import interact
from matplotlib.widgets import Slider


def tooth_coord(t):
    angle_step = 2 * np.pi / n
    coord = []
    tool_x = tool_start_x + V_f*t
    tool_y = tool_start_y
    for i in range(n):
        angle = alpha + np.pi * 3 / 2 + i * angle_step + omega*t
        x = tool_x + D / 2 * np.cos(angle)  # Если добавить tool_start_x, tool_start_y то получим координаты в
        y = tool_y + D / 2 * np.sin(angle)  # глобальной СК, а это координаты относительно центра фрезы
        coord.append((x, y))
    return coord


def draw_points_and_rectangle(points, rectangle):
    # points - список кортежей [(x1, y1), (x2, y2), ...]
    # rectangle - список [x, y, width, height]

    # Создаем новый график
    fig, ax = plt.subplots()

    # Рисуем точки
    for point in points:
        x, y = point
        ax.plot(x, y, 'ro', markersize=3)  # 'ro' означает красные кружки (red circles)

    # Рисуем прямоугольник
    x, y, width, height = rectangle
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)

    # Устанавливаем пределы графика
    ax.set_xlim(-width, 2*width)
    ax.set_ylim(-height, 2*height)

    plt.show()


def update_plot(val):
    global t0
    t0 = slider.val
    updated_coord = tooth_coord(t0)
    draw_points_and_rectangle(updated_coord, rectangle)


# Определяем параметры фрезы и резания
n = 2  # Кол-во зубьев фрезы [безразм]
D = 6.0  # Диаметр фрезы [мм]
a = 4.0  # Ширина резания [мм]
Hr = 1.0  # Радиальная глубина резания [мм]
omega = 3.8  # Скорость вращения шпинделя [об/мин]
f = 40.0  # Подача на зуб [мкм]
b = 150  # Длина заготовки [мм]
# dt =   # Время, которое прошло с момента запуска фрезы
dx = 2.0  # Разбиение детали dx [мкм]

t0 = 0  # Начало испытаний(время для слайдера)

#  Угол вступления в первый контакт фрезы и заготовки
alpha = np.arccos(1 - 2*Hr/D)

# Высота заготовки(для примера) [мм]
b_workpiece = 50.0

# Скорость подачи стола [мм/c]
V_f = f * omega * n / 60

# Начальные координаты фрезы (ноль СК - это левый нижний угол заготовки)
tool_start_y = b_workpiece - Hr + D/2
tool_start_x = - D/2 * np.sin(alpha)

rectangle = [0, 0, b, b_workpiece]

tooth_coord(0)
tooth_coord(1)
tooth_coord(10)
draw_points_and_rectangle(tooth_coord(0), rectangle)
draw_points_and_rectangle(tooth_coord(1), rectangle)
draw_points_and_rectangle(tooth_coord(10), rectangle)
# interact(update_plot, t=(0, 1, 0.1))


# Создаем графическое окно
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Создаем ползунок
slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(slider_ax, 'Время (t)', 0, 1, valinit=t0)
slider.on_changed(update_plot)

# Инициализируем график
update_plot(t0)

plt.show()
