import numpy as np
import matplotlib.pyplot as plt

# Определите параметры фрезы и резания
# n = int(input('Кол-во зубьев фрезы: '))  # Кол-во зубьев фрезы [безразм]
# D = float(input('Диаметр фрезы[мм]: '))  # Диаметр фрезы [мм]
# a = float(input('Ширина резания[мм]: '))  # Ширина резания [мм]
# Hr = float(input('Радиальная глубина резания[мм]: '))  # Радиальная глубина резания [мм]
# omega = float(input('Скорость вращения шпинделя[об/мин]: '))  # Скорость вращения шпинделя [об/мин]
# f = float(input('Подача[мкм]: '))  # Подача [мкм]
# b = float(input('Длина заготовки[мм]: '))  # Длина заготовки
# dt = float(input('Время, прошедшее с запуска фрезы[с]: '))  # Время, которое прошло с момента запуска фрезы
# dx = float(input('Разбиение детали на отрезки dx[мм]: '))  # Разбиение детали dx
n = 2  # Кол-во зубьев фрезы [безразм]
D = 6.0  # Диаметр фрезы [мм]
a = 4.0  # Ширина резания [мм]
Hr = 1.0  # Радиальная глубина резания [мм]
omega = 3.8  # Скорость вращения шпинделя [об/мин]
f = 40.0  # Подача на зуб [мкм]
b = 150  # Длина заготовки [мм]
# dt =   # Время, которое прошло с момента запуска фрезы
dx = 2.0  # Разбиение детали dx [мкм]


#  Угол вступления в первый контакт фрезы и заготовки
alpha = np.arccos(1 - 2*Hr/D)

# Высота заготовки(для примера) [мм]
b_workpiece = 50.0

# Скорость подачи стола [м/c]
V_f = f * omega * n / (60 * 10**6)

# # Определим скорость резания [м/с]
# Vx = np.pi * D * omega / (60 * 1000)
# andgle_between_teeth = 2 * np.pi / n

# Начальные координаты фрезы (ноль СК - это левый нижний угол заготовки)
tool_start_x = b_workpiece - Hr + D/2
tool_start_y = - D/2 * np.sin(alpha)


def define_a_quarter(omega, t, angle=0):
    k = omega * t // (2 * np.pi)  # Сколько оборотов уже совершено
    angle_new = angle - 2*np.pi*k  # Угол поворота [0,2pi]
    if 0 <= angle_new <= np.pi/2:
        quarter = 1
    elif np.pi/2 <= angle_new <= np.pi:
        quarter = 4
    elif np.pi <= angle_new <= 3*np.pi/2:
        quarter = 3
    else:
        quarter = 2

    return quarter


# Функция для вычисления координат вершин фрезы в зависимости от времени t
# alpha - изначальный поворот фрезы относительно вертикали
def calculate_tool_path(t, alpha=0):
    # Расчет угла поворота фрезы в зависимости от времени t
    # angular_velocity = 2 * np.pi * n  # Угловая скорость вращения фрезы (полный оборот в секунду)
    # angle = angular_velocity * t

    # Начальные параметры
    angle = omega * t + alpha  # Текущий угол
    tool_x = tool_start_x + V_f * t
    tool_y = tool_start_y
    # Расчет координат вершин зубьев фрезы
    tool_vertices = []
    for i in range(n):
        print(i)
        tooth_angle = angle + 2 * np.pi * i / n
        # x_loc = D / 2 * np.cos(tooth_angle)  # Локальный x относительно центра фрезы
        # y_loc = D / 2 * np.sin(tooth_angle)  # Локальный y относительно центра фрезы
        # tool_vertices.append((x_loc, y_loc))  # Список координат вершин зубьев относительно центра фрезы
        # В глобальной СК, где центр - это нижняя левая часть заготовки
        quarter = define_a_quarter(omega, t, tooth_angle)  # Возвращает список, где [0] - четверть, а [1:2] - коэф-ты прибавления/вычитания X, Y
        if quarter[0] == 1:
            x_tooth = tool_x + np.sin(tooth_angle) * D/2
            y_tooth = tool_y + np.cos(tooth_angle) * D/2
        elif quarter[0] == 2:
            x_tooth = tool_x - np.sin(tooth_angle) * D / 2
            y_tooth = tool_y + np.cos(tooth_angle) * D / 2
        elif quarter[0] == 3:
            x_tooth = tool_x + np.sin(tooth_angle) * D / 2
            y_tooth = tool_y + np.cos(tooth_angle) * D / 2
        elif quarter[0] == 4:
            x_tooth = tool_x + np.sin(tooth_angle) * D / 2
            y_tooth = tool_y + np.cos(tooth_angle) * D / 2

    return tool_vertices



# Функция для отображения текущего состояния резания
def plot_tool_path(t, alpha=0):
    plt.figure(figsize=(8, 6))

    # Отобразите заготовку (прямоугольник в примере)
    plt.plot([0, b, b, 0, 0], [0, 0, b_workpiece, b_workpiece, 0], 'b-', label='Workpiece')

    # Получите координаты вершин фрезы
    tool_vertices = calculate_tool_path(t, alpha)

    # Отобразите фрезу (зубья фрезы)
    tool_x, tool_y = zip(*tool_vertices)
    plt.plot(tool_x, tool_y, 'r-', label='Cutter')

    plt.axis('equal')
    plt.title(f"Time: {t} s")
    plt.legend()
    plt.show()


# Пример использования:
plot_tool_path(0.0, np.pi - alpha)  # Отобразить состояние резания в начальный момент времени
plot_tool_path(1.0, np.pi - alpha)  # Отобразить состояние резания в начальный момент времени
plot_tool_path(10.0, np.pi - alpha)  # Отобразить состояние резания в начальный момент времени
plot_tool_path(100.0, np.pi - alpha)  # Отобразить состояние резания в начальный момент времени
