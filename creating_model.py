import numpy as np
import matplotlib.pyplot as plt

# Определите параметры фрезы и резания
n = int(input('Кол-во зубьев фрезы: '))  # Кол-во зубьев фрезы [безразм]
D = float(input('Диаметр фрезы[мм]: '))  # Диаметр фрезы [мм]
a = float(input('Ширина резания[мм]: '))  # Ширина резания [мм]
Hr = float(input('Радиальная глубина резания[мм]: '))  # Радиальная глубина резания [мм]
omega = float(input('Скорость вращения шпинделя[об/мин]: '))  # Скорость вращения шпинделя [об/мин]
f = float(input('Подача[мкм]: '))  # Подача [мкм]
b = float(input('Длина заготовки[мм]: '))  # Длина заготовки
dt = float(input('Время, прошедшее с запуска фрезы[с]: '))  # Время, которое прошло с момента запуска фрезы
dx = float(input('Разбиение детали на отрезки dx[мм]: '))  # Разбиение детали dx

# Определим скорость резания [м/с]
Vx = np.pi * D * omega / (60 * 1000)
# andgle_between_teeth = 2 * np.pi / n

# Начальные координаты фрезы (ноль СК - это левый нижний угол заготовки)
tool_start_x = - D * np.sqrt(2) / 2
tool_start_y = a + D * np.sqrt(2) / 2


# Функция для вычисления координат вершин фрезы в зависимости от времени t
def calculate_tool_path(t):
    # Расчет угла поворота фрезы в зависимости от времени t
    # angular_velocity = 2 * np.pi * n  # Угловая скорость вращения фрезы (полный оборот в секунду)
    # angle = angular_velocity * t
    angle = omega * t  # Текущий угол(для t = 0 верхня точка фрезы, так проще считать)

    # Угол поворота (первого зуба фрезы) в зависимости от времени t
    print(n)
    # Расчет координат вершин зубьев фрезы
    tool_vertices = []
    for i in range(n):
        print(i)
        tooth_angle = angle + 2 * np.pi * i / n
        x_loc = D / 2 * np.cos(tooth_angle)  # Локальный x относительно центра фрезы
        y_loc = D / 2 * np.sin(tooth_angle)  # Локальный y относительно центра фрезы
        tool_vertices.append((x_loc, y_loc))

    return tool_vertices


def tool_coord(t):
    pass


# Функция для отображения текущего состояния резания
def plot_tool_path(t):
    plt.figure(figsize=(8, 6))

    # Отобразите заготовку (прямоугольник в примере)
    plt.plot([0, b, b, 0, 0], [0, 0, a, a, 0], 'b-', label='Workpiece')

    # Получите координаты вершин фрезы
    tool_vertices = calculate_tool_path(t)

    # Отобразите фрезу (зубья фрезы)
    tool_x, tool_y = zip(*tool_vertices)
    plt.plot(tool_x, tool_y, 'r-', label='Cutter')

    plt.axis('equal')
    plt.title(f"Time: {t} s")
    plt.legend()
    plt.show()


# Пример использования:
plot_tool_path(0.0)  # Отобразить состояние резания в начальный момент времени
plot_tool_path(1.0)  # Отобразить состояние резания в начальный момент времени
plot_tool_path(10.0)  # Отобразить состояние резания в начальный момент времени
plot_tool_path(100.0)  # Отобразить состояние резания в начальный момент времени
