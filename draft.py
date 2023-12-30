import numpy as np
import matplotlib.pyplot as plt


def tooth_coord(t):
    """Функция, которая на вход получает время t и возвращает список координат вершин зубьев фрезы через время t"""
    angle_step = 2 * np.pi / n
    coord = []
    tool_x = tool_start_x + V_f*t  # Координата x центра фрезы через время t
    tool_y = tool_start_y  # Координата y центра фрезы через время t
    for i in range(n):
        angle = i * angle_step + oomega * t
        x = tool_x + D / 2 * np.cos(angle)
        y = tool_y + D / 2 * np.sin(angle)
        coord.append((x, y))
    return coord


def cutting(t, step):
    global buffer
    xy_old = tooth_coord(t)
    xy_new = tooth_coord(t + step)
    for j, (x, y) in enumerate(buffer):
        for i in range(n):
            x_old = xy_old[i][0]
            y_old = xy_old[i][1]
            x_new = xy_new[i][0]
            y_new = xy_new[i][1]

            # Ищем уравнение прямой, проходящей через эти точки (y = mx + b)
            try:
                m = (y_new - y_old) / (x_new - x_old)
                b1 = y_old - m * x_old
            except RuntimeWarning:
                print('RuntimeWarning')

            if x_old <= x <= x_new and m*x + b1 < y < b_workpiece:
                buffer[j] = (x, m*x + b1)


def find_thickness(t, number):
    """number отвечает за номер кромки, для которой находим толщину срезаемого слоя"""
    global buffer

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

    xy_tooth = tooth_coord(t)[number]
    x_tooth = xy_tooth[0]
    y_tooth = xy_tooth[1]
    for j, (x, y) in enumerate(buffer):
        if buffer[j - 1][0] <= x_tooth <= buffer[j][0] and y_tooth < y < b_workpiece:
            # Определение коэф-ов для режущей кромки
            tool_x = tool_start_x + V_f * t  # Координата x центра фрезы через время t
            tool_y = tool_start_y  # Координата y центра фрезы через время t
            m_tooth = (y_tooth - tool_y) / (x_tooth - tool_x)
            b_tooth = tool_y - m_tooth * tool_x
            # Определение коэф-ов для z-буффера
            m_buffer = (buffer[j][1] - buffer[j - 1][1]) / (buffer[j][0] - buffer[j - 1][0])
            b_buffer = buffer[j - 1][1] - m_buffer * buffer[j - 1][0]

            intersection_point = find_intersection_point(m_tooth, b_tooth, m_buffer, b_buffer)

            k = 0
            while True:
                k += 1
                if intersection_point[0] < buffer[j - k][0]:
                    m_buffer = (buffer[j - k][1] - buffer[j - k - 1][1]) / (buffer[j - k][0] - buffer[j - k - 1][0])
                    b_buffer = buffer[j - k - 1][1] - m_buffer * buffer[j - k - 1][0]

                    intersection_point = find_intersection_point(m_tooth, b_tooth, m_buffer, b_buffer)
                    if k == 2:
                        print(k)
                else:
                    break

            point1 = intersection_point
            point2 = (x_tooth, y_tooth)

            if intersection_point is not None:
                distance = distance_between_points(point1, point2)
                thickness_list[number][count_len_thikness_list] = distance
                # print(distance)
            else:
                print("Прямые параллельны, невозможно найти расстояние")


def plot_thickness(step, thickness_list, number=-1):
    '''   Функция строит график толщины срезаемого слоя от времени
    На вход она принимает: step - шаг разбиения по времени
                           thickness_list - словарь со списком толщин для каждой режущей кромки   '''
    # Создание списка значений x и y для построения графика
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
    plt.xlim(0.3125, 0.319)
    plt.ylim(0, 0.09)

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

T = 1 / (omega / 60)  # Период одного оборота шпинделя
oomega = 2 * np.pi / T  # Угловая скорость рад/с

# Высота заготовки(для примера) [мм]
b_workpiece = 50.0

# Скорость подачи стола [мм/c]. В моем примере 12.8 мм/c
V_f = f * omega * n / (60 * (10**3))
print("V_f = ", V_f)

# Начальные координаты фрезы (ноль СК - это левый нижний угол заготовки)
tool_start_y = b_workpiece - Hr + D/2
tool_start_x = - D/2 * np.sin(alpha)

num_points = int(b / dx)  # Кол-во отрезков разбиения
x_values = np.arange(num_points) * dx
buffer = np.column_stack((x_values, np.full(num_points, b_workpiece)))


# Часть с зацикливанием, можем взять любое dt и t и тем самым получим процесс фрезерования
step = 0.000125 / 4  # Шаг по времени (т.к. мы взяли 4800 об/мин -> 1/400 оборота время step)
dt = 0.4  # Через сколько остановить выполнение программы(здесь это время полного оборота)
t = 1  # Начальный момент времени(потом будет изменяться с каждым шагом)
finish = 1.4  # Конечный момент времени

thickness_list = {}

temporary_buffer = []

count_len_thikness_list = 0  # Ведем подсчет какой сейчас номер шага dt / step

while True:
    if t >= finish:
        break
    for i in range(n):
        find_thickness(t, i)
        cutting(t, step)
    t += step
    count_len_thikness_list += 1


plot_thickness(step, thickness_list, -1)