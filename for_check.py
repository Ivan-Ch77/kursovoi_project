import numpy as np
import matplotlib.pyplot as plt


def plot_figures(points1, points2):
    '''Здесь происходит пострение заготовки в момент времени t+dt и фрезы в моменты времени t и t+dt
    points1 - список координат фрезы [[x1, y1] , [x2, y2], ..., [xn, yn]], где n - кол-во зубьев фрезы
    points2 список координат z-буфера вида: [..., [xi, yi],...], его длина равна b_заготовки / dx_буфера'''
    # Извлекаем координаты x и y из списков точек
    x1, y1 = zip(*points1)
    X2, Y2 = zip(*points2)
    x3, y3 = zip(*tooth_coord(0))
    x4, y4 = zip(*tooth_coord(1.0125))

    # Строим первую фигуру - набор точек
    plt.scatter(x1, y1, label='Фреза t=0.4')
    plt.scatter(x3, y3, label='Фреза t=0')
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

    plt.xlim(-5, 40)
    # Показываем график
    plt.show()


def find_thickness(buffer, t):
    '''Функция поиска толщины срезаемого слоя
    buffer - z-буфер поверхности
    t - начальный момент времени. Нужен, чтобы построить прямую y = mx+b
    dt - момент времени для которого определяем толщину срезаемого слоя'''
    # global zds
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
        # print(x1, y1)
        x2, y2 = point2
        # print(x2, y2)
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    tooth_coordinate = tooth_coord(t)

    # tooth_coordinate1 = tooth_coord(t-step)

    for i in range(n):

        xi_new = tooth_coordinate[i][0]
        yi_new = tooth_coordinate[i][1]
        #
        # xi_new_old = tooth_coordinate1[i][0]
        # yi_new_old = tooth_coordinate1[i][1]

        j = int(xi_new // dx)  # int всегда округляет вниз, так что это нам подходит
        xj = buffer[j][0]
        yj = buffer[j][1]
        xj1 = buffer[j+1][0]
        yj1 = buffer[j+1][1]

        # Ищем уравнение прямой, проходящей через эти точки (y = mx + b)
        try:
            mj = (yj1 - yj)/(xj1 - xj)
            bj = yj - mj*xj
        except RuntimeWarning:
            print('RuntimeWarning')

        # Здесь нужно, чтобы yi_new была меньше y = mj*xi_new + bj для j и j+1 координат z-буфера
        if yi_new < mj*xi_new + bj < b_workpiece:
            # Определение коэф-ов для режущей кромки
            x_tooth = xi_new
            y_tooth = yi_new
            tool_x = tool_start_x + V_f * (t)  # Координата x центра фрезы через время t
            tool_y = tool_start_y  # Координата y центра фрезы через время t
            m_tooth = (y_tooth - tool_y) / (x_tooth - tool_x)
            b_tooth = tool_y - m_tooth * tool_x
            # Определение коэф-ов для z-буффера
            m_buffer = (buffer[j+1][1] - buffer[j][1]) / (buffer[j+1][0] - buffer[j][0])
            b_buffer = buffer[j][1] - m_buffer * buffer[j][0]

            if m_buffer < 0:
                continue

            intersection_point = find_intersection_point(m_tooth, b_tooth, m_buffer, b_buffer)

            # Часть кода, котора исключает неточности, связанные с разбиением, т.к. иногда может возникнуть ситуация,
            # когда точка пересечения intersection_point и buffer[j][0] находятся в разных ячейках z-буфера
            # (т.е. точка пересечения лежит не в buffer[j], а в j-1 или еще дальше в зависимости от параметров резания)
            # , тогда нужно искать точку пересечения не с j-ой поверхностью заготовки, а с j - k поверхностью заготовки
            k = int(x_tooth // dx) - int(tool_x // dx)
            while True:
                k -= 1
                if intersection_point[0] < buffer[j - k][0]:
                    m_buffer = (buffer[j - k][1] - buffer[j - k-1][1]) / (buffer[j-k][0] - buffer[j - k-1][0])
                    b_buffer = buffer[j - k-1][1] - m_buffer * buffer[j - k-1][0]

                    intersection_point = find_intersection_point(m_tooth, b_tooth, m_buffer, b_buffer)
                    break
            point1 = intersection_point  # Точка пересечения
            point2 = [x_tooth, y_tooth]  # Координаты режущей кромки
            # point3 = [xi_new_old, yi_new_old]
            point4 = [tool_x, tool_y]  # Координаты центра фрезы

            # X = [point1[0], point2[0], point3[0], point4[0]]
            # Y = [point1[1], point2[1], point3[1], point4[1]]

            if intersection_point is not None:
                distance = distance_between_points(point1, point2)
                # print(distance)

                #  Обрубаем то что больше f
                # if distance * 1000 > f:
                #     distance = f / 1000

                # Записываем для i-ой режущей кромки толщину срезаемого слоя в словарь thickness_list
                thickness_list[i][count_len_thikness_list] = distance
                # plot_points(X, Y, buffer_list)

                if distance * 1000 > 1.3*f:
                    print(f'{distance * 1000} мкм > {f} мкм ----------- ОШИБКА!!!')
                    print('Точка пересечения', point1)
                    print('Координаты режущей кромки', point2)
                    print('Координаты центра фрезы', point4)
                    print(tooth_coordinate)
                    print(np.sqrt((point2[0] - point4[0])**2 + (point2[1] - point4[1]) ** 2))
                    print(i)
                    print(t)
                    if 2.0 < point4[0] < 2.04:
                        pass
                        plot_situation(point1, point2, point4, buffer, i)
                    # zds = True

            else:
                print("Прямые параллельны, невозможно найти расстояние")
            break


def part_of_cutting(buffer, temp_buf0, temp_buf1):
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
        # # Если y не изменяется -> пропускаем шаг
        # if yi == yi_new == b_workpiece:
        #     continue
        # Ищем уравнение прямой, проходящей через эти точки (y = mx + b)
        try:
            m = (yi_new - yi)/(xi_new - xi)
            b1 = yi - m*xi
        except RuntimeWarning:
            print('RuntimeWarning')

        # Находим ячейку, в которой в момент времени t - (lag-1)*step остановилась фреза
        j = int(xi_new // dx)
        # Определяем координаты x,y, которые отвечают координатам начала ячейки
        x = buffer[j][0]
        y = buffer[j][1]
        # Изменяем координаты y точек буфера, лежащих между xi и xi_new
        if xi <= x <= xi_new and y > m*x + b1:
            buffer[j] = (x, m*x + b1)
    return buffer


def tooth_coord(t):
    """Функция, которая на вход получает время t и возвращает список координат вершин зубьев фрезы через время t"""
    angle_step = 2 * np.pi / n
    T = 1/(omega/60)  # Период одного оборота шпинделя [с]
    oomega = 2*np.pi / T  # Угловая скорость рад/с
    coord = []
    tool_x = tool_start_x + V_f*t  # Координата x центра фрезы через время t
    tool_y = tool_start_y  # Координата y центра фрезы через время t
    for i in range(n):
        angle = i * angle_step + oomega * t
        x = tool_x + D / 2 * np.cos(angle)
        y = tool_y + D / 2 * np.sin(angle)
        coord.append([x, y])
    return coord


def plot_situation(crossdot, cuttingpoint, millingcenter, buff, i):

    X2, Y2 = zip(*buff)
    # Строим вторую фигуру - соединенные точки
    plt.plot(X2, Y2, label='Заготовка', linestyle='dashed')

    # Координаты точек
    x_coords = [crossdot[0], cuttingpoint[0], millingcenter[0]]
    y_coords = [crossdot[1], cuttingpoint[1], millingcenter[1]]

    # Цвета и подписи для каждой точки
    colors = ['red', 'blue', 'green']
    labels = ['ТчкПер', 'КРежКрки', 'координаты центра фрезы']

    # Построение точек
    plt.scatter(x_coords, y_coords, c=colors, label=labels)

    # Координаты центра окружности
    circle_center = [millingcenter[0], millingcenter[1]]
    circle_radius = D/2

    # Построение точек
    plt.scatter(x_coords, y_coords, c=colors, label=labels)

    # Добавление подписей к точкам
    for label, x, y in zip(labels, x_coords, y_coords):
        plt.text(x, y, label, fontsize=12, ha='right')

    # Построение окружности
    circle = plt.Circle((circle_center[0], circle_center[1]), circle_radius, color='orange', fill=False)
    plt.gca().add_patch(circle)

    # Настройка графика
    plt.title(f'График точек с подписями и цветами для {i} режущей кромки')
    # plt.xlabel('X-координата')
    # plt.ylabel('Y-координата')
    plt.legend()

    # # Ограничиваем вывод графика, чтобы детальнее его рассмотреть
    plt.xlim(1, 5.5)
    plt.ylim(49, 50.25)

    # Отображение графика
    plt.show()

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

    # Ограничиваем вывод графика, чтобы детальнее его рассмотреть
    plt.xlim(0.28, 0.3)
    plt.ylim(0, 0.1)

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
dx = 0.1  # Разбиение детали dx [мм]

#  Угол вступления в первый контакт фрезы и заготовки
alpha = np.arccos(1 - 2*Hr/D)

T = 1 / (omega / 60)  # Период одного оборота шпинделя
oomega = 2 * np.pi / T  # Угловая скорость рад/с

# Высота заготовки(для примера) [мм]
b_workpiece = 50.0

# Скорость подачи стола [мм/с]. В моем примере 12.8 мм/с
V_f = f * n * omega / (60 * (10**3))
print("V_f = ", V_f)

# Начальные координаты фрезы (ноль СК - это левый нижний угол заготовки)
tool_start_y = b_workpiece - Hr + D/2
tool_start_x = - D/2 * np.sin(alpha)
# tool_start_x = 0.001
print("tool_start_x = ", tool_start_x)
# Задаем z-буфер
num_points = int(b / dx)  # Кол-во отрезков разбиения
x_values = np.arange(num_points) * dx
buffer_list = np.column_stack((x_values, np.full(num_points, b_workpiece)))
buffer_list_test = np.column_stack((x_values, np.full(num_points, b_workpiece)))

temporary_buffer = []  # Временный буфер для реализации запаздывания

# Часть с зацикливанием, можем взять любое dt и t и тем самым получим процесс фрезерования
step = 0.0000125  # Шаг по времени (т.к. мы взяли 4800 об/мин -> 1/100 оборота время step)
dt = 0.4  # Через сколько остановить выполнение программы
t = 0  # Начальный момент времени(потом будет изменяться с каждым шагом)

# Будем считать, что толщина срезаемого слоя снимается для каждой режущей кромки т.е. будем так их хранить:
# thickness_list = {1: [...], 2: [...], ... }
# где 1, 2, ... - номер режущей кромки, а [...] - список толщин срезаемого слоя, в нем есть нулевые элементы
thickness_list = {}

print("tooth_coord = ", tooth_coord(0))

for i in range(n):
    thickness_list[i] = [0] * int(dt / step + 1)  # Задаем длину списка в словаре толщин для каждой режущей кромки

finish = t + dt  # Конечный момент времени
count_len_thikness_list = 0  # Ведем подсчет какой сейчас номер шага (всего шагов = dt / step)
# Т.к. у нас время дискретно => мы разбиваем t + dt на шаги по времени
# и будем для каждого такого шага искать толщину срезаемого слоя

# Запаздывание
lag = 20
shag = 0

while True:
    if t >= finish:
        plot_figures(tooth_coord(t), buffer_list)  # Строим заготовку и фрезу
        break
    else:
        temporary_buffer.append(tooth_coord(t))  # Добавляем во временное хранилище координаты режущих кромок в момент t

        # Ищем толщину срезаемого слоя для z-буфера и времени t
        find_thickness(buffer_list, t)

        if len(temporary_buffer) == lag:  # Когда кол-во элементов в хранилище станет равно запаздыванию
            # Изменяем координаты z-буфера в соответствии с 0 и 1 значениями, хранящимися во временном хранилище
            buffer_list = part_of_cutting(buffer_list_test, temporary_buffer[0], temporary_buffer[1])
            # print(temporary_buffer[0], temporary_buffer[1])
            temporary_buffer.pop(0)  # Удаляем 0 элемент, тем самым сдвигаем все элементы хранилища влево на один
            # if t > 0.3 and zds:
                    # print(t / step)
                    # break

        # Добавляем step ко времени, переходим к следующему шагу
        t = t + step
        count_len_thikness_list += 1  # Необходимо, чтобы в словаре thickness_list пройти от 0 до dt / step + 1

plot_thickness(step, thickness_list, -1)  # Построение графика толщины срезаемого слоя
# plot_thickness(step, thickness_list, 0)  # Построение графика толщины срезаемого слоя
# plot_thickness(step, thickness_list, 1)  # Построение графика толщины срезаемого слоя
# plot_thickness(step, thickness_list, 2)  # Построение графика толщины срезаемого слоя
# plot_thickness(step, thickness_list, 3)  # Построение графика толщины срезаемого слоя
