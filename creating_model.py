import numpy as np
import matplotlib.pyplot as plt
import pprint
# import matplotlib.patches as patches
# from ipywidgets import interact
# from matplotlib.widgets import Slider


def plot_figures(points1, points2):
    '''Здесь происходит пострение заготовки в момент времени t+dt и фрезы в моменты времени t и t+dt
    points1 - список координат фрезы [[x1, y1] , [x2, y2], ..., [xn, yn]], где n - кол-во зубьев фрезы
    points2 список координат z-буфера вида: [..., [xi, yi],...], его длина равна b_заготовки / dx_буфера'''
    # Извлекаем координаты x и y из списков точек
    x1, y1 = zip(*points1)
    X2, Y2 = zip(*points2)
    x3, y3 = zip(*tooth_coord(1))
    x4, y4 = zip(*tooth_coord(1.0125))

    # Строим первую фигуру - набор точек
    plt.scatter(x1, y1, label='Фреза t=1.4')
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


def find_thickness(buffer, t):
    '''Функция поиска толщины срезаемого слоя
    buffer - z-буфер поверхности
    t - начальный момент времени. Нужен, чтобы построить прямую y = mx+b
    dt - момент времени для которого определяем толщину срезаемого слоя'''

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

    tooth_coordinate = tooth_coord(t)

    for i in range(n):

        xi_new = tooth_coordinate[i][0]
        yi_new = tooth_coordinate[i][1]

        j = int(xi_new // dx)
        xj = buffer[j][0]
        yj = buffer[j][1]
        xj1 = buffer[j+1][0]
        yj1 = buffer[j+1][1]

        # if yj == yj1 == b_workpiece:
        #     continue
        # Ищем уравнение прямой, проходящей через эти точки (y = mx + b)
        try:
            mj = (yj1 - yj)/(xj1 - xj)
            bj = yj - mj*xj
        except RuntimeWarning:
            print('RuntimeWarning')

        # for j, (x, y) in enumerate(buffer):

        # Здесь нужно, чтобы yi_new была меньше y = m*xi_new+b1 для j и j+1 координат z-буфера
        if yi_new < mj*xi_new + bj < b_workpiece:
            # Определение коэф-ов для режущей кромки
            x_tooth = xi_new
            y_tooth = yi_new
            # x_tooth = tooth_coord(t + dt)[i][0]
            # y_tooth = tooth_coord(t + dt)[i][1]
            # temporary_buffer.append((x_tooth, y_tooth))
            tool_x = tool_start_x + V_f * (t)  # Координата x центра фрезы через время t
            tool_y = tool_start_y  # Координата y центра фрезы через время t
            m_tooth = (y_tooth - tool_y) / (x_tooth - tool_x)
            b_tooth = tool_y - m_tooth * tool_x
            # Определение коэф-ов для z-буффера
            m_buffer = (buffer[j+1][1] - buffer[j][1]) / (buffer[j+1][0] - buffer[j][0])
            b_buffer = buffer[j][1] - m_buffer * buffer[j][0]

            intersection_point = find_intersection_point(m_tooth, b_tooth, m_buffer, b_buffer)

            # Часть кода, котора исключает неточности, связанные с разбиением, т.к. иногда может возникнуть ситуация,
            # когда точка пересечения intersection_point и buffer[j][0] находятся в разных ячейках z-буфера
            # (т.е. точка пересечения лежит не в buffer[j], а в j-1 или еще дальше в зависимости от параметров резания)
            # , тогда нужно искать точку пересечения не с j-ой поверхностью заготовки, а с j - k поверхностью заготовки
            k = 10
            while True:
                k -= 1
                if intersection_point[0] < buffer[j - k][0]:
                    m_buffer = (buffer[j - k][1] - buffer[j - k-1][1]) / (buffer[j-k][0] - buffer[j - k-1][0])
                    b_buffer = buffer[j - k-1][1] - m_buffer * buffer[j - k-1][0]

                    intersection_point = find_intersection_point(m_tooth, b_tooth, m_buffer, b_buffer)
                    break

            point1 = intersection_point
            point2 = (x_tooth, y_tooth)

            if intersection_point is not None:
                distance = distance_between_points(point1, point2)
                thickness_list[i][count_len_thikness_list] = distance
                if distance > 0.05:
                    if point1[0] > point2[0]:
                        print(point1, point2)
            else:
                print("Прямые параллельны, невозможно найти расстояние")
            break


def checking_the_cutting_tool(t, dt, i):

    # Получаем массивы длины n из списка множеств вида (x, y)
    tooth_coordinate = tooth_coord(t)  # Положение зубьев фрезы в момент времени t
    tooth_coordinate_new = tooth_coord(t+dt)  # Положение зубьев фрезы через время dt

    list_xiyi = []  # Структура будет содержать списки из 2х значений: (xi, yi)
    list_xiyi_new = []  # Структура будет содержать списки из 2х значений: (xi_new, yi_new)
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
    xi = list_xiyi[0][0]
    yi = list_xiyi[0][1]
    xi_new = list_xiyi_new[0][0]
    yi_new = list_xiyi_new[0][1]

    return xi, yi, xi_new, yi_new


# def part_of_cutting(buffer, t, dt, count_len_thikness_list=0):
def part_of_cutting(buffer, temp_buf0, temp_buf1, count_len_thikness_list=0):
    """Функция, которая отвечает за резание заготовки.
        На вход получает buffer - z-буфер
                         temp_buf0 - положение зубьев фрезы в момент времени t - lag*step
                         temp_buf1 - положение зубьев фрезы в момент времени t - lag*step + 1
                         lag - кол-во шагов запаздывания"""
    print(temp_buf0, '\n', temp_buf1)
    # Один раз пробегаюсь по всем режущим кромкам
    for i in range(n):
        # xi = checking_the_cutting_tool(t, dt, i)[0]
        # yi = checking_the_cutting_tool(t, dt, i)[1]
        # xi_new = checking_the_cutting_tool(t, dt, i)[2]
        # yi_new = checking_the_cutting_tool(t, dt, i)[3]
        # Получаем массивы длины n из списка множеств вида (x, y)

        # Определяем координаты режущих кромок в двух положениях
        xi = temp_buf0[i][0]
        yi = temp_buf0[i][1]
        xi_new = temp_buf1[i][0]
        yi_new = temp_buf1[i][1]
        # Если y не изменяется -> пропускаем шаг
        if yi == yi_new == b_workpiece:
            continue
        # Ищем уравнение прямой, проходящей через эти точки (y = mx + b)
        try:
            m = (yi_new - yi)/(xi_new - xi)
            b1 = yi - m*xi
        except RuntimeWarning:
            print('RuntimeWarning')

        # Находим ячейку, в которой в момент времени t - lag*step + 1 остановилась фреза
        j = int(xi_new // dx)
        # Определяем координаты x,y, которые отвечают координатам начала ячейки
        x = buffer[j][0]
        y = buffer[j][1]
        # Изменяем координаты y точек буфера, лежащих между xi и xi_new
        if xi <= x <= xi_new and y > m*x + b1:
            buffer[j] = (x, m*x + b1)
        #  Один раз пробегаюсь по буферу
        # for j, (x, y) in enumerate(buffer):
            # if buffer[j-1][0] <= xi_new <= buffer[j][0] and yi_new < y < b_workpiece:
            #     # Определение коэф-ов для режущей кромки
            #     x_tooth = tooth_coord(t+dt)[i][0]
            #     y_tooth = tooth_coord(t + dt)[i][1]
            #     temporary_buffer.append((x_tooth, y_tooth))
            #     tool_x = tool_start_x + V_f * (t+dt)  # Координата x центра фрезы через время t
            #     tool_y = tool_start_y  # Координата y центра фрезы через время t
            #     m_tooth = (y_tooth - tool_y) / (x_tooth - tool_x)
            #     b_tooth = tool_y - m_tooth * tool_x
            #     # Определение коэф-ов для z-буффера
            #     m_buffer = (buffer[j][1] - buffer[j-1][1]) / (buffer[j][0] - buffer[j-1][0])
            #     b_buffer = buffer[j-1][1] - m_buffer * buffer[j-1][0]



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



                # def find_intersection_point(m1, b1, m2, b2):
                #     if m1 == m2:
                #         # Прямые параллельны, нет точки пересечения
                #         return None
                #     else:
                #         x = (b2 - b1) / (m1 - m2)
                #         y = m1 * x + b1
                #         return x, y
                #
                # def distance_between_points(point1, point2):
                #     x1, y1 = point1
                #     x2, y2 = point2
                #     distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                #     # print(distance)
                #     return distance
                #
                # intersection_point = find_intersection_point(m_tooth, b_tooth, m_buffer, b_buffer)
                # point1 = intersection_point
                # point2 = (x_tooth, y_tooth)
                #
                # if intersection_point is not None:
                #     distance = distance_between_points(point1, point2)
                #     thickness_list[i][count_len_thikness_list] = distance
                # else:
                #     print("Прямые параллельны, невозможно найти расстояние")
            # Изменяем координаты y точек буфера, лежащих между xi и xi_new
            # if xi <= x <= xi_new and y > m*x + b1:
            #     buffer[j] = (x, m*x + b1)
                # break
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
    plt.xlim(0.315, 0.319)
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

# Задаем z-буфер
num_points = int(b / dx)  # Кол-во отрезков разбиения
x_values = np.arange(num_points) * dx
buffer_list = np.column_stack((x_values, np.full(num_points, b_workpiece)))
buffer_list_test = np.column_stack((x_values, np.full(num_points, b_workpiece)))

# thickness_list = [0] * len(buffer_list)

temporary_buffer = []  # Временный буфер для реализации запаздывания


# Часть с зацикливанием, можем взять любое dt и t и тем самым получим процесс фрезерования
step = 0.000125  # Шаг по времени (т.к. мы взяли 4800 об/мин -> 1/100 оборота время step)
dt = 0.4  # Через сколько остановить выполнение программы
t = 1  # Начальный момент времени(потом будет изменяться с каждым шагом)

# Будем считать, что толщина срезаемого слоя снимается для каждой режущей кромки т.е. будем так их хранить:
# thickness_list = {1: [...], 2: [...], ... }
# где 1, 2, ... - номер режущей кромки, а [...] - список толщин срезаемого слоя, в нем есть нулевые элементы
thickness_list = {}


for i in range(n):
    thickness_list[i] = [0] * int(dt / step + 1)  # Задаем длину списка в словаре толщин для каждой режущей кромки

finish = t + dt  # Конечный момент времени
count_len_thikness_list = 0  # Ведем подсчет какой сейчас номер шага (всего шагов = dt / step)
# Т.к. у нас время дискретно => мы разбиваем t + dt на шаги по времени
# и будем для каждого такого шага искать толщину срезаемого слоя

# Запаздывание
lag = 20

while True:
    if t >= finish:
        plot_figures(tooth_coord(t), buffer_list)  # Строим заготовку и фрезу
        break
    else:

        temporary_buffer.append(tooth_coord(t))  # Добавляем во временное хранилище координаты режущих кромок в момент t

        if len(temporary_buffer) == lag:  # Когда кол-во элементов в хранилище станет равно запаздыванию
            # Изменяем координаты z-буфера в соответствии с 0 и 1 значениями, хранящимися во временном хранилище
            buffer_list = part_of_cutting(buffer_list_test, temporary_buffer[0], temporary_buffer[1])
            temporary_buffer.pop(0)  # Удаляем 0 элемент, тем самым сдвигаем все элементы хранилища влево на один

            # part_of_cutting(buffer_list, t - step*99, step)
            # temporary_buffer.pop(0)
            # score_lag -= 1

        # Ищем толщину срезаемого слоя для z-буфера и времени t
        find_thickness(buffer_list, t)

        # Добавляем step ко времени, переходим к следующему шагу
        t = t + step
        count_len_thikness_list += 1  # Необходимо, чтобы в словаре thickness_list пройти от 0 до dt / step + 1
        # print(t)
# pprint.pprint(temporary_buffer)
# print(len(temporary_buffer))
# print(thickness_list)
# print(len(thickness_list[0]))
# print(count_len_thikness_list)
# print(''.join(str(thickness_list)))
plot_thickness(step, thickness_list, -1)  # Построение графика толщины срезаемого слоя


