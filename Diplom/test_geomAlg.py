from Dynamics import x_Duhamel_start, dx_Duhamel_start, x_Duhamel, dx_Duhamel, sys_param
import numpy as np
import matplotlib.pyplot as plt


# Определяем параметры фрезы и резания
n = 2  # Кол-во зубьев фрезы [безразм]
D = 6.0  # Диаметр фрезы [мм]
a = 4.0  # Ширина резания [мм]
Hr = 3.0  # Радиальная глубина резания [мм]
omega = 1684  # Скорость вращения шпинделя [об/мин]
f = 40.0  # Подача на зуб [мкм]
bx = 150.0  # Длина заготовки [мм]
dx = 0.005  # Разбиение детали dx [мм]

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
tool_start_x = -0.3
print("tool_start_x = ", tool_start_x)
print("tool_start_y = ", tool_start_y)

# Часть с зацикливанием, можем взять любое dt и t и тем самым получим процесс фрезерования
step = 60/omega/200  # Шаг по времени (т.к. мы взяли 4800 об/мин -> 1/200 оборота время step)


# # Феноменологическая модель
Krc = 558.0  # [Н/мм^2]
Ktc = 970.0  # [Н/мм^2]
Krb = 2.0  # [Н/мм^2]
Ktb = 7.7  # [Н/мм^2]

m = 2.3*10**6/((73*2*np.pi)**2)
c = 2.3*10**6
b = 2*m*0.0057*73*2*np.pi

p_sys = sys_param(m, b, c)

print(p_sys)


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



def plot_thickness(step, thickness_list, number=-1):
    ''' Функция строит график толщины срезаемого слоя от времени
    На вход она принимает: step - шаг разбиения по времени
                           thickness_list - словарь со списком толщин для каждой режущей кромки
                           number - номер режущей кромки, для которой нужно вывести толщины,
                           если number = -1 - выводим для всех кромок (отсчет идет от 0 кромки)'''
    # Создание списка значений x и y для построения графика
    if number != -1:
            x = [i * step for i in range(1, len(thickness_list[number]) + 1)]
            print(x)
            y = thickness_list[number]
            print(y)
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


def plot_amplitude(data_dict):
    """
    Функция для построения графика значений словаря словарей.

    :param data_dict: Словарь, где ключи - метки на оси X,
                      значения - списки чисел на оси Y.
    """
    for omega, values in data_dict.items():
        if not isinstance(omega, (int, float, str)):
            raise TypeError(f"Key '{omega}' is not a valid type. Must be int, float, or str.")
        if not all(isinstance(value, (int, float)) for value in values):
            raise TypeError(
                f"One or more values in the list for key '{omega}' are not valid types. Must be int or float.")

        plt.scatter([omega] * len(values), values, s=5, label=f'{omega}', color='blue', alpha=0.7)

    plt.xlabel('p, Hz')
    plt.ylabel('Amplitude, mm')
    plt.title('Extremes of vibrations')
    # plt.legend(loc='upper right', title='Omega')
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
    Fr = Krc * a * h + Krb * a  # Такую формулу использовать?
    Ft = Ktc * a * h + Ktb * a  # Такую формулу использовать?
    # Fr = Krc * a * h  # Такую формулу использовать?
    # Ft = Ktc * a * h  # Такую формулу использовать?
    return [Fr, Ft]

def decart_fenom_model(cosfi, sinfi, FrFt: list):
    Fr, Ft = FrFt[0], FrFt[1]
    Fx = Fr * cosfi + Ft * sinfi
    Fz = Ft * cosfi - Fr * sinfi
    return Fx, Fz

# Функция математически определяет distance (толщину срез.слоя)
def dist(x_tooth, y_tooth, tool_x, tool_y, buffer):
    ''' Возвращает distance - толщину срез. слоя'''
    j = int(x_tooth // dx)  # int всегда округляет вниз, так что это нам подходит
    xj = buffer[j][0]
    yj = buffer[j][1]
    xj1 = buffer[j + 1][0]
    yj1 = buffer[j + 1][1]
    # Ищем уравнение прямой, проходящей через эти точки (y = mx + b)
    try:
        mj = k_xy(xj, yj, xj1, yj1)
        bj = b_xy(xj, yj, xj1, yj1)
    except RuntimeWarning:
        print('RuntimeWarning')
    # print('y_tooth = ', y_tooth, 'mj * x_tooth + bj = ', mj * x_tooth + bj, 'b_workpiece = ', b_workpiece)
    # Здесь нужно, чтобы yi_new была меньше y = mj*xi_new + bj для j и j+1 координат z-буфера
    if y_tooth < mj * x_tooth + bj < b_workpiece:
        # Определение коэф-ов для режущей кромки
        # tool_x = tool_start_x + V_f * t - xDuh_c  # Координата x центра фрезы через время t
        # tool_y = tool_start_y - yDuh_c  # Координата y центра фрезы через время t
        m_tooth = k_xy(x_tooth, y_tooth, tool_x, tool_y)
        b_tooth = b_xy(x_tooth, y_tooth, tool_x, tool_y)
        # Определение коэф-ов для z-буффера
        m_buffer = k_xy(buffer[j][0], buffer[j][1], buffer[j + 1][0], buffer[j + 1][1])
        b_buffer = b_xy(buffer[j][0], buffer[j][1], buffer[j + 1][0], buffer[j + 1][1])

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
                m_buffer = k_xy(buffer[j - k1][0], buffer[j - k1][1], buffer[j - k1 + 1][0], buffer[j - k1 + 1][1])
                b_buffer = b_xy(buffer[j - k1][0], buffer[j - k1][1], buffer[j - k1 + 1][0], buffer[j - k1 + 1][1])

                intersection_point = xy(m_tooth, b_tooth, m_buffer, b_buffer)

                if intersection_point[0] >= buffer[j - k1][0] or k1 > k:
                    break

        point1 = intersection_point  # Точка пересечения
        if intersection_point is not None:
            distance = distance_between_points(point1[0], point1[1], x_tooth, y_tooth)
    else:
        distance = 0
    return distance

def find_thickness(t, buffer, count, fenlist, forces, thicklist, xyDuh, dxdyDuh):
    '''     Fi - сила на конец текущего шага
            Fi_1 - сила на начало текущего шага
            Ft - список сил на начало текущего шага в проекциях [Fx, Fy]
            forces - суммарные силы по шагам [F_sunX, F_sumY]
            fenlist - словарь с силами по режущим кромкам {..., i: [Fx, Fy], ...}   '''
    # Начальные условия на момент шага t
    x0_Duh, y0_Duh = xyDuh[0], xyDuh[1]
    dx0_Duh, dy0_Duh = dxdyDuh[0], dxdyDuh[1]
    # x0_Duh = xyDuh[0]
    # dx0_Duh = dxdyDuh[0]

    # Fxy = np.zeros(2)

    # Ft = np.zeros(2)
    if count != 0:
        Ft = forces[count-1]
    else:
        Ft = np.zeros(2)
    # # Взяли рандомные числа, в дальнейшем в цикле мы их изменим на числа из списка forces
    Fi_1 = np.sqrt(Ft[0]**2 + Ft[1]**2)
    Fi = 2 * Fi_1 + 1
    F_proc = 1
    eps = 0.001

    # if count != 0:
    #     if forces[count-1] == [0, 0]:
    #         Ft = np.zeros(2)
    #         # Взяли рандомные числа, в дальнейшем в цикле мы их изменим на числа из списка forces
    #         Fi = Ft[0]
    #         Fi_1 = 2 * Fi + 1
    #         F_proc = 1
    #     else:  # При добавлении оси y будем искать Fi как np.sqrt(Ft[0]**2+Ft[1]**2)  # конец текущего шага
    #         Ft = forces[count-1]
    #         Fi = Ft[0]
    #         Fi_1 = 2 * Fi + 1
    #         F_proc = Fi


    # задаем этот блок, чтобы питон не ругался
    ############################################3
    # Временное хранилище для суммарного по всем реж. кромкам Fx и Fy
    # F_sum = np.zeros(2)
    # xc_Duhamel = x0_Duh
    # yc_Duhamel = y0_Duh

    # x_tooth = tool_start_x + 3
    # y_tooth = tool_start_y - 10
    ##################################

    iteration_number = 0

    # Список, в который мы будет записывать координаты режущих кромок на этом шаге, чтобы сделать return
    coord = []

    # print('---------------------------------------')
    # print("count: ", count)
    while abs(Fi_1-Fi)/F_proc > eps:
        # print("iteration_number:  ", iteration_number)
        # print('-.-.-.-.-.-.-.-.-.-.-..-.-.-')
        # print("abs(Fi-Fi_1)/F_proc  =  ", abs(Fi_1-Fi)/F_proc)
        # print("Fi_1 = ", Fi_1, "Fi = ", Fi)
        if iteration_number == 0:
            # xc_Duhamel = x_Duhamel_start(p_sys, step, x0_Duh, dx0_Duh, Ft[0])
            xc_Duhamel = x_Duhamel(p_sys, step, x0_Duh, dx0_Duh, Ft[0], Ft[0])
            # print('xc_Duhamel0 = ', xc_Duhamel)
            yc_Duhamel = x_Duhamel_start(p_sys, step, y0_Duh, dy0_Duh, Ft[1])
        else:
            xc_Duhamel = x_Duhamel(p_sys, step, x0_Duh, dx0_Duh, Fi_1, Fi)
            # print("xc_Duhamel = ", xc_Duhamel)
            # print(f'xc_Duhamel{iteration_number} = ', xc_Duhamel)
            # yc_Duhamel = x_Duhamel(p_sys, step, y0_Duh, dy0_Duh, Ft[1], Fi_1)

        # Задаем положение центра фрезы
        tool_x = tool_start_x +V_f*t - abs(xc_Duhamel*10**3)
        # tool_x = tool_start_x + V_f * t
        tool_y = tool_start_y
        # tool_y = tool_start_y + V_f * t - yc_Duhamel*10**3


        # # Временное хранилище для суммарного по всем реж. кромкам Fx и Fy
        F_sum = np.zeros(2)

        # Проходимся по каждой режущей кромке
        for i in range(n):
            # Находим положение i-ой режущей кромки
            # Шаг по углу переходя от одной кромки к другой
            angle_step = 2 * np.pi / n
            # Сам угол
            angle = i * angle_step + oomega * t
            x_tooth = tool_x + D / 2 * np.cos(angle)
            y_tooth = tool_y + D / 2 * np.sin(angle)
            # print('xy_tooth', x_tooth, y_tooth)

            # if x_tooth > bx or y_tooth > b_workpiece:
            #     print('Пропуск')
            #     # print(x_tooth, bx)
            #     # print(y_tooth, b_workpiece)
            #     continue
            # print(x_tooth, '--------------', y_tooth)
            coord.append([x_tooth, y_tooth])

            distance = dist(x_tooth, y_tooth, tool_x, tool_y, buffer)
            # print("distance = ", distance)
            # if distance != 0:
                # print("xc_Duhamel = ", xc_Duhamel, '\n', "distance = ", distance, '\n', "Ft = ", Ft)
            # print('dist = ', distance)

            # Записываем для i-ой режущей кромки толщину срезаемого слоя в словарь thickness_list
            thicklist[i][count] = distance

            FrFt = fenom_model(distance)
            # print("FrFt=", FrFt)
            cosfi = 2 * (x_tooth - tool_x) / D
            sinfi = 2 * (y_tooth - tool_y) / D
            Fx, Fy = decart_fenom_model(cosfi, sinfi, FrFt)
            # print("iteration_number=", iteration_number, '        ', 'FX FY', Fx, Fy)
            fenlist[i][count] = [Fx, Fy]  # Находим силы Fx, Fy и записываем их в итоговый словарь списков [Fx, Fy]
            F_sum[0] += Fx
            F_sum[1] += Fy
            # print("F_sum = ", F_sum)
        # print("F_sum = ", F_sum)
        # Зануляем Fy, чтобы исследовать только по x
        Ft[1] = 0
        F_sum[1] = 0
        if iteration_number == 0:
            Fi = np.sqrt(F_sum[0] ** 2 + F_sum[1] ** 2)
        else:
            Fi_1 = Fi
        Fi = np.sqrt(F_sum[0] ** 2 + F_sum[1] ** 2)
        # print("Fi = ", Fi, "Fi_1 = ", Fi_1)
        iteration_number += 1
        if iteration_number > 100:
            print(iteration_number)
            break
        if Fi != 0:
            F_proc = Fi
        else:
            F_proc = 1
        # iteration_number += 1
    if iteration_number == 1:
        dx0_Duh = dx_Duhamel_start(p_sys, step, x0_Duh, dx0_Duh, Fi)
        # dy0_Duh = dx_Duhamel_start(p_sys, step, y0_Duh, dy0_Duh, Ft[1])
    else:
        dx0_Duh = dx_Duhamel(p_sys, step, x0_Duh, dx0_Duh, Ft[0], F_sum[0])
        # dy0_Duh = dx_Duhamel(p_sys, step, y0_Duh, dy0_Duh, Ft[1], F_sum[1])

    x0_Duh = xc_Duhamel
    y0_Duh = 0
    dy0_Duh = 0

    result_xyDuh = [x0_Duh, y0_Duh]
    result_dxdyDuh = [dx0_Duh, dy0_Duh]
    forces[count] = F_sum
    # print("F_sum = ", F_sum)
    # print(forces)

    # print("abs(Fi-Fi_1)/F_proc  =  ", abs(Fi_1-Fi)/F_proc)


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



