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
tool_start_x = 0
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
    # Fr = Krc * a * h + Krb * a  # Такую формулу использовать?
    # Ft = Ktc * a * h + Ktb * a  # Такую формулу использовать?
    Fr = Krc * a * h  # Такую формулу использовать?
    Ft = Ktc * a * h  # Такую формулу использовать?
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
            Ft - список сил на начало текущего шага в проекциях [Fx, Fy]    '''
    # Начальные условия на момент шага t
    # x0_Duh, y0_Duh = xyDuh[0], xyDuh[1]
    # dx0_Duh, dy0_Duh = dxdyDuh[0], dxdyDuh[1]
    x0_Duh = xyDuh[0]
    dx0_Duh = dxdyDuh[0]

    # Fxy = np.zeros(2)

    Ft = np.zeros(2)
    # Взяли рандомные числа, в дальнейшем в цикле мы их изменим на числа из списка forces
    Fi = Ft[0]
    Fi_1 = 2 * Fi + 1
    F_proc = 1
    eps = 0.01

    # if count == 0:
    #     Ft = np.zeros(2)
    #     # Взяли рандомные числа, в дальнейшем в цикле мы их изменим на числа из списка forces
    #     Fi = Ft[0]
    #     Fi_1 = 2 * Fi + 1
    #     F_proc = 1
    #     # yDuh_c = x_Duhamel_start(p_sys, step, x0_Duh, dx0_Duh, Ft[1])
    # elif count >= 1:  # При добавлении оси y будем искать Fi как np.sqrt(Ft[0]**2+Ft[1]**2)  # конец текущего шага
    #     Ft = forces[count-1]
    #     Fi = Ft[0]
    #     Fi_1 = 2 * Fi + 1
    #     F_proc = Fi


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
    while abs(Fi_1-Fi)/F_proc > eps:
        # print('-.-.-.-.-.-.-.-.-.-.-..-.-.-')
        # print("abs(Fi-Fi_1)/F_proc  =  ", abs(Fi_1-Fi)/F_proc)
        if iteration_number != 0:
            print(iteration_number)
        if iteration_number == 0:
            # xc_Duhamel = x_Duhamel_start(p_sys, step, x0_Duh, dx0_Duh, Ft[0])
            xc_Duhamel = x_Duhamel(p_sys, step, x0_Duh, dx0_Duh, Ft[0], Ft[0])
            # print('xc_Duhamel0 = ', xc_Duhamel)
            # yc_Duhamel = x_Duhamel_start(p_sys, step, y0_Duh, dy0_Duh, Ft[1])
        else:
            xc_Duhamel = x_Duhamel(p_sys, step, x0_Duh, dx0_Duh, Ft[0], Fi_1)
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
            if distance != 0:
                print("xc_Duhamel = ", xc_Duhamel, '\n', "distance = ", distance, '\n', "Ft = ", Ft)
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
        if iteration_number == 0:
            #Зануляем Fy, чтобы исследовать только по x
            Ft[1] = 0
            F_sum[1] = 0

            Fi = np.sqrt(F_sum[0] ** 2 + F_sum[1] ** 2)
        else:
            # Зануляем Fy, чтобы исследовать только по x
            F_sum[1] = 0

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
        dx0_Duh = dx_Duhamel_start(p_sys, step, x0_Duh, dx0_Duh, Ft[0])
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



