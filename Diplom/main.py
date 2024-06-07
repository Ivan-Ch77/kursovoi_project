# import Dynamics
# from geomAlg import find_thickness, plot_thickness, plot_figures,\
#     part_of_cutting, fenom_model, plot_force
# from Dynamics import x_Duhamel_start, dx_Duhamel_start, x_Duhamel, dx_Duhamel, sys_param
import numpy as np
from test_geomAlg import find_thickness as ff, part_of_cutting, plot_amplitude, fenom_model, plot_force, plot_thickness, plot_figures
import matplotlib.pyplot as plt  # Plotting functionality
from Dynamics import sys_param


def find_extrema(points):
    extrema = []

    for i in range(1, len(points) - 1):
        prev_point = points[i - 1]
        current_point = points[i]
        next_point = points[i + 1]
        # print("prev_point=", prev_point)
        # print("current_point = ", current_point)
        # print("next_point = ", next_point)
        #
        # # Отладочные сообщения
        # print(f"prev_point: {type(prev_point)}, current_point: {type(current_point)}, next_point: {type(next_point)}")

        if len(points) <= 2:
            extrema.append(current_point)

        # Проверка на локальный максимум
        elif current_point > prev_point and current_point > next_point:
            extrema.append(current_point)

        # Проверка на локальный минимум
        elif current_point < prev_point and current_point < next_point:
            extrema.append(current_point)

    return extrema


# Определяем параметры фрезы и резания
n = 2  # Кол-во зубьев фрезы [безразм]
D = 6.0  # Диаметр фрезы [мм]
a = 4.0  # Ширина резания [мм]
Hr = 3.0  # Радиальная глубина резания [мм]
# omega = 1200  # Скорость вращения шпинделя(начальная) [об/мин]
# domega = 20  # То сколько шагов будет варьировать omega [об/мин]
f = 40.0  # Подача на зуб [мкм]
bx = 150.0  # Длина заготовки [мм]
dx = 0.005  # Разбиение детали dx [мм]


# Высота заготовки(для примера) [мм]
b_workpiece = 50.0


# Начальные координаты фрезы (ноль СК - это левый нижний угол заготовки)
tool_start_y = b_workpiece - Hr + D/2
# tool_start_x = - D/2 * np.sin(alpha)
# tool_start_x = -D/2
tool_start_x = -0.3
print("tool_start_x = ", tool_start_x)
print("tool_start_y = ", tool_start_y)


# Запаздывание
lag = 30


# Феноменологическая модель
Krc = 558.0  # [Н/мм^2]
Ktc = 970.0  # [Н/мм^2]
Krb = 2.0  # [Н/мм^2]
Ktb = 7.7  # [Н/мм^2]

m = 2.3*10**6/((73*2*np.pi)**2)
c = 2.3*10**6
b = 2*m*0.0057*73*2*np.pi
print("b=", b)

p_sys = sys_param(m, b, c)

vibration_amplitude_x = {}  # Список словарей со значениями амплитуд и характерной для них скорости вращения
vibration_amplitude_y = {}  # Список словарей со значениями амплитуд и характерной для них скорости вращения

p_otnos = 0.3
p_otnos_finish = 3.5

while True:
    if p_otnos > p_otnos_finish:
        break
    # print(p_otnos)
# for j in range(int(domega)):
    print('-------------------------------------------------')
    koeff = 0  # Коэффициент, отвечающий за график амплитуд, был ли вызов функции extrema или нет
    max_amplitude = -100
    min_amplitude = 100
    # omega += 20
    # T = 1 / (omega / 60)  # Период одного оборота шпинделя
    # # print('T = ', T)
    # oomega = 2 * np.pi / T  # Угловая скорость рад/с
    # # print('oomega = ', oomega)

    # Собственная частота системы [рад/с]
    p = p_sys[2]
    print(f"p = {p} рад/с")
    # СЧ системы в герцах
    p = p/(2*np.pi)
    print(f"p = {p} Гц")
    print("p_otnos = ", p_otnos)
    p_pk = p/p_otnos
    print(f"p_pk = {p_pk} Гц")
    oomega = p_pk*(2*np.pi)/n
    T = 2*np.pi/oomega
    omega = 60/T
    # # Собственная частота системы
    # p0 = p_sys[3]
    # # Частота прохождения режущих кромок
    # p_pk = oomega*n/(2*np.pi)
    # # Относительная частота вращения шпинделя
    # p_otnos = p0/p_pk

    print(f"omega = {oomega} рад/с")
    print(f"omega = {omega} об/мин")

    list_amplitude_x = []
    list_amplitude_y = []
    vibration_amplitude_x[p_otnos] = list_amplitude_x
    vibration_amplitude_y[p_otnos] = list_amplitude_y



    # Скорость подачи стола [мм/с]. В моем примере 12.8 мм/с
    V_f = f * n * omega / (60 * (10 ** 3))
    # print("V_f = ", V_f)

    # Часть с зацикливанием, можем взять любое dt и t и тем самым получим процесс фрезерования
    step = 60 / omega / 150  # Шаг по времени (т.к. мы взяли 4800 об/мин -> 1/200 оборота время step)
    print("step = ", step)
    # print(j)
    dt = 6  # Через сколько остановить выполнение программы
    t = 0  # Начальный момент времени(потом будет изменяться с каждым шагом)
    t0 = 0
    finish = t + dt  # Конечный момент времени

    # Задаем z-буфер
    num_points = int(bx / dx)  # Кол-во отрезков разбиения
    x_values = np.arange(num_points) * dx
    buffer_list = np.column_stack((x_values, np.full(num_points, b_workpiece)))
    buffer_list_test = np.column_stack((x_values, np.full(num_points, b_workpiece)))

    # Временный буфер для реализации запаздывания
    temporary_buffer = []

    # Будем считать, что толщина срезаемого слоя снимается для каждой режущей кромки т.е. будем так их хранить:
    # thickness_list = {1: [...], 2: [...], ... }
    # где 1, 2, ... - номер режущей кромки, а [...] - список толщин срезаемого слоя, в нем есть нулевые элементы
    thickness_list = {}
    for i in range(n):
        thickness_list[i] = [0] * int(dt / step + 1)  # Задаем длину списка в словаре толщин для каждой режущей кромки
    # pprint.pprint(thickness_list)
    count_len_thikness_list = 0  # Ведем подсчет какой сейчас номер шага (всего шагов = dt / step)
    # Т.к. у нас время дискретно => мы разбиваем t + dt на шаги по времени
    # и будем для каждого такого шага искать толщину срезаемого слоя

    # Будем считать, что окружные и радиальные силы снимаются для каждой режущей кромки т.е. будем так их хранить:
    # fenom_list = {1: [[Fr, Ft],[Fr,Ft], [Fr,Ft],...], 2: [[Fr, Ft],[Fr,Ft], [Fr,Ft],...], ... }
    # где 1, 2, ... - номер режущей кромки, а [...] - список толщин срезаемого слоя, в нем есть нулевые элементы
    fenom_list = {}
    for i in range(n):
        fenom_list[i] = [[0, 0]] * int(dt / step + 1)  # Задаем длину списка в словаре толщин для каждой режущей кромки
    # pprint.pprint(fenom_list)

    result_forces_for_p_otnos1_25 = {}
    for i in range(n):
        result_forces_for_p_otnos1_25[i] = [[0, 0]] * int(dt / step + 1)  # Задаем длину списка в словаре толщин для каждой режущей кромки

    forces = {}
    # Отдельный массив под суммарные силы от всех режущих интрсументов на каждом шаге
    for i in range(int(dt / step + 1)):
        forces[i] = [0, 0]
    # print("forces=", forces)
    # A = 10
    # w = 100
    T = np.linspace(t0, finish, int(finish / step))
    # delta_t = T[1]-T[0]  # step
    # m, b, c = 1, 0, 100
    # p_sys = sys_param(m, b, c)
    # X = np.zeros(len(T))
    # dX = np.zeros(len(T))
    # F_1 = A
    # x0_Duh, y0_Duh = 0, 0
    # dx0_Duh, dy0_Duh = 0, 0
    xy0_Duh = [0, 0]
    dx0dy0_Duh = [0, 0]
    Fxy = np.zeros(2)
    Ft = Fxy
    tooth_cord = []
    X = []
    for i in range(int((finish - t0)/step)+1):
        if t >= finish:
            # plot_figures(t, buffer_list, tooth_cord)  # Строим заготовку и фрезу
            vibration_amplitude_x[p_otnos] = list_amplitude_x
            vibration_amplitude_y[p_otnos] = list_amplitude_y
            # print("vibration_amplitude = ", vibration_amplitude)
            # plot_thickness(step, thickness_list, -1)  # Построение графика толщины срезаемого слоя
            break
        else:
            # list_amplitude = []
            # Fi = np.sqrt(Ft[0] ** 2 + Ft[1] ** 2)
            # Fi_1 = 2 * Fi + 1
            # if t == t0:
            #     xDuh_c = x_Duhamel_start(p_sys, dt, x0_Duh, dx0_Duh, Ft[0])
            #     yDuh_c = x_Duhamel_start(p_sys, dt, x0_Duh, dx0_Duh, Ft[1])
            # else:
            #     xDuh_c = x_Duhamel(p_sys, dt, x0_Duh, dx0_Duh, Ft[0], Fi_1)
            #     yDuh_c = x_Duhamel(p_sys, dt, x0_Duh, dx0_Duh, Ft[1], Fi_1)
            # X[i + 1] = x_Duhamel(p_sys, dt, x0, dx0, F_1, F_2)
            # dX[i + 1] = dx_Duhamel(p_sys, dt, x0, dx0, F_1, F_2)
            # Ищем толщину срезаемого слоя для z-буфера и времени t. Попутно изменяем xy0_Duh, dx0dy0_Duh
            # [thickness_list, fenom_list, tooth_cord, xy0_Duh,  dx0dy0_Duh, forces] = find_thickness(t, buffer_list, count_len_thikness_list, fenom_list, forces, thickness_list, xy0_Duh, dx0dy0_Duh)
            #
            [thickness_list, fenom_list, tooth_cord, xy0_Duh, dx0dy0_Duh, forces] = \
                ff(t, buffer_list,
                   count_len_thikness_list,
                   fenom_list, forces,
                   thickness_list, xy0_Duh,
                   dx0dy0_Duh, step, omega, oomega, tool_start_x, tool_start_y)

            # # Изменяем начальные условия для следующего шага по времени
            x0_Duh, y0_Duh = xy0_Duh[0], xy0_Duh[1]
            dx0_Duh, dy0_Duh = dx0dy0_Duh[0], dx0dy0_Duh[1]

            if x0_Duh >= 0.2 or y0_Duh >= 0.2:
                print(f"Для omega = {omega} режим слишком неустановивш.")
                break
                # plot_thickness(step, vibration_amplitude, 1684)


            # print(tooth_cord)

            # X.append(x_DUHAMEL[1])
            # print(count_len_thikness_list)
            # Ищем силы
            # fenom_list = find_thickness(t, buffer_list, count_len_thikness_list, fenom_list, thickness_list)[1]

            # try:
            #     Fxy[0] = fenom_list[i][count_len_thikness_list][0]
            #     Fxy[1] = fenom_list[i][count_len_thikness_list][1]
            #     print(Fxy[0], '-------------------', Fxy[1], '\n\n')
            # except Exception:
            #     print(i)
            #     print(len(fenom_list))
            #     print(fenom_list)
            #     break

            temporary_buffer.append(tooth_cord)
            # print("tooth_cord", tooth_cord)

            if len(temporary_buffer) == lag:  # Когда кол-во элементов в хранилище станет равно запаздыванию
                # Изменяем координаты z-буфера в соответствии с 0 и 1 значениями, хранящимися во временном хранилище
                buffer_list = part_of_cutting(temporary_buffer[0], temporary_buffer[1], buffer_list)
                # print(temporary_buffer[0], temporary_buffer[1])
                del temporary_buffer[0]  # Удаляем 0 элемент, тем самым сдвигаем все элементы хранилища влево на один

                # if t > 0.3 and zds:
                        # print(t / step)
                        # break

            # Добавляем step ко времени, переходим к следующему шагу
            t += step
            count_len_thikness_list += 1  # Необходимо, чтобы в словаре thickness_list пройти от 0 до dt / step + 1

            if t-step >= 4 and t-step <= 5.5:
                list_amplitude_x.append(x0_Duh)
                list_amplitude_y.append(y0_Duh)
            elif t-step > 5.5 and koeff == 0:
                list_amplitude_x = find_extrema(list_amplitude_x)
                list_amplitude_y = find_extrema(list_amplitude_y)
                vibration_amplitude_x[p_otnos] = list_amplitude_x
                vibration_amplitude_y[p_otnos] = list_amplitude_y
                # plot_thickness(step, vibration_amplitude, omega)
                koeff = 1
            # if 99 * step < t < 100*step:
            #     print('dsdwawd', fenom_model(t))
            # plot_thickness(step, thickness_list, -1)  # Построение графика толщины срезаемого слоя
    p_otnos += 0.05
    # Проверка того, для какого этапа будем выводить силы
    if abs(p_otnos - 1.2) < 0.0000005:
        result_forces_for_p_otnos1_25 = fenom_list
        plot_force(step, result_forces_for_p_otnos1_25, -1)
        # plot_thickness(step, vibration_amplitude_x, p_otnos)
        print("result_forces_for_p_otnos1_25 = ", result_forces_for_p_otnos1_25)

# print(fenom_list)
# print(thickness_list)

plot_amplitude(vibration_amplitude_x)
plot_amplitude(vibration_amplitude_y)
# print("vibration_amplitude = ", vibration_amplitude)

# step = 60 / omega / 150
# plot_thickness(step, thickness_list, -1)  # Построение графика толщины срезаемого слоя
# plot_force(step, result_forces_for_p_otnos1_25, -1)
# plot_force(step, result_forces_for_p_otnos1_25, -1)
# plot_thickness(step, vibration_amplitude_x, p_otnos)
# plot_thickness(step, vibration_amplitude_y, p_otnos)


# print(X)
#
# plt.title("Перемещения")
# plt.xlabel("T")
# plt.ylabel("M")
# # plt.xlim(0.1, 1.1)
# # plt.ylim(-1.8, 1.8)
# plt.grid()
# plt.plot(T, X)
# plt.show()
