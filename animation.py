import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_points_and_rectangle(points, rectangle):
    # points - список кортежей [(x1, y1), (x2, y2), ...]
    # rectangle - список [x, y, width, height]

    # Создаем новый график
    fig, ax = plt.subplots()

    # Рисуем точки
    for point in points:
        x, y = point
        ax.plot(x, y, 'ro')  # 'ro' означает красные кружки (red circles)

    # Рисуем прямоугольник
    x, y, width, height = rectangle
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)

    # Устанавливаем пределы графика
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    plt.show()

# Пример использования
points = [(2, 3), (5, 6)]
rectangle = [0, 0, 6, 4]  # Начало координат в левом нижнем углу, прямоугольник 6x4

draw_points_and_rectangle(points, rectangle)
