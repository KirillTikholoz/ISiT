import cv2
import numpy as np
import matplotlib.pyplot as plt


# Функция для вычисления средней яркости в окне 64x64 пикселя
def compute_window_brightness(image, x, y):
    window = image[y : y + 64, x : x + 64]  # Выбираем окно 64x64
    return np.mean(window) / 3  # Усредняем и нормируем по каналам


# Загружаем изображения
image_paths = ["../images/task1/image{}.jpg".format(i) for i in range(1, 21)]
images = [cv2.imread(path) for path in image_paths]

# Значения экспозиции (EV) для каждого изображения
exposures = list(range(-100, 100, 10))

# Вычисляем яркость для каждого изображения
brightness_values = [compute_window_brightness(image, x=0, y=0) for image in images]

# Построение графика зависимости logBr от EV
log_brightness = np.log(brightness_values)
plt.plot(exposures, log_brightness)
plt.xlabel("Значение экспозиции (EV)")
plt.ylabel("log(Яркость)")
plt.title("Зависимость log(Яркость) от значения экспозиции (EV)")
plt.grid(True)
plt.show()
