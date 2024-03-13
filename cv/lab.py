import os
import cv2
import numpy as np
from utils_cv import print_similar_images, detect_object
import colorspacious

def calculate_ciede2000(color1, color2):
    # из RGB в LAB
    lab_color1 = colorspacious.cspace_convert(color1, "sRGB1", "CAM02-UCS")
    lab_color2 = colorspacious.cspace_convert(color2, "sRGB1", "CAM02-UCS")

    # Вычисление CIEDE2000
    delta_e = colorspacious.deltaE(lab_color1, lab_color2, input_space="CAM02-UCS")
    return delta_e


images_folder = "../images"
image_files = os.listdir(images_folder)

images = []
average_bgr_values = []
for image_file in image_files:
    image_path = os.path.join(images_folder, image_file)
    image = cv2.imread(image_path)
    images.append(image)
    mask = detect_object(image)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    average_bgr_values.append(cv2.mean(masked_image, mask=mask)[:3])


# матрица расстояний между средними значениями цветов
num_images = len(images)
distances = np.zeros((num_images, num_images))
for i in range(num_images):
    for j in range(i+1, num_images):
        distances[i, j] = distances[j, i] = calculate_ciede2000(
            average_bgr_values[i],
            average_bgr_values[j]
        )


# порог сходства
threshold = 4

print_similar_images(image_files, images_folder, distances, threshold)
