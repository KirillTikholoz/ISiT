import os
import cv2
import numpy as np
from .utils_cv import detect_object, search_similar_images
import colorspacious
from .model import Image, init_db
import logging


def bgr_to_rgb(bgr_color):
    r, g, b = bgr_color
    return (b, g, r)


def calculate_ciede2000(color1, color2):
    colo1rgb = bgr_to_rgb(color1)
    colo2rgb = bgr_to_rgb(color2)
    # из RGB в LAB
    lab_color1 = colorspacious.cspace_convert(colo1rgb, "sRGB255", "CIELab")
    lab_color2 = colorspacious.cspace_convert(colo2rgb, "sRGB255", "CIELab")

    # Вычисление CIEDE2000
    delta_e = colorspacious.deltaE(lab_color1, lab_color2, input_space="CIELab")
    return delta_e


def calculate_distances_lab(images):
    images_arr = []
    average_bgr_values = []
    for image in images:
        image_data = image.data
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image_cv2 = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        images_arr.append(image_cv2)
        mask = detect_object(image_cv2)
        masked_image = cv2.bitwise_and(image_cv2, image_cv2, mask=mask)
        average_bgr_values.append(cv2.mean(masked_image, mask=mask)[:3])
    logging.info("Изображения извлечены")

    num_images = len(images_arr)
    distances = np.zeros((num_images, num_images))
    for i in range(num_images):
        for j in range(i + 1, num_images):
            delta_e = calculate_ciede2000(average_bgr_values[i], average_bgr_values[j])
            distances[i, j] = distances[j, i] = delta_e
    logging.info("Матрица расстояний построена")

    path = os.path.join(os.path.dirname(__file__), "../dataRepository/distance_lab.npz")
    np.savez_compressed(path, arr=distances)


def search_similar_images_lab(image_name):
    session = init_db()
    images = session.query(Image).all()

    calculate_distances_lab(images)

    file_path = os.path.join(
        os.path.dirname(__file__), "../dataRepository/distance_lab.npz"
    )
    distances = np.load(file_path)["arr"]
    threshold = 3

    return search_similar_images(images, image_name, distances, threshold)


# interface_lab()
