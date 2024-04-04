import cv2
import numpy as np
from .utils_cv import detect_object, search_similar_images
from .model import Image, init_db
import os
import logging


def calculate_average_hsv(image):
    mask = detect_object(image)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

    hsv_values = cv2.mean(hsv_image, mask=mask)
    return hsv_values[0], hsv_values[1], hsv_values[2]


def calculate_distances_hsv(images):
    images_arr = []
    average_hsv_values = []
    for image in images:
        image_data = image.data
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image_cv2 = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        # image_path = os.path.join(images_folder, image_file)
        # image = cv2.imread(image_path)
        images_arr.append(image_cv2)
        average_hsv_values.append(calculate_average_hsv(image_cv2))
    logging.info("Изображения извлечены")

    # матрица расстояний между средними значениями цветов (между двумя точками в евклидовом пространстве)
    num_images = len(images_arr)
    distances = np.zeros((num_images, num_images))
    for i in range(num_images):
        for j in range(i + 1, num_images):
            distances[i, j] = distances[j, i] = np.sqrt(
                (average_hsv_values[i][0] - average_hsv_values[j][0]) ** 2
                + (average_hsv_values[i][1] - average_hsv_values[j][1]) ** 2
                + (average_hsv_values[i][2] - average_hsv_values[j][2]) ** 2
            )

    path = os.path.join(os.path.dirname(__file__), "../dataRepository/distance_hsv.npz")
    np.savez_compressed(path, arr=distances)
    logging.info("Матрица расстояний hsv вычислена")


def search_similar_images_hsv(image_name):
    session = init_db()
    images = session.query(Image).all()

    calculate_distances_hsv(images)

    file_path = os.path.join(
        os.path.dirname(__file__), "../dataRepository/distance_hsv.npz"
    )
    distances = np.load(file_path)["arr"]
    threshold = 15

    return search_similar_images(images, image_name, distances, threshold)
