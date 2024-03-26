import cv2
import numpy as np

# import matplotlib.pyplot as plt
from .model import Image, init_db


def detect_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            cv2.drawContours(
                mask, [contour], -1, (255, 255, 255), -1
            )  # контур белым цветом
    return mask


# def display_image(image):
#     image_data = image.data
#     image_array = np.frombuffer(image_data, dtype=np.uint8)
#     image_cv2 = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#
#     # image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
#     plt.imshow(image_rgb)
#     plt.axis('off')
#     plt.show()
#
#
# def display_similar_images(image, similar_list):
#     print(f"Изображение {image.name} похоже на:")
#     display_image(image)
#     for similar_image_name in similar_list:
#         display_image(similar_image_name)
#
#
# def print_similar_images(images, distances, threshold):
#     num_images = len(images)
#     similar_images = []
#     for i in range(num_images):
#         similar_to_i = []
#         for j in range(num_images):
#             if distances[i, j] < threshold and i != j:
#                 similar_to_i.append(images[j])
#         similar_images.append((images[i], similar_to_i))
#
#     # print("Изображения с похожими цветами:")
#     # for image, similar_list in similar_images:
#     #     print(f"Изображение {image} похоже на: {', '.join(similar_list)}")
#
#     while True:
#         image_name = input("Введите имя изображения для поиска похожих изображений: ")
#         found = False
#         for image, similar_list in similar_images:
#             if image.name == image_name:
#                 display_similar_images(image, similar_list)
#                 break


def search_similar_images(images, image_name, distances, threshold):
    num_images = len(images)
    similar_images = []
    for i in range(num_images):
        similar_to_i = []
        for j in range(num_images):
            if distances[i, j] < threshold and i != j:
                similar_to_i.append(images[j])
        similar_images.append((images[i], similar_to_i))

    for image, similar_list in similar_images:
        if image.name == image_name:
            list = [image] + similar_list
            return list


def extract_all_image_name():
    session = init_db()
    images = session.query(Image).all()
    images_names = [image.name for image in images]
    return images_names
