import os

import cv2
import numpy as np


images_folder = "../images"
image_filename = "25af62b32fe771ace65a5b820fe0f34875dc9fb6_f268da5da8e53e2f7ab4bc57521e8bdb7a1cb8e2.jpg"

combined_images_folder = "../combined_images"
combined_images_filename = "comb_image.jpg"

image_path = os.path.join(images_folder, image_filename)
combined_images_path = os.path.join(combined_images_folder, combined_images_filename)

# Загрузка изображений
query_image = cv2.imread(combined_images_path, cv2.IMREAD_GRAYSCALE)
train_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Инициализация детектора и описателя (например, ORB)
orb = cv2.ORB_create()

# Нахождение ключевых точек и их описаний для обоих изображений
keypoints_query, descriptors_query = orb.detectAndCompute(query_image, None)
keypoints_train, descriptors_train = orb.detectAndCompute(train_image, None)

# Настройка параметров BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Сопоставление дескрипторов
matches = bf.match(descriptors_query, descriptors_train)

# Сортировка сопоставлений по расстоянию
matches = sorted(matches, key=lambda x: x.distance)

# Отображение первых 10 сопоставлений
matching_result = cv2.drawMatches(query_image, keypoints_query, train_image, keypoints_train, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Количество сопоставлений
num_matches = len(matches)
# Среднее расстояние между сопоставлениями
mean_distance = np.mean([match.distance for match in matches])

# Визуальная оценка
matching_result = cv2.drawMatches(query_image, keypoints_query, train_image, keypoints_train, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.putText(matching_result, f"Matches: {num_matches}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(matching_result, f"Mean Distance: {mean_distance:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("Matching result", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Проверка условий успешного обнаружения объекта
if num_matches > 50 and mean_distance < 50:
    print("Объект успешно найден!")
else:
    print("Объект не найден.")