import cv2
import numpy as np
import os
from utils_cv import detect_object

images_folder = "../images"
background_image_filename = "yellow_background.jpg"
object_image_filenames = ["25af62b32fe771ace65a5b820fe0f34875dc9fb6_f268da5da8e53e2f7ab4bc57521e8bdb7a1cb8e2.jpg",
                          "b9fc128d293251c0e0e6fc17fbc3e56c706ca9fa_48651d8ed76143cd389b2b94559eddaa2b80df38.jpg"]

background_image_path = os.path.join(images_folder, background_image_filename)
object_image_paths = [os.path.join(images_folder, filename) for filename in object_image_filenames]
background_image = cv2.imread(background_image_path)

# Пирамида фона
background_pyramid = [background_image.copy()]
for i in range(2):  # Количество уровней пирамиды
    background_image = cv2.pyrDown(background_image)
    background_pyramid.append(background_image.copy())


object_pyramids = []
object_positions = []

for object_image_path in object_image_paths:
    object_image = cv2.imread(object_image_path)
    object_pyramid = [object_image.copy()]
    for i in range(2):  # Количество уровней пирамиды
        object_image = cv2.pyrDown(object_image)
        object_pyramid.append(object_image.copy())
    object_pyramids.append(object_pyramid)

    # Определение случайных координат вставки объекта
    object_x = np.random.randint(0, background_pyramid[-1].shape[1] - object_pyramid[-1].shape[1])
    object_y = np.random.randint(0, background_pyramid[-1].shape[0] - object_pyramid[-1].shape[0])
    object_positions.append((object_x, object_y))


# Пирамидальное сопоставление и блендинг
for i in range(len(background_pyramid)):
    for j, object_pyramid in enumerate(object_pyramids):
        object_level = object_pyramid[i]
        object_x, object_y = object_positions[j]

        background_level = background_pyramid[i]

        mask = detect_object(object_level)
        object_cropped = cv2.bitwise_and(object_level, object_level, mask=mask)

        roi = background_level[object_y:object_y+object_level.shape[0], object_x:object_x+object_level.shape[1]]
        background_roi_with_mask = cv2.bitwise_and(roi, cv2.merge((255 - mask, 255 - mask, 255 - mask)))

        blended_object = cv2.add(background_roi_with_mask, object_cropped)
        background_level[object_y:object_y+object_level.shape[0], object_x:object_x+object_level.shape[1]] = blended_object


# последний уровень пирамиды
cv2.imshow('Blended Image', background_pyramid[-1])
cv2.waitKey(0)
cv2.destroyAllWindows()


for i in range(len(background_pyramid) - 1, 0, -1):
    background_pyramid[i - 1] = cv2.pyrUp(background_pyramid[i])

cv2.imshow('Blended Image', background_pyramid[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

comb_images_folder = "../combined_images"
comb_image_filename = "comb_image.jpg"
path = os.path.join(comb_images_folder, comb_image_filename)

# cv2.imwrite(path, background_pyramid[0])