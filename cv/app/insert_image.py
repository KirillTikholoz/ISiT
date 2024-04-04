import cv2
import numpy as np
import os
from .utils_cv import detect_object
from .model import init_db, Image


def insert_img(background_image_filename, object_image_filenames):
    combined_images_folder = os.path.join(
        os.path.dirname(__file__), "../images/combined_images"
    )

    background_image_path = os.path.join(
        combined_images_folder, background_image_filename
    )
    background_image = cv2.imread(background_image_path)

    session = init_db()
    images = [
        session.query(Image).filter_by(name=filename).first()
        for filename in object_image_filenames
    ]

    levels = 1
    background_pyramid, object_pyramids, object_positions = create_pyramid(
        background_image, images, levels
    )
    res = blending(background_pyramid, object_pyramids, object_positions)
    return extract_combined_image(res)


# Пирамида фона
def create_pyramid(background_image, images, levels):
    background_pyramid = [background_image.copy()]
    for i in range(levels):  # Количество уровней пирамиды
        background_image = cv2.pyrDown(background_image)
        background_pyramid.append(background_image.copy())

    object_pyramids = []
    object_positions = []

    for image in images:
        image_data = image.data
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        object_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        object_pyramid = [object_image.copy()]

        for i in range(levels):  # Количество уровней пирамиды
            object_image = cv2.pyrDown(object_image)
            object_pyramid.append(object_image.copy())
        object_pyramids.append(object_pyramid)

        # Определение случайных координат вставки объекта
        object_x = np.random.randint(
            0, background_pyramid[-1].shape[1] - object_pyramid[-1].shape[1]
        )
        object_y = np.random.randint(
            0, background_pyramid[-1].shape[0] - object_pyramid[-1].shape[0]
        )
        object_positions.append((object_x, object_y))

    return background_pyramid, object_pyramids, object_positions


# Пирамидальное сопоставление и блендинг
def blending(background_pyramid, object_pyramids, object_positions):
    for i in range(len(background_pyramid)):
        for j, object_pyramid in enumerate(object_pyramids):
            object_level = object_pyramid[i]
            object_x, object_y = object_positions[j]

            background_level = background_pyramid[i]

            mask = detect_object(object_level)
            object_cropped = cv2.bitwise_and(object_level, object_level, mask=mask)

            roi = background_level[
                object_y : object_y + object_level.shape[0],
                object_x : object_x + object_level.shape[1],
            ]
            background_roi_with_mask = cv2.bitwise_and(
                roi, cv2.merge((255 - mask, 255 - mask, 255 - mask))
            )

            blended_object = cv2.add(background_roi_with_mask, object_cropped)
            background_level[
                object_y : object_y + object_level.shape[0],
                object_x : object_x + object_level.shape[1],
            ] = blended_object

    return background_pyramid


def extract_combined_image(background_pyramid):
    # cv2.imshow('Blended Image', background_pyramid[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # cv2.imshow('Blended Image', background_pyramid[1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for i in range(len(background_pyramid) - 1, 0, -1):
        background_pyramid[i - 1] = cv2.pyrUp(background_pyramid[i])

    _, image_encoded = cv2.imencode(".jpg", background_pyramid[0])
    result_data = image_encoded.tobytes()
    return result_data
