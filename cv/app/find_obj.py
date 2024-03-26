import cv2
from .utils_cv import detect_object
import numpy as np
from .model import Image, init_db
import logging


def find_object(image_name):
    session = init_db()

    image_record = session.query(Image).filter_by(name=image_name).first()
    if image_record:
        image_data = image_record.data
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image_cv2 = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image_cv2 is not None:
            mask = detect_object(image_cv2)
            result = cv2.bitwise_and(image_cv2, image_cv2, mask=mask)

            _, image_encoded = cv2.imencode(".jpg", result)
            result_data = image_encoded.tobytes()
            return result_data
        else:
            logging.info("Не удалось прочитать изображение с помощью cv2")
    else:
        logging.info("Изображение с таким именем не найдено в базе данных")

    session.close()


# image_name = 'd914262c6830f8713590dfad226e92cf78fb9049_7e25e2042c47d38038f6f7d84291a20b97bcb48c.jpg'
# find_object(image_name)
