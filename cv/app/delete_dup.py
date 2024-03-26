import hashlib
from .model import Image, init_db
import logging


def delete_duplicates():
    session = init_db()

    images = session.query(Image).all()
    image_hashes = {}

    for image in images:
        image_data = image.data

        image_hash = hashlib.md5(image.data).hexdigest()
        if not image_data:
            session.delete(image)
            session.commit()
            logging.info(f"Удалено неправильное изображение: {image.name}")
        elif image_hash in image_hashes:
            session.delete(image)
            session.commit()
            logging.info(f"Удален дубликат изображения: {image.name}")
        else:
            image_hashes[image_hash] = image.name

    logging.info("Процесс завершен.")

    session.close()
