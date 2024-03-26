import io
import os

import tensorflow as tf
import keras
from PIL import Image as PILImage
import numpy as np
from .model import Image, init_db


def predict(image_name):
    model_save_path = os.path.join(
        os.path.dirname(__file__), "../dataRepository/my_model2.h5"
    )
    loaded_model = keras.models.load_model(model_save_path)

    loaded_model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    session = init_db()
    image_record = session.query(Image).filter_by(name=image_name).first()

    image_data = image_record.data
    image = PILImage.open(io.BytesIO(image_data))
    image = image.resize((224, 224))
    image_array = np.array(image)

    # Нормализация данных
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = loaded_model.predict(image_array)

    predictions_in_percentage = np.array(
        [format(prob * 100, ".2f") for prob in predictions[0]]
    )
    classes_with_percent = {
        "T-shirts": predictions_in_percentage[0],
        "bags": predictions_in_percentage[1],
        "pants": predictions_in_percentage[2],
        "sneakers": predictions_in_percentage[3],
        "sweatshirts": predictions_in_percentage[4],
    }

    return classes_with_percent
