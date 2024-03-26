import os

import tensorflow as tf


def create_model(train_dir, validation_dir):
    resnet = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )

    for layer in resnet.layers[:-4]:
        layer.trainable = False

    num_classes = 5
    x = tf.keras.layers.GlobalAveragePooling2D()(resnet.output)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=resnet.input, outputs=output)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255
    )

    batch_size = 32
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
    )

    num_epochs = 30
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
    )

    path = os.path.join(os.path.dirname(__file__), "../dataRepository/my_model3.h5")
    model.save(path)


train_directory = os.path.join(os.path.dirname(__file__), "../images/train_images")
validation_directory = os.path.join(
    os.path.dirname(__file__), "../images/validation_images"
)
create_model(train_directory, validation_directory)
