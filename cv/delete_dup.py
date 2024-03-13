import os
import hashlib


def calculate_hash(image_path):
    with open(image_path, 'rb') as f:
        image_hash = hashlib.md5(f.read()).hexdigest()
    return image_hash


images_folder = "../images"
image_hashes = {}

for filename in os.listdir(images_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(images_folder, filename)
        image_hash = calculate_hash(image_path)
        if image_hash in image_hashes:
            os.remove(image_path)
            print(f"Удалено дубликат: {filename}")
        else:
            image_hashes[image_hash] = image_path

print("Процесс завершен.")