import os
import cv2
import pytesseract

path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = path_to_tesseract

images_folder = "../images"
image_files = os.listdir(images_folder)
images = []
for image_file in image_files:
    image_path = os.path.join(images_folder, image_file)
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images.append((image, pytesseract.image_to_string(gray_image, lang="eng").lower()))


while True:
    text = input("Введите текст: ")
    for elem in images:
        if elem[1].find(text.lower()) != -1:
            print(elem[1])
            cv2.imshow("searched image", elem[0])
            cv2.waitKey(0)