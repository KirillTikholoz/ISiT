import cv2
import os
from utils_cv import detect_object

images_folder = "../images"
image_filename = "31a8337711ed27c1e46b183a1d90392aebb5e5f6_c4fd34695d36dbc4c44f3b7a3180c9f40aa56a3e.jpg"
image_path = os.path.join(images_folder, image_filename)

image = cv2.imread(image_path)

mask = detect_object(image)
result = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow('Segmented Objects', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

