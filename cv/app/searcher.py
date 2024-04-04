import os
import cv2
import numpy as np
from .model import init_db, Image


def image_compare(query_image, train_image):
    orb = cv2.ORB_create()

    keypoints_query, descriptors_query = orb.detectAndCompute(query_image, None)
    keypoints_train, descriptors_train = orb.detectAndCompute(train_image, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(descriptors_query, descriptors_train)
    matches = sorted(matches, key=lambda x: x.distance)
    matching_result = cv2.drawMatches(
        query_image,
        keypoints_query,
        train_image,
        keypoints_train,
        matches[:10],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    num_matches = len(matches)
    mean_distance = np.mean([match.distance for match in matches])

    # Визуальная оценка
    matching_result = cv2.drawMatches(
        query_image,
        keypoints_query,
        train_image,
        keypoints_train,
        matches[:10],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.putText(
        matching_result,
        f"Matches: {num_matches}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        matching_result,
        f"Mean Distance: {mean_distance:.2f}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    # cv2.imshow("Matching result", matching_result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return num_matches, mean_distance, matching_result


def image_search(image_filename, combined_images_filename):
    combined_images_folder = os.path.join(
        os.path.dirname(__file__), "../images/combined_images"
    )
    combined_images_path = os.path.join(
        combined_images_folder, combined_images_filename
    )

    session = init_db()
    image = session.query(Image).filter_by(name=image_filename).first()

    image_data = image.data
    image_array = np.frombuffer(image_data, dtype=np.uint8)

    query_image = cv2.imread(combined_images_path, cv2.IMREAD_GRAYSCALE)
    train_image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    num_matches, mean_distance, matching_result = image_compare(
        query_image, train_image
    )

    _, image_encoded = cv2.imencode(".jpg", matching_result)
    result_data = image_encoded.tobytes()

    # Проверка условий успешного обнаружения объекта
    if num_matches > 50 and mean_distance < 50:
        return True, result_data
    else:
        return False, result_data
