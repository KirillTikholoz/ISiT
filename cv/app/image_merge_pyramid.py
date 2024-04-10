import cv2
import numpy as np
from .model import Image, init_db


def merge_images(image_name1, image_name2):
    session = init_db()
    image_record1 = session.query(Image).filter_by(name=image_name1).first()
    image_record2 = session.query(Image).filter_by(name=image_name2).first()

    image_data1 = image_record1.data
    image_data2 = image_record2.data

    image_array1 = np.frombuffer(image_data1, dtype=np.uint8)
    image_array2 = np.frombuffer(image_data2, dtype=np.uint8)

    A = cv2.imdecode(image_array1, cv2.IMREAD_COLOR)
    B = cv2.imdecode(image_array2, cv2.IMREAD_COLOR)

    A = cv2.resize(A, (512, 512), interpolation=cv2.INTER_CUBIC)
    B = cv2.resize(B, (512, 512), interpolation=cv2.INTER_CUBIC)

    G = A.copy()
    gpA = [G]
    for i in range(7):
        G = cv2.pyrDown(G)
        gpA.append(G)

    G = B.copy()
    gpB = [G]
    for i in range(7):
        G = cv2.pyrDown(G)
        gpB.append(G)

    LR = []
    for la, lb in zip(gpA, gpB):
        rows, cols, dpt = la.shape
        lr = np.hstack((la[:, 0 : cols // 2], lb[:, cols // 2 :]))
        LR.append(lr)

    lr = LR[7]
    for i in range(6, -1, -1):
        lr = cv2.pyrUp(lr)
        lr = cv2.add(lr, LR[i])

    lpA = [gpA[7]]

    for i in range(7, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)

    lpB = [gpB[7]]
    for i in range(7, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)

    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, 0 : cols // 2], lb[:, cols // 2 :]))
        LS.append(ls)

    ls = LS[0]
    for i in range(1, 8):
        ls = cv2.pyrUp(ls)
        ls = cv2.add(ls, LS[i])

    with_pyramid = lr + ls

    _, image_encoded = cv2.imencode(".jpg", with_pyramid)
    result_data = image_encoded.tobytes()
    return result_data
