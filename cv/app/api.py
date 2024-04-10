import os

from fastapi import FastAPI, HTTPException, Response
from .find_obj import find_object
from .delete_dup import delete_duplicates
import logging
import base64
from .hsv import search_similar_images_hsv, calculate_distances_hsv
from .lab import search_similar_images_lab, calculate_distances_lab
from .insert_image import insert_img
from .searcher import image_search
from .searcher_text import searcher_image_text, create_embeddings
from .prediction import predict
from .utils_cv import extract_all_image_name
from .image_merge_pyramid import merge_images
import io

app = FastAPI()
logging.basicConfig(level=logging.INFO)


@app.get("/")
async def root():
    return {"message": "Hello Cv"}


@app.get("/find_obj")
async def find_obj(image_filename):
    # ?image_filename=d914262c6830f8713590dfad226e92cf78fb9049_7e25e2042c47d38038f6f7d84291a20b97bcb48c.jpg
    # ?image_filename=1ac437dbcd873542e07f8005a637ad8f54a1583f_699efa89d1772ed0547c0de3be39caf8b2faca19.jpg
    image_data = find_object(image_filename)

    image_base64 = base64.b64encode(image_data).decode("utf-8")
    html_image = f'<img src="data:image/jpeg;base64,{image_base64}" alt="Image">'
    return Response(content=html_image, media_type="text/html")


@app.get("/delete_dup")
async def delete_dup():
    delete_duplicates()
    return {"message": "Дубликаты удалены"}


@app.get("/merge")
async def merge(image_filename1, image_filename2):
    # ?image_filename1=0c3e7aaea822bbd968eebf9397cd1105ef4e42ce_512e0631f69ea3bd3e64fd4a46e03c2c4c1a056e.jpg
    # &image_filename2=01d8dd24eda298c330d55f93dcbdaa4dbc27c9de_f2ed0ce905ba714b9b153b854bffb2acda238446.jpg
    image_data = merge_images(image_filename1, image_filename2)

    image_base64 = base64.b64encode(image_data).decode("utf-8")
    html_image = f'<img src="data:image/jpeg;base64,{image_base64}" alt="Image">'
    return Response(content=html_image, media_type="text/html")


@app.get("/hsv")
async def search_similar_hsv(image_filename):
    # ?image_filename=d914262c6830f8713590dfad226e92cf78fb9049_7e25e2042c47d38038f6f7d84291a20b97bcb48c.jpg
    # ?image_filename=1fab47e3a4ff84b1c49d6725f95d805cebecbfff_93c9fa5677cc581b247281620c24e45b42151fca.jpg
    images = search_similar_images_hsv(image_filename)

    html_images = ""
    for image in images:
        image_data = image.data
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        html_image = f'<img src="data:image/jpeg;base64,{image_base64}" alt="Image">'
        html_images += html_image
    return Response(content=html_images, media_type="text/html")


@app.get("/lab")
async def search_similar_lab(image_filename):
    # ?image_filename=d914262c6830f8713590dfad226e92cf78fb9049_7e25e2042c47d38038f6f7d84291a20b97bcb48c.jpg
    # ?image_filename=3e334082d5daa987aa5796914a8411160c4d45b2_9a807e611d04220521b582d31dad2f0a74188cab.jpg
    images = search_similar_images_lab(image_filename)

    html_images = ""
    for image in images:
        image_data = image.data
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        html_image = f'<img src="data:image/jpeg;base64,{image_base64}" alt="Image">'
        html_images += html_image
    return Response(content=html_images, media_type="text/html")


@app.get("/image_search_text")
async def search_text(query):
    # ?query=nike
    # ?query=paris
    # ?query=23
    create_embeddings()
    image_data = searcher_image_text(query)

    img_byte_array = io.BytesIO()
    image_data.save(img_byte_array, format="JPEG")
    img_byte_array = img_byte_array.getvalue()

    image_base64 = base64.b64encode(img_byte_array).decode("utf-8")
    html_image = f'<img src="data:image/jpeg;base64,{image_base64}" alt="Image">'

    return Response(content=html_image, media_type="text/html")


@app.get("/predict")
async def class_predict(image_filename):
    # ?image_name=c0d5acfe95343c3768cae93df2f94e41b10fc621_d110e4328e194ce72236c37690f96d9f38a27d95.jpg
    # ?image_name=0c57f832e141958174193ba3fb8ecdfe1cf383bc_464243f6f53c4e020829b24df6b3c204932f68c4.jpg
    result = predict(image_filename)

    return result


@app.get("/all_names")
async def all_names():
    result = extract_all_image_name()

    return {"Все имена изображений": result}
