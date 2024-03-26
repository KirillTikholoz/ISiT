import os
from PIL import Image
import torch
from clip import clip
from .model import init_db, Image
from io import BytesIO
from PIL import Image as PILImage
import logging


def text_image_search(query_text: str, imgs_embeddings: torch.Tensor, model, device):
    query_embeddings = model.encode_text(clip.tokenize([query_text]).to(device))
    similarities = query_embeddings @ imgs_embeddings.T
    return similarities


def load_images_from_db():
    session = init_db()
    images_from_db = session.query(Image).all()

    images = []
    for img_db in images_from_db:
        img_data = img_db.data
        img_pil = PILImage.open(BytesIO(img_data))
        if img_pil.mode == "RGBA":
            img_pil = img_pil.convert("RGB")
        images.append(img_pil)
    return images


def calculate_embeddings_batch(images, model, preprocess, batch_size=32):
    num_images = len(images)
    embeddings = []

    cnt = 0
    for i in range(0, num_images, batch_size):
        batch_images = images[i : i + batch_size]
        batch_images_processed = [preprocess(image) for image in batch_images]

        with torch.no_grad():
            batch_embeddings = model.encode_image(torch.stack(batch_images_processed))

        embeddings.append(batch_embeddings)

        cnt += batch_size
        logging.info(f"Количество изображений для которых вычисленны вложения: {cnt}")

    embeddings = torch.cat(embeddings)
    return embeddings


def calculate_embeddings(images, model, preprocess):
    images_embeddings = calculate_embeddings_batch(images, model, preprocess)

    path = os.path.join(
        os.path.dirname(__file__), "../dataRepository/images_embeddings.pth"
    )
    torch.save(images_embeddings, path)
    logging.info("Вложения вычислены")


def create_embeddings():
    images = load_images_from_db()
    logging.info("Изображения загружены для создания вложений")

    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    calculate_embeddings(images, model, preprocess)


def searcher_image_text(query):
    images = load_images_from_db()
    logging.info("Изображения загружены для поиска по запросу")

    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    path = os.path.join(
        os.path.dirname(__file__), "../dataRepository/images_embeddings.pth"
    )
    images_embeddings = torch.load(path)

    sim = text_image_search(query, images_embeddings, model, device)
    sim_dict = dict(zip(range(len(sim[0])), sim[0]))
    sorted_sim = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
    top_index = sorted_sim[0][0]
    logging.info("Поиск завершился")

    return images[top_index]
