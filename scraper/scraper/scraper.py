import hashlib
import os.path

import httpx
from bs4 import BeautifulSoup
import asyncio
from .connect import (
    extract_url,
    insert_url,
    insert_image,
    db_pool_connection,
    create_model,
)
import aiofiles
import logging


async def save_file(url, response, connection):
    try:
        hash_object = hashlib.sha1()
        hash_object.update(response.content)
        hash_digest = hash_object.hexdigest()
        name = f"{hash_digest}_{url.split('/')[-1]}"

        await insert_image(name, response.content, connection)
    except Exception as e:
        logging.info("Ошибка при сохранении файла ")


async def scrap_url(url, connection):
    async with httpx.AsyncClient(timeout=20) as client:
        if not await extract_url(url, connection):
            try:
                response = await client.get(url)

                if response.status_code == 200:
                    soap = BeautifulSoup(response.text, "html.parser")

                    div_element = soap.find(
                        "div", class_="product-preview-carousel__wrapper"
                    )

                    if div_element:
                        links = div_element.find_all("a")

                        for link in links:
                            href = link.get("href")

                            response_href = await client.get(href)
                            if response_href.status_code == 200:
                                await save_file(
                                    href,
                                    response_href,
                                    connection,
                                )

                await insert_url(url, connection)
            except Exception as e:
                logging.info("Ошибка при получении данных из URL:")
                logging.info("Приостановка работы...")
                await asyncio.sleep(10)
                logging.info("Работа восстановлена!")


async def scrap():
    logging.info("Scrapper начал работу")
    directory = os.path.dirname(__file__)
    path_file_links = os.path.join(directory, "part_links.txt")
    async with aiofiles.open(path_file_links, "r") as file:
        urls = [line.rstrip("\n") for line in await file.readlines()]

        pool = await db_pool_connection()
        await create_model(pool)

        tasks = [scrap_url(url, pool) for url in urls]
        await asyncio.gather(*tasks)
        await pool.close()
        logging.info("Scrapper успешно завершил свою работу")
