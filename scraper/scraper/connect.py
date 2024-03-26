import os.path

import aiofiles
import asyncpg
import logging


async def db_connection():
    return await asyncpg.connect(
        database="isit_db",
        user="postgres",
        password="Kirill2305",
        # host="localhost",
        host="database",
        port="5432",
    )


async def db_pool_connection():
    return await asyncpg.create_pool(
        database="isit_db",
        user="postgres",
        password="Kirill2305",
        # host="localhost",
        host="database",
        port="5432",
    )


async def count_images():
    connection = await db_connection()
    return await connection.fetchval("SELECT count(*) FROM images")


async def count_visited_urls():
    connection = await db_connection()
    return await connection.fetchval("SELECT count(*) FROM visited_urls")


async def create_visited_urls(connection):
    await connection.execute(
        """
        CREATE TABLE IF NOT EXISTS visited_urls (
            id SERIAL PRIMARY KEY,
            url VARCHAR (255)
        )
        """
    )


async def create_images(connection):
    await connection.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id SERIAL PRIMARY KEY,
            name VARCHAR (255),
            data BYTEA
        )
        """
    )


async def create_model(connection):
    await create_visited_urls(connection)
    await create_images(connection)


async def insert_url(url, connection):
    try:
        await connection.execute("INSERT INTO visited_urls (url) VALUES ($1)", url)
    except Exception as e:
        logging.info("Ошибка при сохранении посещенной страницы ")


async def extract_url(url, connection):
    try:
        result = await connection.fetch(
            "SELECT * FROM visited_urls WHERE url = $1", url
        )
        return bool(result)
    except Exception as e:
        logging.info("Ошибка при проверки страницы на посещенность ")


async def insert_image(name, data, connection):
    try:
        await connection.execute(
            "INSERT INTO images (name, data) VALUES ($1, $2)", name, data
        )
    except Exception as e:
        logging.info("Ошибка при сохранении изображения ")


async def extract_all_images(directory, connection):
    try:
        images = await connection.fetch("SELECT name, data FROM images")

        for i, (name, data) in enumerate(images):
            path = os.path.join(directory, name)
            with open(path, "wb") as file:
                file.write(data)

    except Exception as e:
        logging.info("Ошибка при сохранении изображений в папку ", e)
