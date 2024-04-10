from fastapi import FastAPI
from .crawler import crawl_product_links
from .scraper import scrap
from .connect import count_images, count_visited_urls, extract_all_images, db_connection
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/crawler")
async def root():
    await crawl_product_links("https://poizon.io/sitemap.xml")
    return {"message": "Crawler завершил работу"}


@app.get("/scraper")
async def root():
    await scrap()
    return {"message": "Scraper завершил работу"}


@app.get("/count_images")
async def root():
    res = await count_images()
    return {"message": res}


@app.get("/count_visited_urls")
async def root():
    res = await count_visited_urls()
    return {"message": res}


@app.get("/extract_all_images")
async def root():
    conn = await db_connection()
    dir = "/images"
    await extract_all_images(dir, conn)
    return {"message": "Все изображения загружены в папку images"}
