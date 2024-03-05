from fastapi import FastAPI
from .crawler import crawl_product_links
from .scraper import scrap
from .connect import count_images, count_visited_urls
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


@app.get("/images")
async def root():
    res = await count_images()
    return {"message": res}


@app.get("/visited_urls")
async def root():
    res = await count_visited_urls()
    return {"message": res}
