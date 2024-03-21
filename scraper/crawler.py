import gzip
import httpx
import xml.etree.ElementTree as ET
import aiofiles
import logging


async def process_xml_file(url):
    async with httpx.AsyncClient() as client:
        if url.endswith(".xml"):
            response = await client.get(url)
            if response.status_code == 200:
                return ET.fromstring(response.content)
        elif url.endswith(".xml.gz"):
            response = await client.get(url)
            if response.status_code == 200:
                xml_data = gzip.decompress(response.content)
                return ET.fromstring(xml_data)
        else:
            return None


async def extract_sitemap_links(sitemap_index_url):
    async with httpx.AsyncClient() as client:
        response = await client.get(sitemap_index_url)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            sitemap_links = [
                loc.text
                for loc in root.findall(
                    ".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
                )
            ]
            return sitemap_links


async def extract_product_links(sitemap_url):
    root = await process_xml_file(sitemap_url)
    product_links = [
        url.text
        for url in root.findall(
            ".//{http://www.sitemaps.org/schemas/sitemap/0.9}url/{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
        )
    ]
    return product_links


async def crawl_product_links(sitemap_index_url):
    try:
        logging.info("Crawler начал работу")
        sitemap = await extract_sitemap_links(sitemap_index_url)
        cnt = 0

        for sitemap_link in sitemap:
            product_links = await extract_product_links(sitemap_link)

            async with aiofiles.open("all_links.txt", "w") as file:
                for product_link in product_links:
                    await file.write(product_link + "\n")
                    cnt = cnt + 1

        logging.info(f"Количество ссылок: {cnt}")
        logging.info("Crawler закончил работу")
    except Exception as e:
        logging.info("Ошибка при работе crawler'а:")
