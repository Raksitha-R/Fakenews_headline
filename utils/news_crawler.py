import time
import pandas as pd
import logging

from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from utils.news_checker import news_data

logger = logging.getLogger(__name__)

WEB_CONFIG = {
    "BBC": {
        "url": "https://www.bbc.com/news",
        "content_selector": 'h2[data-testid="card-headline"], p[data-testid="card-description"]',
        "publisher": "BBC"
    },
    "CNN": {
        "url": "https://edition.cnn.com/world",
        "content_selector": 'span.container__headline-text, div.l-container p',
        "publisher": "CNN"
    },
    "The Hindu": {
        "url": "https://www.thehindu.com/",
        "content_selector": 'strong, a.cx-item.cx-main',
        "publisher": "The Hindu"
    },
    "Google News": {
        "url": "https://news.google.com/topstories",
        "content_selector": 'a.gPFEn',
        "publisher": "Google News"
    }
}

def get_edge_driver():
    options = EdgeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--log-level=3")
    options.add_argument("--disable-logging")

    service = EdgeService(
        executable_path=r"drivers/msedgedriver.exe",
        log_path="NUL"
    )

    return webdriver.Edge(service=service, options=options)

def crawl_news(site):
    global news_data
    config = WEB_CONFIG[site]
    driver = get_edge_driver()

    try:
        logger.info(f"Starting crawl for {site}")

        driver.get(config["url"])
        time.sleep(2)

        WebDriverWait(driver, 30).until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, config["content_selector"])
            )
        )

        elements = driver.find_elements(By.CSS_SELECTOR, config["content_selector"])

        data = []
        for idx, element in enumerate(elements):
            text = element.text.strip()
            if text:
                data.append({
                    "S.No": idx + 1,
                    "Content": text,
                    "Publisher": config["publisher"]
                })

        news_data[site] = pd.DataFrame(data)

        logger.info(f"{site}: {len(data)} articles")

    except Exception:
        logger.info(f"{site}: crawl failed")

    finally:
        driver.quit()

def schedule_updates():
    while True:
        for site in WEB_CONFIG:
            crawl_news(site)
        logger.info("Crawling completed\n")
        time.sleep(24 * 3600)
