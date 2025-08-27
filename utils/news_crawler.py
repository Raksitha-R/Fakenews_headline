import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils.news_checker import news_data

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

def crawl_news(website):
    global news_data
    config = WEB_CONFIG[website]
    edge_driver_path = r"drivers/msedgedriver.exe"
    options = EdgeOptions()
    options.add_argument("--headless")
    service = EdgeService(executable_path=edge_driver_path)
    driver = webdriver.Edge(service=service, options=options)
    try:
        driver.get(config["url"])
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        WebDriverWait(driver, 30).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, config["content_selector"]))
        )
        elements = driver.find_elements(By.CSS_SELECTOR, config["content_selector"])
        data = []
        for idx, element in enumerate(elements):
            content = element.text.strip()
            if content:
                data.append({
                    "S.No": idx + 1,
                    "Content": content,
                    "Publisher": config["publisher"],
                    "Date": time.strftime("%Y-%m-%d")
                })
        news_data[website] = pd.DataFrame(data)
    except Exception as e:
        print(f"Error while crawling {website}: {e}")
    finally:
        driver.quit()

def schedule_updates():
    while True:
        for site in WEB_CONFIG.keys():
            print(f"Starting crawl for {site}...")
            crawl_news(site)
        time.sleep(24 * 3600)
