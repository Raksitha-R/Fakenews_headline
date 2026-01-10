import os

# ---- SUPPRESS WARNINGS ----
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"

import logging
from flask import Flask, request, render_template_string
import threading
from utils.dependencies import load_dependencies
from utils.news_checker import find_similar_news, predict_news, verify_with_gemini
from utils.news_crawler import schedule_updates
from utils.templates import HTML_TEMPLATE

# ---- LOGGING SETUP ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    logger.info("Home page loaded")
    return render_template_string(HTML_TEMPLATE, result=None)

@app.route("/check", methods=["POST"])
def check_news():
    news = request.form["news"]
    logger.info("Received news for checking")

    similar_news = find_similar_news(news)

    if similar_news:
        logger.info("Similar news found, verifying with Gemini")
        gemini_response = verify_with_gemini(news, similar_news)

        if gemini_response.lower() == "no":
            result = f"Fake (Gemini detected a mismatch with: '{similar_news}...')"
            logger.info("Result: FAKE (Gemini mismatch)")
        else:
            result = f"Real (Matched with: '{similar_news}...')"
            logger.info("Result: REAL (Gemini match)")
    else:
        logger.info("No similar news found, using ML model")
        predicted = predict_news(news)
        result = f"{predicted} (No similar news found)"
        logger.info(f"Model prediction: {predicted}")

    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == "__main__":
    load_dependencies()

    print("\nOPEN THIS IN BROWSER 👉 http://127.0.0.1:5000\n")

    threading.Thread(
        target=schedule_updates,
        daemon=True
    ).start()

    app.run(host="0.0.0.0", port=5000)
