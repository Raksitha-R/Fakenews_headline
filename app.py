from flask import Flask, request, render_template_string
import threading
from utils.dependencies import load_dependencies
from utils.news_checker import find_similar_news, predict_news, verify_with_gemini
from utils.news_crawler import schedule_updates
from utils.templates import HTML_TEMPLATE

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, result=None)

@app.route("/check", methods=["POST"])
def check_news():
    news = request.form["news"]
    similar_news = find_similar_news(news)

    if similar_news:
        gemini_response = verify_with_gemini(news, similar_news)
        if gemini_response.lower() == "no":
            result = f"Fake (Gemini detected a mismatch with: '{similar_news[:100]}...')"
        else:
            result = f"Real (Matched with: '{similar_news[:100]}...')"
    else:
        predicted = predict_news(news)
        result = f"{predicted} (No similar news found)"

    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == "__main__":
    load_dependencies() #model load,tokenizer indexmap
    threading.Thread(target=schedule_updates, daemon=True).start()   #crawler loop
    app.run(debug=True, port=5000)
