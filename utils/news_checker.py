import os
import nltk
import string
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
# from utils.dependencies import TOKENIZER, model, max_length
import utils.dependencies as deps

# Load environment variables from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

nltk.download('punkt')  #tokenizer
nltk.download('stopwords')

news_data = {}
similarity_threshold = 0.7   #how strict tfidf match should be

# Configure Gemini API for gemini calls
genai.configure(api_key=GEMINI_API_KEY)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    # sequence = TOKENIZER.texts_to_sequences([" ".join(filtered_tokens)])   #get numeric array based on the index map formed using datatset
    # return pad_sequences(sequence, maxlen=max_length)
    sequence = deps.TOKENIZER.texts_to_sequences([" ".join(filtered_tokens)])
    return pad_sequences(sequence, maxlen=deps.max_length)


def predict_news(text):
    processed_text = preprocess_text(text)
    # prediction = model.predict(processed_text)
    prediction = deps.model.predict(processed_text)
    class_label = prediction.argmax(axis=1)[0]
    return "Real" if class_label == 1 else "Fake"

def find_similar_news(input_text):
    global news_data
    all_news = []
    for site, df in news_data.items():
        all_news.extend(df["Content"].values)
    if not all_news:
        return None
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_news + [input_text])
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()  # -1 last input :-1 all expect last, flatten-convert into 1d array
    if similarities.max() >= similarity_threshold:
        return all_news[similarities.argmax()]
    return None

def verify_with_gemini(news_input, matched_news):
    prompt = f"""
    Compare the following two news headlines and tell me if they mean the same thing. Reply with only 'Yes' or 'No'.
    1. {news_input}
    2. {matched_news}
    """
    response = genai.GenerativeModel('gemini-2.0-flash').generate_content(prompt)
    return response.text.strip()
