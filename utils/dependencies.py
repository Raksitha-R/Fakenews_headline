import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

model = None
TOKENIZER = None
max_length = 100

def load_dependencies():
    # global model, TOKENIZER
    # model = load_model("models/fakenewslogr.h5")
    # data = pd.read_csv("models/news_dataset.csv")
    # if "Text" not in data.columns:
    #     raise ValueError("The dataset does not contain a 'Text' column.")
    
    # # selects the top 5k frequent words in Text Coloumn and map it to number indexes
    # TOKENIZER = Tokenizer(num_words=5000) 
    # TOKENIZER.fit_on_texts(data["Text"].values)

    # #Tokenizer.text_to_sequence(text) => gives the numeric op for the input text(based on the mapindex)



    global model, TOKENIZER
    print("Loading model...")
    model = load_model("models/fakenewslogr.h5")
    print("Model loaded successfully.")

    print("Loading dataset...")
    data = pd.read_csv("models/news_dataset.csv")
    print("Dataset loaded successfully. Columns:", data.columns.tolist())

    if "Text" not in data.columns:
        raise ValueError("The dataset does not contain a 'Text' column.")
    
    TOKENIZER = Tokenizer(num_words=5000) 
    TOKENIZER.fit_on_texts(data["Text"].values)
    print("Tokenizer fitted successfully.")
