import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- 1. SETUP & NLTK DATA ---
@st.cache_resource
def download_nltk():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

download_nltk()

# --- 2. DATASET (Problem: Cyberbullying) ---
@st.cache_data
    
def load_data():
    # Load dataset from a local CSV file provided by the user
    # Expected columns: tweet_text, cyberbullying_type
    path = r"C:\cyberbullyingapp\cyberbullying_tweets.csv"
    df = pd.read_csv(path)
    # map the text & label columns to a common format
    df = df.rename(columns={"tweet_text": "tweet"})
    # anything other than 'not_cyberbullying' is considered bullying
    df['label'] = df['cyberbullying_type'].apply(
        lambda x: 0 if str(x).lower() == 'not_cyberbullying' else 1
    )
    # sample to keep training fast (previous version used 1000 rows)
    return df[['tweet', 'label']].sample(1000, random_state=42)

# --- 3. PREPROCESSING ---
def preprocess_text(text):
    text = str(text).lower() # Lowercasing
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ", text) # Tokenization/Cleaning
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # Stopword removal and Lemmatization
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# --- 4. MODEL IMPLEMENTATION (Option A) ---
df = load_data()
df['clean_text'] = df['tweet'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=1500)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)

# --- 5. STREAMLIT INTERFACE ---
st.title("🛡️ Cyberbullying Detection App")
st.write("Enter text below to check if it contains harmful or bullying language.")

user_input = st.text_area("Input Text:", placeholder="Type a comment here...")

if st.button("Analyze"):
    if user_input:
        processed = preprocess_text(user_input)
        vec = vectorizer.transform([processed])
        prediction = model.predict(vec)
        
        if prediction[0] == 1:
            st.error("🚨 Result: Potential Cyberbullying Detected")
        else:
            st.success("✅ Result: Clean / Safe Content")
    else:
        st.warning("Please enter text first.")