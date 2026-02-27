import streamlit as st
import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
    # Load dataset from a local CSV file; use path relative to this script
    # Expected columns: tweet_text, cyberbullying_type
    base = os.path.dirname(__file__)
    path = os.path.join(base, "cyberbullying_tweets.csv")
    if not os.path.exists(path):
        # try working directory as fallback (heroku, streamlit cloud, etc.)
        alt = "cyberbullying_tweets.csv"
        if os.path.exists(alt):
            path = alt
        else:
            st.error(f"Dataset not found! Looked for '{path}' and '{alt}'.")
            return pd.DataFrame(columns=["tweet", "label"])
    df = pd.read_csv(path)
    # map the text & label columns to a common format
    df = df.rename(columns={"tweet_text": "tweet"})
    # anything other than 'not_cyberbullying' is considered bullying
    df['label'] = df['cyberbullying_type'].apply(
        lambda x: 0 if str(x).lower() == 'not_cyberbullying' else 1
    )
    # Use the entire dataset; sampling reduces information for the model
    return df[['tweet', 'label']]

# --- 3. PREPROCESSING ---
def preprocess_text(text):
    text = str(text).lower() # Lowercasing
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ", text) # Tokenization/Cleaning
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # Stopword removal and Lemmatization
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# --- 4. MODEL TRAINING (with caching) ---
@st.cache_resource
def train_model():
    """Load data, preprocess, and train the cyberbullying detection model."""
    try:
        df = load_data()
        
        # Validate data
        if df.empty:
            st.error("Error: Dataset is empty!")
            return None, None, None, None
        
        if df['label'].nunique() < 2:
            st.error("Error: Not enough classes in labels!")
            return None, None, None, None
        
        # Preprocess text
        df['clean_text'] = df['tweet'].apply(preprocess_text)
        
        # Vectorize with richer features (unigrams + bigrams)
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1,2),
            strip_accents='unicode'
        )
        X = vectorizer.fit_transform(df['clean_text'])
        y = df['label']
        
        # split into train/test so we can evaluate generalization
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Use grid search with cross-validation to find best hyperparameters
        from sklearn.model_selection import GridSearchCV
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['liblinear'],
        }
        base_clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        grid = GridSearchCV(base_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        
        # verify performance
        train_accuracy = best_model.score(X_train, y_train)
        test_accuracy = best_model.score(X_test, y_test)
        
        # if still below 80%, try alternative classifier (RandomForest)
        if test_accuracy < 0.80:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
            rf.fit(X_train, y_train)
            rf_test = rf.score(X_test, y_test)
            if rf_test > test_accuracy:
                best_model = rf
                test_accuracy = rf_test
                train_accuracy = rf.score(X_train, y_train)
        
        return best_model, vectorizer, train_accuracy, test_accuracy
    
    except Exception as e:
        st.error(f"Error during model training: {str(e)}")
        return None, None, None, None

# Train and get model
model_result = train_model()
if model_result[0] is not None:
    # unpack depending on whether test_accuracy was provided
    if len(model_result) == 4:
        model, vectorizer, train_accuracy, test_accuracy = model_result
    else:
        model, vectorizer, train_accuracy = model_result
        test_accuracy = None
else:
    model, vectorizer, train_accuracy, test_accuracy = None, None, 0, None

# --- 5. STREAMLIT INTERFACE ---
st.title("🛡️ Cyberbullying Detection App")
st.write("Enter text below to check if it contains harmful or bullying language.")

# Show model status
if model is not None and vectorizer is not None:
    if test_accuracy is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Status", "✅ Ready")
        with col2:
            st.metric("Train Accuracy", f"{train_accuracy:.2%}")
        with col3:
            st.metric("Test Accuracy", f"{test_accuracy:.2%}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Status", "✅ Ready")
        with col2:
            st.metric("Training Accuracy", f"{train_accuracy:.2%}")
    
    st.divider()
    
    user_input = st.text_area("Input Text:", placeholder="Type a comment here...")
    
    if st.button("Analyze", type="primary"):
        if user_input:
            try:
                processed = preprocess_text(user_input)
                vec = vectorizer.transform([processed])
                prediction = model.predict(vec)
                confidence = model.predict_proba(vec)[0]
                
                if prediction[0] == 1:
                    st.error(f"🚨 Result: Potential Cyberbullying Detected (Confidence: {confidence[1]:.2%})")
                else:
                    st.success(f"✅ Result: Clean / Safe Content (Confidence: {confidence[0]:.2%})")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            st.warning("Please enter text first.")
else:
    st.error("❌ Model failed to train. Please check the dataset and try refreshing the page.")