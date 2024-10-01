import joblib
import re
import string
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

nltk.download('stopwords')

stopwd = stopwords.words('english')
vectorizer = CountVectorizer(stop_words='english', lowercase=True)

def clean_text(text):
    text = text.lower()
    text = re.sub('-', ' ', text.lower())
    text = re.sub(r'http\S+', '', text)
    text = re.sub(f'[{string.punctuation}]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    words = nltk.tokenize.word_tokenize(text, language="english", preserve_line=True)
    text = " ".join([i for i in words if i not in stopwd and len(i) > 2])
    return text.strip()

# Assume you have trained the model using RandomForestClassifier
model = joblib.load('rf.joblib')

# Assume you have a LabelEncoder
le = joblib.load('le.joblib')

def get_input(text):
    cleaned_input = clean_text(text)
    
    # Load the vectorizer used during training
    vectorizer_fit = joblib.load('vectorizer.joblib')
    
    # Transform the input using the same vectorizer
    input_vectorized = vectorizer_fit.transform([cleaned_input])
    
    # Ensure the feature dimensions match
    if input_vectorized.shape[1] != vectorizer_fit.transform(['']).shape[1]:
        raise ValueError("Feature dimensions mismatch. Ensure consistent vectorizer.")
    
    # Use the fitted model for prediction
    prediction = model.predict(input_vectorized)
    
    # Inverse transform the prediction using LabelEncoder
    # print(cleaned_input)
    return le.inverse_transform(prediction)
