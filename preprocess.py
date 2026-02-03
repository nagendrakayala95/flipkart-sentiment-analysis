import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Clean and preprocess text by converting to lowercase, removing special characters,
    filtering stopwords, and lemmatizing.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    try:
        # Split by whitespace
        words = text.split()
        
        # Lemmatize, filter stopwords and short words
        words = [lemmatizer.lemmatize(word) 
                 for word in words 
                 if word not in stop_words and len(word) > 2]
    except Exception as e:
        print(f"Error processing text: {e}")
        return ""
    
    return ' '.join(words) if words else ""
