import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

print("1. Loading Data...")
# Read the CSV file
try:
    df = pd.read_csv('reviews.csv')
except FileNotFoundError:
    print("Error: 'reviews.csv' not found.")
    exit()

print("2. Preprocessing Data...")


df.dropna(subset=['Review text', 'Ratings'], inplace=True)


df = df[df['Ratings'] != 3]
df['Sentiment'] = df['Ratings'].apply(lambda x: 1 if x > 3 else 0)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text)) # Remove special chars
    text = text.lower() # Lowercase
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df['Cleaned_Review'] = df['Review text'].apply(clean_text)

print("3. Vectorizing Text...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['Cleaned_Review']).toarray()
y = df['Sentiment'].values

print("4. Training Model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

print("Model Accuracy:", model.score(X_test, y_test))

print("5. Saving Model Files...")
pickle.dump(model, open('sentiment_model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))
print("Done! Files saved.")