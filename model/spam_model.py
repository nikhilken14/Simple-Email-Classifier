import pandas as pd
import pickle
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Download NLTK stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("dataset/spam.csv", encoding="latin-1")[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'spam': 1, 'ham': 0})  # Convert labels to binary

# Text Preprocessing
def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join([word for word in text.split() if word not in stopwords.words('english')])

df['cleaned_text'] = df['text'].apply(preprocess_text)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)

# Build Model (TF-IDF + Random Forest)
model = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100, random_state=42))

# Train the Model
model.fit(X_train, y_train)

# Save Model
with open("model/spam_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
