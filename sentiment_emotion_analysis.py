
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("sentiment-emotion-labelled_Dell_tweets.csv")

# Ensure required columns exist and drop missing data
df = df[['Text', 'sentiment', 'emotion']].dropna()

# === Common Text Vectorization ===
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['Text'])

# === Sentiment Classification ===
print("=== Sentiment Classification ===")
X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(
    X, df['sentiment'], test_size=0.2, random_state=42)

sentiment_model = LogisticRegression(max_iter=1000)
sentiment_model.fit(X_train_sent, y_train_sent)
y_pred_sent = sentiment_model.predict(X_test_sent)
print(classification_report(y_test_sent, y_pred_sent))

# === Emotion Classification ===
print("=== Emotion Classification ===")
X_train_em, X_test_em, y_train_em, y_test_em = train_test_split(
    X, df['emotion'], test_size=0.2, random_state=42)

emotion_model = LogisticRegression(max_iter=1000)
emotion_model.fit(X_train_em, y_train_em)
y_pred_em = emotion_model.predict(X_test_em)
print(classification_report(y_test_em, y_pred_em))
