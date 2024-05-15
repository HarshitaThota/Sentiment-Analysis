import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('IMDB.csv')
texts = df['review'].tolist()  
labels = df['sentiment'].tolist()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

processed_texts = [preprocess_text(text) for text in texts]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


def predict_sentiment(text):
    processed_text = preprocess_text(text)
    features = vectorizer.transform([processed_text])
    prediction = clf.predict(features)[0]
    return prediction


test_text = "This was an absolutely stunning film."
print("Sentiment:", predict_sentiment(test_text))
