from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Text dataset
texts = [
    "Machine learning is fascinating",
    "Deep learning uses neural networks",
    "I love cricket",
    "Football is an exciting sport"
]

# Labels: 1 = Technology, 0 = Sports
labels = [1, 1, 0, 0]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Stratified split (IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.5, stratify=labels, random_state=42
)

# Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))