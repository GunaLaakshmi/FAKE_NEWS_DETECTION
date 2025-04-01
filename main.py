import pandas as pd
import numpy as np
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# **Step 1: Data Collection**
df_true = pd.read_csv("True.csv")
df_fake = pd.read_csv("Fake.csv")

# **Step 2: Data Preprocessing**
df_true["label"] = 1  # Real news (1)
df_fake["label"] = 0  # Fake news (0)

# Combining both datasets
df = pd.concat([df_true, df_fake], ignore_index=True)

# Cleaning text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in square brackets
    text = re.sub(r"https?://\S+|www\.\S+", '', text)  # Remove URLs
    text = re.sub(r"<.*?>+", '', text)  # Remove HTML tags
    text = re.sub(r"[{}]".format(string.punctuation), '', text)  # Remove punctuation
    return text

# Apply cleaning to the 'text' column
df['text'] = df['text'].astype(str).apply(clean_text)

# **Step 3: Exploratory Data Analysis (EDA)**
plt.figure(figsize=(2,2))
sns.countplot(x='label', data=df)
plt.title('Distribution of Real vs Fake News')
plt.xlabel('Label (1: Real, 0: Fake)')
plt.ylabel('Count')
plt.show()

# **Step 4: Feature Engineering**
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorization using TF-IDF (convert text to numerical data)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# **Step 5: Model Training**
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# **Step 6: Model Evaluation**
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', report)

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# **Step 7: Model Saving and Deployment**
with open('fake_news_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

def predict_news(news_text):
    cleaned_text = clean_text(news_text)
    transformed_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(transformed_text)[0]
    return "Real News" if prediction == 1 else "Fake News"

# Example usage
example_news = "Breaking: Government announces new policies to tackle inflation."
print(f'Prediction: {predict_news(example_news)}')
