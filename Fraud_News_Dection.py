import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('data.csv')
print("Dataset preview:\n", data.head())

data = data.dropna(subset=['title', 'real'])

X = data['title']
y = data['real']

labels = ['Fake', 'Real']
sizes = [y.value_counts()[0], y.value_counts()[1]]
colors = ['#e67e22','#3498db']
explode = (0.1, 0)

fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
ax[0].set_title("Distribution of Real vs Fake News")
ax[0].axis('equal')
ax[1].bar(labels, sizes, color=['#ff9999', '#66b3ff'])
ax[1].set_title('Bar Chart: Real vs Fake News')
ax[1].set_xlabel('Category')
ax[1].set_ylabel('Count')
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
