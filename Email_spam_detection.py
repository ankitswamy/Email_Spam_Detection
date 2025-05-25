#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[ ]:


# Load dataset
df = pd.read_csv("C:/Users/lenovo/Downloads/spam.csv", encoding='latin-1')
df = df.rename(columns={df.columns[0]: 'label', df.columns[1]: 'text'})
df = df[['label', 'text']] 
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['label'] = df['label'].astype(int)


# In[ ]:


# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE) 
    text = re.sub(r'\@w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

df['text'] = df['text'].astype(str).apply(preprocess_text)


# In[14]:


# Features and labels
X = df['text']
y = df['label']


# In[15]:


# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# TF-IDF Vectorization
vectorizer = TfidfVectorizer(min_df=3, max_df=0.9, stop_words='english')  
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# In[17]:


# Model
model = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=1000)
model.fit(X_train_tfidf, y_train)


# In[18]:


# Evaluation
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[22]:


# Command-line interface
print("\nEmail Spam Detector")
print("Enter an email message below to check if it's Spam or Ham.")
def predict_email(email):
    email = preprocess_text(email)
    email_tfidf = vectorizer.transform([email])
    prediction = model.predict(email_tfidf)
    return "Spam" if prediction[0] == 1 else "Ham"
while True:
    user_input = input("Enter your email text (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    print(f"This email is: {predict_email(user_input)}\n")


# In[ ]:
import joblib
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')




