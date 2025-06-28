import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load the conversation dataset
df = pd.read_csv("C:/Users/Ritu Srivastava/OneDrive/Desktop/ML PROJECT(INTERN) JUNE/Conversation.csv")

# Use only necessary columns
df = df[['question', 'answer']]

# Drop missing values
df = df.dropna(subset=['question', 'answer'])

# Split data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF and Naive Bayes
model = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(train_df['question'], train_df['answer'])

# Evaluate the model
predictions = model.predict(test_df['question'])
print("Model Accuracy:", accuracy_score(test_df['answer'], predictions))
