import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import datetime
from collections import Counter
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

color_palette = sns.color_palette("plasma")
sns.set_palette(color_palette)
df = pd.read_csv(r"C:\Users\ARABIA\OneDrive\Desktop\Telegram smeanting analysis\telegram_channels_messages14021213_with_sentiment.csv")
df.head().style.background_gradient(cmap='plasma')
df.describe().style.background_gradient(cmap='tab20c')
null=df.isnull().sum()
ratio=null/df.shape[0]
pd.DataFrame({'null':null,'ratio':ratio}).T
def clean_text(text):
    """
    Cleans the input text by removing special characters, URLs, and converting to lowercase.
    """
    if isinstance(text, str):  # Check if the input is a string
        text = re.sub(r'http\S+|www\S+|@\S+', '', text, flags=re.MULTILINE) # Remove URLs and mentions
        text = re.sub(r'[^a-zA-Z]', ' ', text) # Remove special characters and numbers
        text = text.lower() # Convert to lowercase
        return text
    else:
        return ""  # Return an empty string for non-string values

df['cleaned_text'] = df['text'].apply(clean_text)
print(df[['text', 'cleaned_text']].head())

def preprocess_text(text):
    """
    Tokenizes, removes stopwords, and lemmatizes the input text.
    """
    # Download stopwords if not already downloaded
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')

    # Download wordnet if not already downloaded
    try:
        WordNetLemmatizer().lemmatize('test') # Use a dummy word to check if wordnet is available
    except LookupError:
        nltk.download('wordnet')


    # Tokenize the text
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    # Join the words back into one string
    return ' '.join(words)

df['processed_text'] = df['cleaned_text'].apply(preprocess_text)
print(df[['cleaned_text', 'processed_text']].head())

df['text'] = df['text'].fillna('')
df['processed_text'] = df['processed_text'].fillna('')
print("\nMissing values after handling:")
df.isnull().sum()

# Sentiment Distribution
sentiment_counts = df['sentiment_type'].value_counts()
print("\nSentiment Distribution:\n", sentiment_counts)
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title('Distribution of Sentiment Types', fontsize = 14, fontweight = 'bold', color = 'forestgreen')
plt.xlabel('Sentiment',fontsize = 12, fontweight = 'bold', color = 'darkblue')
plt.ylabel('Number of Posts',fontsize = 12, fontweight = 'bold', color = 'darkblue')
plt.gca().set_facecolor('#dff2e1')

# Add values on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.show()

# Most Common Words
def plot_most_common_words(text_series, top_n=20):
    """
    Plots the most common words in a text series.
    """
    all_words = ' '.join(text_series).split()
    word_counts = Counter(all_words)
    most_common_words = word_counts.most_common(top_n)
    print("Most Common Words:", most_common_words)
    words, counts = zip(*most_common_words)
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=list(words), y=list(counts))
    plt.xticks(rotation=45)
    plt.title('Most Common Words', fontsize = 14, fontweight = 'bold', color = 'forestgreen')
    plt.xlabel('Words',fontsize = 12, fontweight = 'bold', color = 'deeppink')
    plt.ylabel('Frequency',fontsize = 12, fontweight = 'bold', color = 'deeppink')
    plt.gca().set_facecolor('#dff2e1')

    #Add values to the  most common words
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    plt.show()

plot_most_common_words(df['processed_text'])

# Sentiment Distribution by Channel
channel_sentiment = df.groupby('channel')['sentiment_type'].value_counts().unstack(fill_value=0)
print("Sentiment Distribution by Channel:\n", channel_sentiment)
plt.figure(figsize=(12, 6))
ax = channel_sentiment.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Sentiment Distribution by Channel', fontsize = 14, fontweight = 'bold', color = 'forestgreen')
plt.xlabel('Channel',fontsize = 12, fontweight = 'bold', color = 'deeppink')
plt.ylabel('Number of Posts',fontsize = 12, fontweight = 'bold', color = 'deeppink')
plt.legend(title='Sentiment')
plt.gca().set_facecolor('#dff2e1')

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0f}', (x + width/2, y + height/2),
                ha='center', va='center', xytext=(0, 0), textcoords='offset points', color='black')
plt.show()

# Sentiment Trends Over Time
# Convert 'date' to datetime objects, handling potential parsing issues
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# If 'date' column has any NaT values (failed parsing), handle them
df = df.dropna(subset=['date'])

# Group by date and sentiment type
daily_sentiment = df.groupby([df['date'].dt.date, 'sentiment_type']).size().unstack(fill_value=0)
print("Daily Sentiment Counts:\n", daily_sentiment.head())
# Plot sentiment trends over time
daily_sentiment.plot(figsize=(14, 7))
plt.title('Sentiment Trends Over Time', fontsize = 14, fontweight = 'bold', color = 'forestgreen')
plt.xlabel('Date',fontsize = 12, fontweight = 'bold', color = 'darkcyan')
plt.ylabel('Number of Posts',fontsize = 12, fontweight = 'bold', color = 'darkcyan')
plt.legend(title='Sentiment')
plt.xticks(rotation=45)  # Rotate date labels for readability
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.gca().set_facecolor('#dff2e1')
plt.show()

# Using TF-IDF to convert text to numerical data.
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limiting features to 5000
X = tfidf_vectorizer.fit_transform(df['processed_text'])
y = df['sentiment_type']

# Split the Data:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()

model.fit(X_train, y_train)

# Evaluate the Model:
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='gist_rainbow_r', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix', fontsize = 14, fontweight = 'bold', color = 'forestgreen')
plt.xlabel('Predicted Label',fontsize = 12, fontweight = 'bold', color = 'saddlebrown')
plt.ylabel('True Label',fontsize = 12, fontweight = 'bold', color = 'saddlebrown')
plt.gca().set_facecolor('#dff2e1')
plt.show()

# Hyperparameter Tuning for TF-IDF
from sklearn.model_selection import GridSearchCV
# Define parameter grid
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}
# Setup grid search
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=3, scoring='accuracy')
# Fit grid search to the data
grid_search.fit(X_train, y_train)
# Print best parameters and score
print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", grid_search.best_score_)

# Use the best estimator found by grid search
best_model = grid_search.best_estimator_

# Evaluate the Model:
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# --- Model Tuning and Comparison ---

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluates a model and plots the confusion matrix."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='gist_rainbow_r', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# 1. Multinomial Naive Bayes with Hyperparameter Tuning

print("\n--- Multinomial Naive Bayes with Hyperparameter Tuning ---")

param_grid_nb = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}

grid_search_nb = GridSearchCV(MultinomialNB(), param_grid_nb, cv=3, scoring='accuracy', n_jobs=-1)
grid_search_nb.fit(X_train, y_train)

print("Best parameters found: ", grid_search_nb.best_params_)
print("Best score found: ", grid_search_nb.best_score_)

best_model_nb = grid_search_nb.best_estimator_
evaluate_model(best_model_nb, X_test, y_test, "Multinomial Naive Bayes")


