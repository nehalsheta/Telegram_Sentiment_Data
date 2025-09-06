# ===============================
# 📊 Telegram Sentiment Analysis Dashboard with VADER
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv("telegram_data.csv")  # غيري اسم الملف حسب اسمك

# -------------------------------
# 2. Clean text
# -------------------------------
df['processed_text'] = df['message'].astype(str).str.lower()

# -------------------------------
# 3. Apply VADER Sentiment Analysis
# -------------------------------
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

df['vader_scores'] = df['processed_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['vader_sentiment'] = df['vader_scores'].apply(
    lambda score: 'positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral')
)

# -------------------------------
# 4. Distribution of Sentiments
# -------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x='vader_sentiment', data=df, palette="Set2")
plt.title("Distribution of VADER Sentiments")
plt.show()

# -------------------------------
# 5. WordCloud for Positive / Negative
# -------------------------------
positive_text = " ".join(df[df['vader_sentiment'] == "positive"]['processed_text'])
negative_text = " ".join(df[df['vader_sentiment'] == "negative"]['processed_text'])

wordcloud_pos = WordCloud(width=600, height=400, background_color="white").generate(positive_text)
wordcloud_neg = WordCloud(width=600, height=400, background_color="black").generate(negative_text)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(wordcloud_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Words")

plt.subplot(1,2,2)
plt.imshow(wordcloud_neg, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Words")
plt.show()

# -------------------------------
# 6. Sentiment by Channel (لو عندك عمود channel)
# -------------------------------
if 'channel' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='channel', hue='vader_sentiment', data=df, palette="Set1")
    plt.xticks(rotation=45)
    plt.title("VADER Sentiments by Channel")
    plt.show()

# -------------------------------
# 7. Sentiment Over Time (لو عندك عمود timestamp)
# -------------------------------
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    sentiment_trend = df.groupby(df['timestamp'].dt.date)['vader_scores'].mean()

    plt.figure(figsize=(10, 6))
    sentiment_trend.plot(marker='o')
    plt.title("Sentiment Trend Over Time (VADER)")
    plt.xlabel("Date")
    plt.ylabel("Average Sentiment Score")
    plt.grid()
    plt.show()
