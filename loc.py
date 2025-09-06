import streamlit as st
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

# Set style
color_palette = sns.color_palette("plasma")
sns.set_palette(color_palette)

st.title("📊 Telegram Sentiment Analysis Dashboard")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ======================
    # Text Cleaning
    # ======================
    def clean_text(text):
        if isinstance(text, str):
            text = re.sub(r'http\S+|www\S+|@\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'[^a-zA-Z]', ' ', text)
            text = text.lower()
            return text
        else:
            return ""

    df['cleaned_text'] = df['text'].apply(clean_text)

    def preprocess_text(text):
        try:
            stopwords.words('english')
        except LookupError:
            nltk.download('stopwords')
        try:
            WordNetLemmatizer().lemmatize('test')
        except LookupError:
            nltk.download('wordnet')

        words = text.split()
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in words]
        return ' '.join(words)

    df['processed_text'] = df['cleaned_text'].apply(preprocess_text)
    df['text'] = df['text'].fillna('')
    df['processed_text'] = df['processed_text'].fillna('')

    st.subheader("📌 Data Sample")
    st.write(df.head())

    # ======================
    # User Inputs
    # ======================

    st.sidebar.header("🔍 Custom Analysis")

    # 1️⃣ Enter a word/sentence
    user_text = st.sidebar.text_input("Enter a text to analyze sentiment:")

    if user_text:
        cleaned = clean_text(user_text)
        processed = preprocess_text(cleaned)
        st.subheader("Sentiment for Your Text")
        st.write(f"📝 Original: {user_text}")
        st.write(f"✅ Cleaned: {processed}")

        # Simple rule-based check from dataset (demo)
        found = df[df['processed_text'].str.contains(processed, case=False, na=False)]
        if not found.empty:
            st.write("Matched Sentiments from Dataset:")
            st.write(found[['text', 'sentiment_type']].head(10))
        else:
            st.write("No exact matches found in dataset.")

    # 2️⃣ Select a channel
    if "channel" in df.columns:
        channel_choice = st.sidebar.selectbox("Choose a channel:", df['channel'].unique())
        st.subheader(f"📌 Sentiment Distribution for Channel: {channel_choice}")
        channel_data = df[df['channel'] == channel_choice]['sentiment_type'].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=channel_data.index, y=channel_data.values, ax=ax)
        plt.title(f"Sentiments in {channel_choice}")
        st.pyplot(fig)

    # 3️⃣ Select a date
    if "date" in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        date_choice = st.sidebar.date_input("Select a date:", min(df['date']), min(df['date']), max(df['date']))
        daily_data = df[df['date'].dt.date == date_choice]
        if not daily_data.empty:
            st.subheader(f"📅 Sentiment on {date_choice}")
            fig, ax = plt.subplots()
            sns.countplot(x=daily_data['sentiment_type'], ax=ax)
            st.pyplot(fig)
        else:
            st.warning(f"No messages found on {date_choice}")

else:
    st.info("⬆️ Please upload a CSV file to start the analysis")

