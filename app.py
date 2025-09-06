import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# ------------------- Text Cleaning -------------------
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+|www\S+|@\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = text.lower()
        return text
    else:
        return ""

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

# ------------------- Load Data -------------------
df = pd.read_csv(r"C:\Users\ARABIA\OneDrive\Desktop\Telegram smeanting analysis\telegram_channels_messages14021213_with_sentiment.csv")

df['cleaned_text'] = df['text'].apply(clean_text)
df['processed_text'] = df['cleaned_text'].apply(preprocess_text)
df['processed_text'] = df['processed_text'].fillna('')
df['sentiment_type'] = df['sentiment_type'].fillna('neutral')

# ------------------- Vectorization -------------------
X = df['processed_text']
y = df['sentiment_type']

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# ------------------- Visualize class distribution before oversampling -------------------
st.subheader("📊 توزيع الفئات قبل Oversampling:")
fig1, ax1 = plt.subplots()
sns.countplot(y=y)
ax1.set_title("Before Oversampling")
st.pyplot(fig1)

# ------------------- Apply Random Oversampling -------------------
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_tfidf, y)

# ------------------- Visualize class distribution after oversampling -------------------
st.subheader("📊 توزيع الفئات بعد Oversampling:")
fig2, ax2 = plt.subplots()
sns.countplot(y=y_resampled)
ax2.set_title("After Oversampling")
st.pyplot(fig2)

# ------------------- Train/Test Split -------------------
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ------------------- Train Model -------------------
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ------------------- Streamlit App -------------------
st.title("📊 Sentiment Analysis App")
st.write("أدخل نص وسيتم تحديد إذا كان **Positive / Negative / Neutral**")
st.write(f"📈 Model Accuracy: {acc:.2f}")

# ------------------- User Input -------------------
user_input = st.text_area("✍️ أدخل النص هنا:")

if st.button("تحليل"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        processed = preprocess_text(cleaned)
        vec = tfidf_vectorizer.transform([processed])
        prediction = model.predict(vec)[0]

        st.subheader("✅ النتيجة:")
        if prediction.lower() == "positive":
            st.success("🌟 Positive")
        elif prediction.lower() == "negative":
            st.error("🚨 Negative")
        else:
            st.info("😐 Neutral")
    else:
        st.warning("⚠️ من فضلك أدخل نص أولاً.")
