:

📊 Telegram Sentiment Analysis

This project is about analyzing Telegram channel messages and detecting their sentiment (Positive / Negative / Neutral).

We use Streamlit to build interactive dashboards and applications.

📂 Dataset

The dataset used is telegram_channels_messages14021213_with_sentiment.csv.
The important columns are:

channel → Telegram channel name.

id → Message ID.

text → The original comment text.

date → Date & time of the message.

views → Number of views for the message.

scores → Percentages of sentiment (positive, negative, neutral).

compound → Combined polarity score between -1 and +1.

sentiment_type → The final label (Positive / Negative / Neutral).

(Details were taken from the provided document)

Column_Description

.

🛠️ Features
1. app.py → Sentiment Classifier

Cleans and preprocesses text (stopwords removal, lemmatization).

Applies TF-IDF vectorization.

Handles class imbalance with Random Oversampling.

Trains a Naive Bayes classifier.

Shows class distribution before & after oversampling.

Lets the user enter text and get sentiment prediction live.

2. loc.py → Sentiment Dashboard

Upload your own CSV file.

Preprocess and clean text.

Explore:

Sample of the data.

Custom text sentiment preview.

Sentiment distribution by channel.

Sentiment distribution by date.

Includes interactive plots with Seaborn and Matplotlib.

▶️ How to Run

Install requirements:

pip install streamlit pandas numpy matplotlib seaborn scikit-learn imbalanced-learn nltk wordcloud


Make sure to download NLTK resources:

import nltk
nltk.download('stopwords')
nltk.download('wordnet')


Run the app:

streamlit run app.py


or

streamlit run loc.py

📈 Model

Algorithm: Multinomial Naive Bayes.

Vectorizer: TF-IDF (max_features = 5000).

Oversampling: RandomOverSampler (to balance classes).

Accuracy score is shown directly inside the app.

📌 Notes

Default dataset is used in app.py, but in loc.py you can upload any dataset with the same structure.

Supports English text preprocessing (stopwords, lemmatization).

The dashboard is interactive for exploring Telegram data.
