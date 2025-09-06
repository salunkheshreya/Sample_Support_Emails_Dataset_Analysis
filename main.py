import pandas as pd
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob 


# Download stopwords if not already present
nltk.download('stopwords')

# --- 1. Load dataset ---
# Make sure filename is exactly correct, update path if necessary
df = pd.read_csv("Sample_Support_Emails_Datasett.csv")  # <-- corrected typo here

# --- 2. Preview ---
print("First 5 rows:\n", df.head())
print("\n--- Dataset Info ---")
print(df.info())
print("\nUnique senders:", df['sender'].nunique())
print("\n--- Top Subjects ---")
print(df['subject'].value_counts().head())

# --- 3. Text Cleaning ---
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep letters and spaces only
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Combine subject + body text
df['combined'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
df['clean_text'] = df['combined'].apply(clean_text)

print("\n--- Sample Cleaned Text ---")
print(df[['subject', 'clean_text']].head())

# --- 4. Most Common Words ---
all_words = " ".join(df['clean_text']).split()
word_freq = Counter(all_words).most_common(10)
print("\n--- Top 10 Words ---")
print(word_freq)

# --- 5. Vectorize and Cluster ---
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_text'])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

print("\n--- Sample Cluster Assignments ---")
print(df[['subject', 'cluster']].head(10))

# --- Convert 'sent_date' to datetime (handle dayfirst correctly) ---
df['sent_date'] = pd.to_datetime(df['sent_date'], errors='coerce', dayfirst=True)
df = df.dropna(subset=['sent_date'])  # drop rows with invalid/missing dates

# --- Plot: Emails per Day ---
emails_per_day = df.groupby(df['sent_date'].dt.date).size()
plt.figure(figsize=(10, 5))
emails_per_day.plot(kind='bar', color='skyblue')
plt.title("Emails per Day")
plt.xlabel("Date")
plt.ylabel("Email Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Plot: Cluster Distribution ---
plt.figure(figsize=(6, 4))
sns.countplot(x='cluster', data=df, palette='Set2')
plt.title("Email Clusters (AI Categories)")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Emails")
plt.tight_layout()
plt.show()

# --- Save clustered emails to CSV ---
df.to_csv("clustered_emails.csv", index=False)
print("\n✅ Clustered results saved as clustered_emails.csv")

# --- Interpret Clusters: Top Keywords per Cluster ---
print("\n--- Top Keywords per Cluster ---")
for i in range(4):
    cluster_text = " ".join(df[df['cluster'] == i]['clean_text'])
    top_words = Counter(cluster_text.split()).most_common(5)
    keywords = ", ".join([word for word, _ in top_words])
    print(f"Cluster {i}: {keywords}")
# 11. Sentiment Analysis (Polarity and Subjectivity)
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

df['polarity'], df['subjectivity'] = zip(*df['clean_text'].apply(analyze_sentiment))

print("\n--- Sentiment Sample ---")
print(df[['subject', 'polarity', 'subjectivity']].head())

# 12. Plot Sentiment Distribution
plt.figure(figsize=(10,5))
sns.histplot(df['polarity'], bins=20, kde=True, color='coral')
plt.title('Distribution of Email Sentiment Polarity')
plt.xlabel('Polarity (-1 negative, +1 positive)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(df['subjectivity'], bins=20, kde=True, color='purple')
plt.title('Distribution of Email Sentiment Subjectivity')
plt.xlabel('Subjectivity (0 objective, 1 subjective)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 13. Summary Dashboard: Emails per Sender + Average Sentiment
sender_summary = df.groupby('sender').agg(
    email_count=('sender', 'size'),
    avg_polarity=('polarity', 'mean'),
    avg_subjectivity=('subjectivity', 'mean')
).reset_index()

print("\n--- Email and Sentiment Summary by Sender ---")
print(sender_summary)

# Optional: Save summary
sender_summary.to_csv("sender_sentiment_summary.csv", index=False)
print("\n✅ Summary saved as sender_sentiment_summary.csv")