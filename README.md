# Sample_Support_Emails_Dataset_Analysis
This project analyzes customer support emails from a sample dataset to help understand common issues, categorize email topics, and gauge customer sentiment.By cleaning and preprocessing the email text data, it applies natural language processing techniques such as tokenization, stopword removal, and clustering.
# 📧 Sample Support Emails Dataset Analysis

This project analyzes a small dataset of 20 support emails. It performs clustering, word frequency analysis, and sentiment analysis using Python, with visualizations to reveal patterns in customer support inquiries.

## 📂 Dataset
The dataset includes the following fields for each email:
- `sender`: Email address of the sender
- `subject`: Email subject line
- `body`: Full message body
- `sent_date`: Timestamp when the email was sent

## 🔍 Key Features

### ✅ Text Preprocessing
- Lowercasing, removing punctuation
- Stopword removal using NLTK
- Combined subject + body into a single cleaned field

### 📊 Data Analysis
- Most common words across all emails
- Count of emails per day
- Top email subjects
- Clustering emails into 4 categories using KMeans
- Sentiment analysis (polarity and subjectivity) using TextBlob

### 📈 Visualizations
- Emails per day (bar chart)
- Cluster distribution (count plot)
- Top keywords per cluster

### 📁 Output Files
- `clustered_emails.csv`: Contains original emails with cluster labels
- `sender_sentiment_summary.csv`: Summarized sentiment scores per sender

---

## 🧠 Key Insights

- **Cluster 0**: Password reset issues  
- **Cluster 1**: Urgent support and account verification  
- **Cluster 2**: Login issues and team communication  
- **Cluster 3**: API and integration issues  

- Most frequent terms: `help`, `account`, `support`, `verification`, `integration`
- Eve was the most frequent sender
- Sentiments were generally neutral to slightly positive

---

## 📌 Technologies Used

- Python (Pandas, Scikit-learn, NLTK, TextBlob)
- Matplotlib & Seaborn (for visualizations)
- KMeans Clustering
- Sentiment Analysis (TextBlob)

---

## 🛠️ Setup Instructions

1. Clone this repository  
```bash
git clone https://github.com/salunkheshreya/Sample_Support_Emails_Dataset_Analysis.git
cd Sample_Support_Emails_Dataset_Analysis
