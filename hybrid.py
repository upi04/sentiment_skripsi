import pandas as pd
import re
import nltk
import os
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from wordcloud import WordCloud
from gensim import corpora
from gensim.models.ldamodel import LdaModel

# Setup
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# Load data
df = pd.read_csv("labeling/tweet.csv", sep=";")
print(f"Jumlah data: {len(df)}")

# Preprocessing
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    tokens = text.split()
    return [w for w in tokens if w not in stop_words]

df['tokens'] = df['clean_text'].astype(str).apply(preprocess)

# Load model & dictionary
lda_model = LdaModel.load("model/lda_model.model")
dictionary = corpora.Dictionary.load("model/lda_dictionary.dict")

# Prediksi topik
def get_topic(text):
    bow = dictionary.doc2bow(text)
    topics = lda_model.get_document_topics(bow)
    return max(topics, key=lambda x: x[1])[0] if topics else -1

df['topic'] = df['tokens'].apply(get_topic)

# Tampilkan topik
for i, topic in lda_model.show_topics(formatted=True):
    print(f"\n🔹 Topik {i+1}:\n{topic}")

# Tabel distribusi sentimen per topik
pivot = pd.crosstab(df['topic'], df['sentimen'], normalize='index') * 100
print("\n📊 Distribusi Sentimen per Topik:")
print(pivot.round(2))

# Simpan hasil
os.makedirs("output", exist_ok=True)
df.to_csv("output/hybrid_result.csv", index=False, sep=";")


# --- Visualisasi WordCloud per Topik ---
for i in range(5):
    plt.figure(figsize=(8, 6))
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate_from_frequencies(dict(lda_model.show_topic(i, 25)))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Topik {i+1}', fontsize=14)
    plt.tight_layout()
    plt.show()
