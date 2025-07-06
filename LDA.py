import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import os

nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# Load data
df = pd.read_csv("labeling/tweet.csv", sep=";")

# Preprocessing
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    tokens = text.split()
    return [w for w in tokens if w not in stop_words]

df['tokens'] = df['clean_text'].astype(str).apply(preprocess)

# Buat dictionary dan corpus
dictionary = corpora.Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(text) for text in df['tokens']]

# Train LDA
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10, random_state=42)

# Simpan model dan dictionary
os.makedirs("model", exist_ok=True)
lda_model.save("model/lda_model.model")
dictionary.save("model/lda_dictionary.dict")

print("✅ LDA model and dictionary saved successfully.")
