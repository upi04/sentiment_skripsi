import pandas as pd

# Lexicon kata positif dan negatif
positive_words = {
    'bagus', 'baik', 'mantap', 'hebat', 'lancar', 'sukses', 'terbaik', 'cerdas', 'berhasil', 'positif',
    'indah', 'luar biasa', 'terpuji', 'produktif', 'efektif', 'inovatif', 'menyenangkan', 'membanggakan',
    'menginspirasi', 'top', 'istimewa', 'keren', 'ramah', 'cepat', 'mudah', 'terpercaya', 'sempurna',
    'supportif', 'menarik', 'luwes', 'fleksibel', 'terampil', 'cermat', 'disiplin', 'peduli'
}

negative_words = {
    'buruk', 'jelek', 'gagal', 'lemah', 'negatif', 'lambat', 'masalah', 'terlambat', 'kacau', 'salah',
    'menyebalkan', 'menjengkelkan', 'ribet', 'tidak jelas', 'tidak efektif', 'tidak ramah', 'benci',
    'kurang', 'tidak bagus', 'tidak baik', 'malas', 'ngawur', 'menyusahkan', 'tidak adil', 'parah',
    'amburadul', 'pelit', 'kasar', 'aneh', 'rumit', 'menakutkan', 'menipu', 'tidak puas', 'zonk'
}

# Fungsi rule-based untuk label sentimen
def label_sentiment(text):
    if pd.isnull(text):
        return 'netral'

    text = text.lower()
    positif = sum(1 for word in text.split() if word in positive_words)
    negatif = sum(1 for word in text.split() if word in negative_words)

    if positif > negatif:
        return 'positif'
    elif negatif > positif:
        return 'negatif'
    else:
        return 'netral'

# Load hasil preprocessing
df = pd.read_csv("after_processing/tantangan_guru_kurikulum.csv", sep=";", encoding="utf-8")

# Terapkan label sentimen
df['sentimen'] = df['clean_text'].apply(label_sentiment)

# Simpan hasil akhir
df.to_csv("labeling/tantangan_guru_kurikulum.csv", sep=";", index=False, encoding="utf-8")
print("✅ Labeling sentimen selesai! File disimpan di 'labeling/tantangan_guru_kurikulum.csv'")
