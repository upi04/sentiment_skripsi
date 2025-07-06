import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Daftar singkatan informal dan penggantiannya
slang_dict = {
    'yg': 'yang',
    'dg': 'dengan',
    'sdh': 'sudah',
    'tdk': 'tidak',
    'ga': 'tidak',
    'gk': 'tidak',
    'klo': 'kalau',
    'klu': 'kalau',
    'tp': 'tapi',
    'aja': 'saja',
    'jg': 'juga',
    'dgn': 'dengan',
    'gitu': 'begitu',
    'lu': 'kamu',
    'gue': 'saya',
    'blm': 'belum',
    'udah': 'sudah',
    'banget': 'sangat',
    'dr': 'dari',
    'sbg': 'sebagai'
}

# Daftar stopwords yang diperluas
stopwords = set([
    'yang', 'dan', 'di', 'ke', 'dari', 'itu', 'untuk', 'pada', 'adalah', 'dengan', 
    'juga', 'karena', 'ini', 'sudah', 'belum', 'akan', 'atau', 'saja', 'tidak', 
    'iya', 'ya', 'oh', 'nah', 'lagi', 'lah', 'kok', 'pun', 'apa', 'jadi', 'banget',
    'rt', 'dkk', 'wkwk', 'hehe', 'si', 'kan', 'dah', 'nya', 'dong', 'mah', 'deh'
])

# Fungsi bersihkan teks
def clean_text(text):
    if pd.isnull(text):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|pic\.twitter\.com\S+", "", text) # hapus URL
    text = re.sub(r"@\w+", "", text)  # hapus mention
    text = re.sub(r"[^a-z\s]", " ", text)  # hapus karakter non-huruf
    text = re.sub(r"(.)\1{2,}", r"\1", text)  # hilangkan huruf berulang (contoh: "baguuuus" -> "bagus")
    text = re.sub(r"\s+", " ", text).strip()  # hapus spasi berlebih

    # Ganti singkatan
    for slang, formal in slang_dict.items():
        text = re.sub(r"\b{}\b".format(slang), formal, text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    stemmed = [stemmer.stem(t) for t in tokens]
    return ' '.join(stemmed)

# Load CSV
file_path = "processing/tantangan_guru_kurikulum.csv"
df = pd.read_csv(file_path, sep=";", encoding="utf-8")  # ganti delimiter ke titik koma
df.columns = df.columns.str.strip()
print("📌 Kolom tersedia:", df.columns.tolist())

# Terapkan fungsi baru
df['clean_text'] = df['full_text'].apply(clean_text)

# Simpan hasil
df.to_csv("after_processing/tantangan_guru_kurikulum.csv", sep=";", index=False, encoding="utf-8")
print("✅ Preprocessing UPGRADE selesai dan disimpan ke 'after_processing/tantangan_guru_kurikulumu.csv'")