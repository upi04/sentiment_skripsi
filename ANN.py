# --- Import Library ---
import pandas as pd
import numpy as np
import re
import nltk
import warnings
import nlpaug.augmenter.word as naw
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

# --- Setup ---
warnings.filterwarnings('ignore')
nltk.download('punkt')
nltk.download('stopwords')

# --- Load Data ---
df = pd.read_csv("labeling/tweet.csv", sep=";")
print(f"Jumlah data awal: {len(df)}")

# --- Preprocessing ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('indonesian'))
    tokens = text.split()
    filtered = [w for w in tokens if w not in stop_words]
    return ' '.join(filtered)

df['clean_text'] = df['full_text'].apply(clean_text)

# --- Bersihkan Label ---
df['sentimen'] = df['sentimen'].str.lower().str.strip()
df = df[df['sentimen'].isin(['positif', 'negatif', 'netral'])]

# --- Data Augmentasi ---
augmenter = naw.ContextualWordEmbsAug(
    model_path='indobenchmark/indobert-base-p1',
    action="substitute"
)

def gment_text_bert(text, n=1):
    gmented_texts = []
    for _ in range(n):
        try:
            gmented = augmenter.augment(text)
            if isinstance(gmented, list):
                gmented = gmented[0]
            gmented_texts.append(gmented)
        except Exception:
            pass
    return gmented_texts

gmented_data = []
num_gments = 1
for label in ['positif', 'negatif', 'netral']:
    subset = df[df['sentimen'] == label]
    for _, row in subset.iterrows():
        g_texts = gment_text_bert(row['clean_text'], n=num_gments)
        for g_text in g_texts:
            gmented_data.append({
                'full_text': row['full_text'],
                'clean_text': g_text,
                'sentimen': label
            })

df_gmented = pd.DataFrame(gmented_data)
df = pd.concat([df, df_gmented], ignore_index=True)
print("Jumlah data setelah augmentasi:\n", df['sentimen'].value_counts())

# --- Tokenisasi ---
tokenizer = Tokenizer(num_words=7000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])
max_length = int(np.percentile([len(s) for s in sequences], 95))
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# --- Label Encoding ---
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['sentimen'])

# --- Split Data ---
x_train, x_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, stratify=labels
)

# --- Class Weights ---
class_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# --- LSTM Model ---
model = Sequential([
    Input(shape=(max_length,)),
    Embedding(input_dim=7000, output_dim=128),
    LSTM(64, return_sequences=True),
    Dropout(0.5),
    LSTM(32),
    Dense(3, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='nadam',
    metrics=['accuracy']
)
model.summary()

# --- Training dengan EarlyStopping ---
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=15,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# --- Evaluasi ---
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))


# NOTE: Bagian visualisasi confusion matrix dan grafik accuracy/loss tidak disertakan sesuai instruksi



# # --- 11. Confusion Matrix ---
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6,5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=label_encoder.classes_,
#             yticklabels=label_encoder.classes_)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.tight_layout()
# plt.show()

# # --- 12. Visualisasi Accuracy & Loss ---
# plt.figure(figsize=(14,5))

# plt.subplot(1,2,1)
# plt.plot(history.history['accuracy'], label='Train Acc')
# plt.plot(history.history['val_accuracy'], label='Val Acc')
# plt.title('Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.subplot(1,2,2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.title('Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
# plt.show()
