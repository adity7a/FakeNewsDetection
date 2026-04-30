import pandas as pd
import numpy as np
import re, os, pickle
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ── SETUP ─────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

MAX_WORDS  = 50000
MAX_LEN    = 200
EMBED_DIM  = 128
BATCH_SIZE = 128
EPOCHS     = 10       # increased — early stopping will auto stop

MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/raw/Data.csv", encoding="latin1")
print("Columns:", df.columns.tolist())
print("Total rows:", len(df))

TEXT_COL  = "Statement"
LABEL_COL = "Label"

# ── CLEAN TEXT ────────────────────────────────────────────
def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = ' '.join(w for w in text.split() if w not in stop_words and len(w) > 2)
    return text.strip()

print("\nCleaning text...")
df[TEXT_COL]   = df[TEXT_COL].astype(str)
df['Category'] = df['Category'].astype(str)

# Combine Category + Statement — same as app.py will do at prediction time
df['combined'] = df['Category'] + " " + df[TEXT_COL]
df['clean']    = df['combined'].apply(clean)
df = df[df['clean'] != ""].reset_index(drop=True)
print(f"Clean rows remaining: {len(df)}")

# ── LABEL PROCESS ─────────────────────────────────────────
# Your labels: TRUE and Fake → after upper() → TRUE and FAKE
df[LABEL_COL] = df[LABEL_COL].astype(str).str.upper()

print("\nLabel distribution:")
print(df[LABEL_COL].value_counts())

le = LabelEncoder()
df['label_enc'] = le.fit_transform(df[LABEL_COL])

print("\nEncoded classes:", le.classes_)
print("Index 0 =", le.classes_[0], "| Index 1 =", le.classes_[1])
# Expected: Index 0 = FAKE | Index 1 = TRUE

with open(f"{MODEL_DIR}/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# ── TOKENIZATION ──────────────────────────────────────────
print("\nTokenizing...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean'])

sequences = tokenizer.texts_to_sequences(df['clean'])
padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

with open(f"{MODEL_DIR}/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print(f"Vocabulary size: {len(tokenizer.word_index)}")

# ── SPLIT DATA ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    padded,
    df['label_enc'].values,
    test_size=0.2,
    random_state=42,
    stratify=df['label_enc']
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ── CLASS WEIGHTS — fixes imbalance (37800 TRUE vs 18914 FAKE) ──
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"\nClass weights: {class_weight_dict}")
# FAKE gets higher weight so model learns it better

# ── BUILD BiLSTM MODEL ────────────────────────────────────
num_classes = len(le.classes_)
print(f"\nBuilding model for {num_classes} classes...")

model = Sequential([
    Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),

    Bidirectional(LSTM(128, return_sequences=True)),   # increased from 64
    Dropout(0.3),

    Bidirectional(LSTM(64, return_sequences=True)),    # added second LSTM
    Dropout(0.3),

    GlobalMaxPooling1D(),

    Dense(128, activation='relu'),                     # increased from 64
    Dropout(0.4),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── TRAIN ─────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        patience=3,
        restore_best_weights=True,
        monitor='val_accuracy',
        verbose=1
    ),
    ModelCheckpoint(
        f"{MODEL_DIR}/bilstm_model.keras",
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    )
]

print("\nTraining model...")
print("(Early stopping will stop automatically when accuracy stops improving)\n")

model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1                          # ← add karo
)

# ── EVALUATE ──────────────────────────────────────────────
print("\nEvaluating on test set...")
y_pred_prob = model.predict(X_test)
y_pred      = np.argmax(y_pred_prob, axis=1)

acc = accuracy_score(y_test, y_pred)
print(f"\n{'='*40}")
print(f"Final Accuracy: {acc*100:.2f}%")
print(f"{'='*40}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── SAVE FINAL MODEL ──────────────────────────────────────
model.save(f"{MODEL_DIR}/bilstm_model.keras")

print("\nSaved:")
print(f"  - {MODEL_DIR}/bilstm_model.keras")
print(f"  - {MODEL_DIR}/tokenizer.pkl")
print(f"  - {MODEL_DIR}/label_encoder.pkl")

print(f"\nExpected accuracy: 88-94%")
print("Now run: python app.py")