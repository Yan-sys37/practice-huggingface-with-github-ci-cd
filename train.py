import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# ======================
# 你的文件路径（不动）
# ======================
df = pd.read_csv("imdb_top_500.csv")
texts = df["text"].astype(str).values
labels = df["label"].values

# ======================
# 超强文本清洗
# ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

texts = [clean_text(t) for t in texts]

# ======================
# 超强 TF-IDF
# ======================
tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    stop_words="english"
)
X = tfidf.fit_transform(texts).toarray()
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# 高准确率神经网络
# ======================
model = Sequential([
    Dense(512, activation="relu", input_shape=(8000,)),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 训练
model.fit(
    X_train, y_train,
    epochs=12,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# ======================
# 输出准确率
# ======================
y_pred = (model.predict(X_test) > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ FINAL TEST ACCURACY: {acc:.4f}")

# 保存
model.save("model.h5")
joblib.dump(tfidf, "tfidf.pkl")
