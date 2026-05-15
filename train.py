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
from tensorflow.keras.callbacks import EarlyStopping

# ======================
# 加载数据（你的路径不动）
# ======================
df = pd.read_csv("imdb_top_500.csv")
texts = df["text"].astype(str).values
labels = df["label"].values

# ======================
# 超强文本清洗
# ======================
def clean(t):
    t = t.lower()
    t = re.sub(r"<.*?>", " ", t)
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t

texts = [clean(t) for t in texts]

# ======================
# 高维 TF-IDF（关键！）
# ======================
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    stop_words="english"
)
X = tfidf.fit_transform(texts).toarray()
y = np.array(labels)

# 分层划分（保证正负样本均衡）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# 高准确率神经网络
# ======================
model = Sequential([
    Dense(512, activation="relu", input_shape=(10000,)),
    BatchNormalization(),
    Dropout(0.6),

    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 早停防止过拟合
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# 训练
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# ======================
# 输出最终准确率
# ======================
y_pred = (model.predict(X_test) > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
print(f"\n==================================")
print(f"✅ FINAL TEST ACCURACY: {acc:.4f}")
print(f"==================================")

# 保存
model.save("model.h5")
joblib.dump(tfidf, "tfidf.pkl")
