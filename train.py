import pandas as pd
import numpy as np
import joblib
import re
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------------------------------------------------------
# 1. 加载 IMDB 数据
# ------------------------------------------------------------------------------
df = pd.read_csv("imdb_balanced_10k.csv")  # 或 imdb_top_500.csv
texts = df["review"].values
labels = df["sentiment"].values

# ------------------------------------------------------------------------------
# 2. 文本清洗
# ------------------------------------------------------------------------------
def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    return text.lower()

texts = [clean_text(t) for t in texts]

# ------------------------------------------------------------------------------
# 3. 选择：TF-IDF + 神经网络 （你老师要的 BoW/TF-IDF first layer）
# ------------------------------------------------------------------------------
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X = tfidf.fit_transform(texts).toarray()
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------------------------------------------
# 4. 构建神经网络（Neural Network for Sentiment Analysis）
# ------------------------------------------------------------------------------
inputs = Input(shape=(X_train.shape[1],))
x = Dense(256, activation="relu")(inputs)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs, output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 训练
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 评估
y_pred = (model.predict(X_test) > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
print(f"✅ 测试集准确率: {acc:.4f}")

# ------------------------------------------------------------------------------
# 5. 保存模型 + 上传到 Hugging Face
# ------------------------------------------------------------------------------
model.save("imdb_sentiment_model.h5")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("✅ 模型保存完成：imdb_sentiment_model.h5")
print("✅ TF-IDF 向量化工具保存：tfidf_vectorizer.pkl")
