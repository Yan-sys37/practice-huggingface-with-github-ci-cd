import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 加载数据
df = pd.read_csv("imdb_top_500.csv")

# 把标签从字符串转成 0/1（解决 acc=0.5 的关键！）
le = LabelEncoder()
y = le.fit_transform(df["label"])

# 文本清洗
def clean(t):
    t = t.lower()
    t = re.sub(r"<.*?>", " ", t)
    t = re.sub(r"[^a-zA-Z ]", " ", t)
    return t

X_raw = [clean(str(t)) for t in df["text"]]

# TF-IDF 特征提取
tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
X = tfidf.fit_transform(X_raw).toarray()

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建模型
model = Sequential([
    Dense(128, activation="relu", input_shape=(3000,)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=8, batch_size=16, validation_split=0.1)

# 评估并打印准确率
y_pred = (model.predict(X_test) > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ FINAL TEST ACCURACY: {acc:.4f}")

# 保存模型和向量化器
model.save("model.h5")
joblib.dump(tfidf, "tfidf.pkl")
