# 极简排查版 - 先让模型跑起来，解决val_acc=0.5的问题
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ----------------------
# 1. 加载数据，先看标签分布
# ----------------------
df = pd.read_csv("imdb_top_500.csv")
print("=== 标签分布 ===")
print(df["label"].value_counts())  # 确认是二分类，比如0/1或pos/neg

# ----------------------
# 2. 极简文本清洗（避免把文本洗没了）
# ----------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)  # 只去掉HTML标签
    return text

df["clean_text"] = df["text"].apply(clean_text)
print("\n=== 清洗后的文本示例 ===")
print(df["clean_text"].head())  # 确认文本不是空的

# ----------------------
# 3. 分层划分数据集（必须先划分，再编码/特征提取）
# ----------------------
X = df["clean_text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n=== 训练集标签分布 ===")
print(pd.Series(y_train).value_counts())
print("=== 测试集标签分布 ===")
print(pd.Series(y_test).value_counts())

# ----------------------
# 4. 标签编码（只在训练集fit）
# ----------------------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
print("\n=== 编码后的标签示例 ===")
print(f"训练集前5个标签: {y_train_enc[:5]}")

# ----------------------
# 5. TF-IDF特征提取（只在训练集fit，降低维度避免稀疏）
# ----------------------
tfidf = TfidfVectorizer(
    max_features=2000,  # 降低维度，减少噪声
    stop_words="english"
)
X_train_vec = tfidf.fit_transform(X_train).toarray()
X_test_vec = tfidf.transform(X_test).toarray()
print(f"\n=== 特征维度: {X_train_vec.shape[1]} ===")

# ----------------------
# 6. 极简模型（先不搞复杂结构，让模型能学到东西）
# ----------------------
model = Sequential([
    Dense(64, activation="relu", input_shape=(2000,)),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 训练（先跑10轮，不早停，看训练/验证准确率变化）
history = model.fit(
    X_train_vec, y_train_enc,
    epochs=10,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# ----------------------
# 7. 评估测试集
# ----------------------
test_loss, test_acc = model.evaluate(X_test_vec, y_test_enc, verbose=0)
print(f"\n==================================")
print(f"✅ FINAL TEST ACCURACY: {test_acc:.4f}")
print(f"==================================")

# 保存模型
model.save("model.h5")
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(le, "label_encoder.pkl")
