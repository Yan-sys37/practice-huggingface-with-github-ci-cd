import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 加载数据
df = pd.read_csv("imdb_top_500.csv")

# 文本列是 text，标签列是 label
text_col = "text"
label_col = "label"

texts = df[text_col].astype(str).values
labels = df[label_col].values

# 文本清洗
def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    return text.lower()

texts = [clean_text(t) for t in texts]

# TF-IDF 词袋特征
tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
X = tfidf.fit_transform(texts).toarray()
y = np.array(labels)

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建MLP模型
model = Sequential([
    Dense(128, activation="relu", input_shape=(3000,)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=16)

# 评估模型
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(f"✅ 测试集准确率: {accuracy_score(y_test, y_pred):.4f}")

# 保存模型（文件名要和yml里的上传列表一致）
model.save("model.h5")
joblib.dump(tfidf, "tfidf.pkl")
print("✅ 模型已保存为 model.h5 和 tfidf.pkl")
