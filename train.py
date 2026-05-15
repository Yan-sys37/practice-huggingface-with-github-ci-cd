import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 加载数据（先打印列名，确保正确）
df = pd.read_csv("imdb_top_500.csv")
print("=== CSV列名 ===")
print(df.columns.tolist())

# 注意：把这里改成你CSV文件里真实的列名！
text_col = "text"       # 你的文本列名
label_col = "sentiment" # 你的标签列名

texts = df[text_col].astype(str).values
labels = df[label_col].values

# 文本清洗
def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    return text.lower()

texts = [clean_text(t) for t in texts]

# TF-IDF
tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
X = tfidf.fit_transform(texts).toarray()
y = np.array(labels)

# 训练
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential([
    Dense(128, activation="relu", input_shape=(3000,)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=3, batch_size=16)

# 评估
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(f"✅ 测试集准确率: {accuracy_score(y_test, y_pred):.4f}")

# 保存（这两个文件名要和yml里的一致！）
model.save("model.h5")
joblib.dump(tfidf, "tfidf.pkl")
print("✅ 模型已保存为 model.h5 和 tfidf.pkl")
