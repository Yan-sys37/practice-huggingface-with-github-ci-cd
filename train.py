import numpy as np
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 自动下载 IMDB 数据（不用手动上传）
url = "https://raw.githubusercontent.com/rasbt/python-machine-learning-book-3rd-edition/master/ch08/movie_data.csv.gz"
df = pd.read_csv(url, compression="gzip")

# 文本清洗
def clean(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    return text.lower()

df["review"] = df["review"].apply(clean)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X = tfidf.fit_transform(df["review"]).toarray()
y = df["sentiment"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 神经网络模型
model = Sequential([
    Dense(256, activation="relu", input_shape=(5000,)),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.1)

# 评估
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(f"\n✅ Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 保存
model.save("imdb_sentiment_model.h5")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("\n✅ Model saved successfully!")
