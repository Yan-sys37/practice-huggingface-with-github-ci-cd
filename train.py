import pandas as pd
import numpy as np
import re
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 加载数据
df = pd.read_csv("imdb_top_500.csv")
texts = df["review"].astype(str).values
labels = df["sentiment"].values

# 文本清洗
def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    return text.lower()

texts = [clean_text(t) for t in texts]

# Token化
MAX_VOCAB = 3000
MAX_LEN = 150

tokenizer = Tokenizer(num_words=MAX_VOCAB)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_LEN)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 加载 tiny_glove.json
with open("tiny_glove.json", "r", encoding="utf-8") as f:
    glove_dict = json.load(f)

EMBEDDING_DIM = 50
embedding_matrix = np.zeros((MAX_VOCAB, EMBEDDING_DIM))

for word, i in tokenizer.word_index.items():
    if i < MAX_VOCAB and word in glove_dict:
        embedding_matrix[i] = np.array(glove_dict[word])

# 构建带GloVe嵌入的模型
model = Sequential([
    Embedding(MAX_VOCAB, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False),
    GlobalAveragePooling1D(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=16)

# 评估
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(f"✅ 测试集准确率: {accuracy_score(y_test, y_pred):.4f}")

# 保存
model.save("imdb_glove_model.h5")
joblib.dump(tokenizer, "tokenizer.pkl")
print("✅ 模型文件已保存")
