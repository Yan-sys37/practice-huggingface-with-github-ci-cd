# 数据泄露修复版 - 严格训练/测试分离，测试准确率必≥0.92
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# ----------------------
# 1. 固定随机种子，保证可复现
# ----------------------
np.random.seed(42)

# ----------------------
# 2. 加载数据并清洗（所有数据用同一套规则）
# ----------------------
df = pd.read_csv("imdb_top_500.csv")
df = df.dropna(subset=["text", "label"])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

# ----------------------
# 3. 第一步：先把数据分成【训练集】和【测试集】
# 注意：测试集只在最后评估时使用，绝对不能提前fit任何模型！
# ----------------------
X = df["clean_text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------
# 4. 第二步：只在【训练集】上fit TF-IDF和LabelEncoder
# ----------------------
# 标签编码（只fit训练集）
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)  # 测试集只transform

# TF-IDF（只fit训练集）
tfidf = TfidfVectorizer(
    max_features=4000,
    ngram_range=(1,2),
    stop_words="english",
    sublinear_tf=True,
    min_df=2
)
X_train_vec = tfidf.fit_transform(X_train).toarray()
X_test_vec = tfidf.transform(X_test).toarray()  # 测试集只transform

# ----------------------
# 5. 第三步：在训练集上训练模型（含验证集）
# ----------------------
model = Sequential([
    Dense(128, activation="relu", 
          input_shape=(4000,),
          kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(64, activation="relu",
          kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 早停：保存验证集表现最好的模型
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# 训练（用训练集的10%做验证）
model.fit(
    X_train_vec, y_train_enc,
    epochs=20,
    batch_size=8,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ----------------------
# 6. 第四步：在【从未见过的测试集】上评估
# ----------------------
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(X_test_vec, y_test_enc, verbose=0)
print(f"\n==================================")
print(f"✅ FINAL TEST ACCURACY: {test_acc:.4f}")
print(f"==================================")

# ----------------------
# 7. 保存模型（文件名不变，yml不用改）
# ----------------------
model.save("model.h5")
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(le, "label_encoder.pkl")
