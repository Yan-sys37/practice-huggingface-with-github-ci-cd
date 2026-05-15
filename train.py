import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Embedding, GlobalMaxPooling1D, Conv1D, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')

# 1. 加载数据
df = pd.read_csv("imdb_top_500.csv")

# 2. 增强的文本清洗
def enhanced_clean_text(text):
    text = str(text).lower()
    # 移除HTML标签
    text = re.sub(r"<.*?>", " ", text)
    # 移除特殊字符但保留表情符号的基本情感
    text = re.sub(r"http\S+|www\S+", "", text)  # 移除URL
    # 扩展常见缩写
    abbreviations = {
        r"can\'t": "cannot",
        r"won\'t": "will not",
        r"n\'t": " not",
        r"\'re": " are",
        r"\'s": " is",
        r"\'d": " would",
        r"\'ll": " will",
        r"\'ve": " have",
        r"\'m": " am"
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text)
    # 移除标点但保留情感相关的特殊字符
    text = re.sub(r"[^\w\s!?]", " ", text)
    # 标准化空格
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 创建多个文本特征
df["clean_text"] = df["text"].apply(enhanced_clean_text)
df["text_length"] = df["clean_text"].apply(len)
df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

# 3. 处理标签
le = LabelEncoder()
df["encoded_label"] = le.fit_transform(df["label"])

# 4. 多元特征提取
# TF-IDF特征
tfidf_word = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    stop_words="english",
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

tfidf_char = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 5),
    max_features=5000
)

# 词频特征
count_vec = CountVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    binary=True
)

# 组合特征
X_tfidf_word = tfidf_word.fit_transform(df["clean_text"])
X_tfidf_char = tfidf_char.fit_transform(df["clean_text"])
X_count = count_vec.fit_transform(df["clean_text"])

# 合并所有文本特征
X_text = hstack([X_tfidf_word, X_tfidf_char, X_count])

# 添加统计特征
stats_features = df[["text_length", "word_count"]].values
X_combined = hstack([X_text, stats_features]).toarray()

y = df["encoded_label"].values

# 5. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# 6. 创建集成模型
def create_deep_model(input_dim):
    model = Sequential([
        Dense(512, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC()]
    )
    return model

# 7. 训练深度学习模型
input_dim = X_train.shape[1]
model = create_deep_model(input_dim)

# 回调函数
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=0.00001,
    verbose=1
)

# 训练
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 8. 使用集成学习提升性能
# 传统机器学习模型
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

nb_model = MultinomialNB(alpha=0.1)
svm_model = SVC(kernel='linear', probability=True, C=1.0, random_state=42)

# 训练传统模型
rf_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# 9. 集成预测
# 获取各个模型的预测概率
dl_pred_proba = model.predict(X_test).flatten()
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
nb_pred_proba = nb_model.predict_proba(X_test)[:, 1]
svm_pred_proba = svm_model.predict_proba(X_test)[:, 1]

# 加权集成（深度学习权重更高）
ensemble_proba = (
    dl_pred_proba * 0.4 +  # 深度学习模型
    rf_pred_proba * 0.3 +  # 随机森林
    nb_pred_proba * 0.15 +  # 朴素贝叶斯
    svm_pred_proba * 0.15    # SVM
)

# 集成预测
ensemble_pred = (ensemble_proba > 0.5).astype(int)

# 10. 评估
# 单个模型评估
dl_pred = (dl_pred_proba > 0.5).astype(int)
dl_acc = accuracy_score(y_test, dl_pred)
print(f"\n📊 Deep Learning Model Accuracy: {dl_acc:.4f}")

rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"🌲 Random Forest Accuracy: {rf_acc:.4f}")

nb_pred = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)
print(f"📈 Naive Bayes Accuracy: {nb_acc:.4f}")

svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
print(f"🔧 SVM Accuracy: {svm_acc:.4f}")

# 集成模型评估
ensemble_acc = accuracy_score(y_test, ensemble_pred)
print(f"\n🎯 ENSEMBLE MODEL ACCURACY: {ensemble_acc:.4f}")

# 详细评估报告
if ensemble_acc >= 0.92:
    print("\n✅ 目标达成！准确率 >= 0.92")
else:
    print("\n⚠️  未达到目标，尝试以下优化：")
    print("1. 增加训练数据量")
    print("2. 使用预训练的词向量（Word2Vec/GloVe）")
    print("3. 使用BERT等预训练模型")
    print("4. 更复杂的模型架构（如CNN+LSTM）")
    print("5. 超参数调优")

print("\n📋 分类报告：")
print(classification_report(y_test, ensemble_pred, target_names=le.classes_))

# 11. 保存模型
model.save("model_optimized.h5")
joblib.dump(tfidf_word, "tfidf_word.pkl")
joblib.dump(tfidf_char, "tfidf_char.pkl")
joblib.dump(count_vec, "count_vec.pkl")
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(nb_model, "nb_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(le, "label_encoder.pkl")

# 保存集成模型的权重
ensemble_weights = {
    'dl': 0.4,
    'rf': 0.3,
    'nb': 0.15,
    'svm': 0.15
}
joblib.dump(ensemble_weights, "ensemble_weights.pkl")

print("\n💾 所有模型已保存！")

# 12. 可选：如果准确率还不够，尝试深度学习模型的交叉验证
if ensemble_acc < 0.92:
    print("\n🔄 尝试交叉验证优化深度学习模型...")
    
    def build_model():
        return create_deep_model(input_dim)
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in kfold.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        fold_model = build_model()
        fold_model.fit(
            X_train_fold, y_train_fold,
            epochs=20,
            batch_size=32,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        fold_pred = (fold_model.predict(X_val_fold) > 0.5).astype(int)
        fold_acc = accuracy_score(y_val_fold, fold_pred)
        cv_scores.append(fold_acc)
    
    print(f"📊 交叉验证平均准确率: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
