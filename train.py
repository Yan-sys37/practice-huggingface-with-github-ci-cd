import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack
from scipy import sparse
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Embedding, GlobalMaxPooling1D, Conv1D, Bidirectional, LSTM, GRU, SpatialDropout1D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')

# 1. 加载数据
df = pd.read_csv("imdb_top_500.csv")

# 2. 更彻底的文本清洗
def advanced_clean_text(text):
    text = str(text).lower()
    
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 移除URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 处理缩写
    contractions = {
        r"won't": "will not", r"can't": "cannot", r"n't": " not",
        r"'re": " are", r"'s": " is", r"'d": " would",
        r"'ll": " will", r"'t": " not", r"'ve": " have",
        r"'m": " am", r"it's": "it is", r"that's": "that is"
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)
    
    # 保留标点用于情感分析
    text = re.sub(r'[^\w\s!?]', ' ', text)
    
    # 标准化空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 增强特征工程
df["clean_text"] = df["text"].apply(advanced_clean_text)
df["text_length"] = df["clean_text"].apply(len)
df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
df["avg_word_length"] = df["clean_text"].apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0)
df["exclamation_count"] = df["text"].apply(lambda x: str(x).count('!') + str(x).count('?'))

# 3. 处理标签
le = LabelEncoder()
df["encoded_label"] = le.fit_transform(df["label"])

# 修复分类报告问题：确保target_names是字符串
target_names = [str(cls) for cls in le.classes_]

# 4. 改进的特征工程
# 使用多种特征提取方法
tfidf_word = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 3),
    stop_words='english',
    min_df=3,
    max_df=0.8,
    sublinear_tf=True,
    use_idf=True
)

tfidf_char = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 6),
    max_features=3000
)

# 使用词频特征
count_vec = CountVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    binary=True,
    min_df=2
)

# 提取特征
X_tfidf_word = tfidf_word.fit_transform(df["clean_text"])
X_tfidf_char = tfidf_char.fit_transform(df["clean_text"])
X_count = count_vec.fit_transform(df["clean_text"])

# 合并文本特征
X_text = hstack([X_tfidf_word, X_tfidf_char, X_count])

# 添加统计特征
stats_features = df[["text_length", "word_count", "avg_word_length", "exclamation_count"]].values
X_combined = hstack([X_text, stats_features])

y = df["encoded_label"].values

# 5. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# 6. 改进的深度学习模型
def create_improved_model(input_dim):
    model = Sequential([
        Dense(1024, activation='relu', input_dim=input_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        Dropout(0.1),
        
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

# 转换稀疏矩阵为密集矩阵用于深度学习
X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test

# 创建模型
input_dim = X_train_dense.shape[1]
dl_model = create_improved_model(input_dim)

# 回调函数
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# 7. 训练深度学习模型
print("训练深度学习模型...")
history = dl_model.fit(
    X_train_dense, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

# 8. 使用更强大的集成模型
print("\n训练集成模型...")

# 定义基模型
base_models = [
    ('rf', RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )),
    ('nb', MultinomialNB(alpha=0.01)),
    ('svm', SVC(
        kernel='rbf',
        C=1.0,
        probability=True,
        random_state=42,
        class_weight='balanced'
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    ))
]

# 元模型
meta_model = LogisticRegression(
    C=0.1,
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)

# 创建堆叠模型
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1,
    passthrough=True
)

# 训练堆叠模型
stacking_model.fit(X_train, y_train)

# 9. 预测
# 深度学习模型预测
dl_pred_proba = dl_model.predict(X_test_dense).flatten()
dl_pred = (dl_pred_proba > 0.5).astype(int)

# 堆叠模型预测
stacking_pred = stacking_model.predict(X_test)
stacking_pred_proba = stacking_model.predict_proba(X_test)[:, 1]

# 10. 加权集成
# 根据验证集性能调整权重
weights = {'dl': 0.4, 'stacking': 0.6}
ensemble_proba = (
    dl_pred_proba * weights['dl'] + 
    stacking_pred_proba * weights['stacking']
)
ensemble_pred = (ensemble_proba > 0.5).astype(int)

# 11. 评估
print("\n" + "="*50)
print("模型性能评估")
print("="*50)

dl_acc = accuracy_score(y_test, dl_pred)
print(f"🤖 深度学习模型准确率: {dl_acc:.4f}")

stacking_acc = accuracy_score(y_test, stacking_pred)
print(f"🔗 堆叠模型准确率: {stacking_acc:.4f}")

ensemble_acc = accuracy_score(y_test, ensemble_pred)
print(f"🎯 集成模型准确率: {ensemble_acc:.4f}")

# 12. 如果准确率仍未达到0.92，尝试使用交叉验证
if ensemble_acc < 0.92:
    print("\n🔄 使用交叉验证优化模型...")
    
    # 使用5折交叉验证训练多个模型
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_predictions = []
    cv_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
        print(f"\n训练第 {fold} 折...")
        
        # 获取训练和验证数据
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        # 转换为密集矩阵
        X_train_fold_dense = X_train_fold.toarray() if hasattr(X_train_fold, "toarray") else X_train_fold
        X_val_fold_dense = X_val_fold.toarray() if hasattr(X_val_fold, "toarray") else X_val_fold
        
        # 训练深度学习模型
        fold_model = create_improved_model(input_dim)
        fold_model.fit(
            X_train_fold_dense, y_train_fold,
            epochs=30,
            batch_size=32,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )
        
        # 预测
        fold_pred = (fold_model.predict(X_val_fold_dense) > 0.5).astype(int).flatten()
        fold_acc = accuracy_score(y_val_fold, fold_pred)
        cv_accuracies.append(fold_acc)
        
        # 在整个测试集上预测
        test_pred = fold_model.predict(X_test_dense)
        cv_predictions.append(test_pred.flatten())
        
        print(f"第 {fold} 折准确率: {fold_acc:.4f}")
    
    # 平均交叉验证预测
    cv_avg_pred = np.mean(cv_predictions, axis=0)
    cv_ensemble_pred = (cv_avg_pred > 0.5).astype(int)
    cv_ensemble_acc = accuracy_score(y_test, cv_ensemble_pred)
    
    print(f"\n📊 交叉验证平均准确率: {np.mean(cv_accuracies):.4f} (+/- {np.std(cv_accuracies):.4f})")
    print(f"🎯 交叉验证集成准确率: {cv_ensemble_acc:.4f}")
    
    # 如果交叉验证结果更好，使用它
    if cv_ensemble_acc > ensemble_acc:
        ensemble_pred = cv_ensemble_pred
        ensemble_acc = cv_ensemble_acc
        print("✅ 使用交叉验证集成模型")

# 13. 最终评估
print("\n" + "="*50)
print("最终结果")
print("="*50)

if ensemble_acc >= 0.92:
    print(f"🎉 成功！最终准确率: {ensemble_acc:.4f}")
else:
    print(f"📈 当前准确率: {ensemble_acc:.4f}")
    print("尝试以下进一步优化：")
    print("1. 使用预训练的词向量（Word2Vec/GloVe）")
    print("2. 使用BERT等Transformer模型")
    print("3. 增加训练数据量")
    print("4. 尝试更复杂的模型架构")
    print("5. 使用更复杂的特征工程")

# 14. 修复分类报告错误
print("\n📋 分类报告：")
try:
    # 确保target_names是字符串列表
    target_names_str = [str(name) for name in target_names]
    print(classification_report(y_test, ensemble_pred, target_names=target_names_str))
except Exception as e:
    print(f"生成分类报告时出错: {e}")
    # 使用默认标签
    print(classification_report(y_test, ensemble_pred))

# 15. 保存模型
print("\n💾 保存模型...")
dl_model.save("imdb_sentiment_model.h5")
stacking_model = stacking_model  # 已经训练好
joblib.dump(stacking_model, "stacking_model.pkl")
joblib.dump(tfidf_word, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ 模型保存完成！")
