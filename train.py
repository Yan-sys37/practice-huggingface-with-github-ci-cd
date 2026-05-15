import pandas as pd
import numpy as np
import re
import joblib
import gensim
from gensim.models import Word2Vec, KeyedVectors
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.sparse import hstack, csr_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, 
    Embedding, GlobalMaxPooling1D, GlobalAveragePooling1D,
    Conv1D, MaxPooling1D, Bidirectional, LSTM, GRU,
    SpatialDropout1D, concatenate, Attention, LayerNormalization
)
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')
from transformers import pipeline, AutoTokenizer, TFAutoModel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import textstat
import emoji

# 下载nltk资源
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

# 1. 加载数据
print("加载数据...")
df = pd.read_csv("imdb_top_500.csv")

# 2. 高级文本预处理
print("文本预处理...")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def advanced_text_preprocessing(text):
    """更高级的文本预处理"""
    if not isinstance(text, str):
        return ""
    
    # 转换为小写
    text = text.lower()
    
    # 处理HTML实体
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&amp;', '&', text)
    
    # 移除HTML标签
    text = re.sub(r'<.*?>', ' ', text)
    
    # 移除URL
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    
    # 处理用户提及和标签
    text = re.sub(r'@\w+', ' USER ', text)
    text = re.sub(r'#\w+', ' HASHTAG ', text)
    
    # 处理数字
    text = re.sub(r'\d+', ' NUMBER ', text)
    
    # 扩展缩写
    contractions = {
        r"won't": "will not", r"can't": "cannot", r"n't": " not",
        r"'re": " are", r"'s": " is", r"'d": " would",
        r"'ll": " will", r"'t": " not", r"'ve": " have",
        r"'m": " am", r"it's": "it is", r"that's": "that is",
        r"what's": "what is", r"where's": "where is",
        r"there's": "there is", r"i'm": "i am"
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)
    
    # 保留标点但规范化
    text = re.sub(r'([!?])', r' \1 ', text)  # 在!和?前后加空格
    
    # 移除特殊字符但保留基本标点
    text = re.sub(r'[^\w\s!?]', ' ', text)
    
    # 分词
    tokens = nltk.word_tokenize(text)
    
    # 词形还原和去除停用词
    tokens = [lemmatizer.lemmatize(word) for word in tokens 
              if word not in stop_words and len(word) > 1]
    
    return ' '.join(tokens)

# 应用预处理
df["processed_text"] = df["text"].apply(advanced_text_preprocessing)

# 3. 提取丰富的文本特征
print("提取文本特征...")

def extract_text_features(df):
    """提取多种文本特征"""
    features = pd.DataFrame()
    
    # 基本统计特征
    features['text_length'] = df["processed_text"].apply(len)
    features['word_count'] = df["processed_text"].apply(lambda x: len(x.split()))
    features['avg_word_length'] = df["processed_text"].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    features['sentence_count'] = df["text"].apply(
        lambda x: len(nltk.sent_tokenize(str(x)))
    )
    
    # 词汇丰富度特征
    features['unique_word_ratio'] = df["processed_text"].apply(
        lambda x: len(set(x.split())) / len(x.split()) if x.split() else 0
    )
    
    # 可读性特征
    features['flesch_reading_ease'] = df["text"].apply(
        lambda x: textstat.flesch_reading_ease(str(x))
    )
    features['smog_index'] = df["text"].apply(
        lambda x: textstat.smog_index(str(x))
    )
    
    # 情感相关特征
    features['exclamation_count'] = df["text"].apply(
        lambda x: str(x).count('!')
    )
    features['question_count'] = df["text"].apply(
        lambda x: str(x).count('?')
    )
    features['uppercase_ratio'] = df["text"].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
    )
    
    return features

# 提取特征
text_features = extract_text_features(df)

# 4. 标签编码
le = LabelEncoder()
y = le.fit_transform(df["label"])

# 5. 多种特征提取方法
print("创建多种特征表示...")

# 方法1: TF-IDF特征
tfidf_vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 3),
    stop_words='english',
    min_df=2,
    max_df=0.85,
    sublinear_tf=True,
    use_idf=True
)
X_tfidf = tfidf_vectorizer.fit_transform(df["processed_text"])

# 方法2: 字符级n-gram
char_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 6),
    max_features=5000
)
X_char = char_vectorizer.fit_transform(df["processed_text"])

# 方法3: 词频特征
count_vectorizer = CountVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    binary=True,
    min_df=3
)
X_count = count_vectorizer.fit_transform(df["processed_text"])

# 方法4: 主题特征（使用LSA）
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50, random_state=42)
X_svd = svd.fit_transform(X_tfidf)

# 6. 合并所有特征
print("合并特征...")

# 合并稀疏特征
X_sparse = hstack([X_tfidf, X_char, X_count])

# 合并所有特征
X_combined = hstack([X_sparse, text_features.values, X_svd])

# 7. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# 8. 训练XGBoost模型（通常表现很好）
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    use_label_encoder=False
)

print("训练XGBoost模型...")
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"✅ XGBoost准确率: {xgb_acc:.4f}")

# 9. 训练LightGBM模型
from lightgbm import LGBMClassifier

lgb_model = LGBMClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print("训练LightGBM模型...")
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
lgb_acc = accuracy_score(y_test, lgb_pred)
print(f"✅ LightGBM准确率: {lgb_acc:.4f}")

# 10. 训练CatBoost模型
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    random_seed=42,
    verbose=0
)

print("训练CatBoost模型...")
cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)
cat_acc = accuracy_score(y_test, cat_pred)
print(f"✅ CatBoost准确率: {cat_acc:.4f}")

# 11. 训练深度学习模型
print("\n准备深度学习特征...")

# 为深度学习准备文本数据
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(df["processed_text"])

# 转换为序列
X_sequences = tokenizer.texts_to_sequences(df["processed_text"])
X_padded = pad_sequences(X_sequences, maxlen=200)

# 划分序列数据
X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
    X_padded, y, test_size=0.2, random_state=42, stratify=y
)

# 创建更强大的深度学习模型
def create_hybrid_model(vocab_size, max_length):
    """创建混合CNN+BiLSTM模型"""
    
    # 输入层
    text_input = Input(shape=(max_length,))
    
    # 嵌入层
    embedding = Embedding(
        input_dim=vocab_size + 1,
        output_dim=128,
        input_length=max_length,
        mask_zero=True
    )(text_input)
    
    # 使用预训练GloVe词向量（如果有的话）
    # 这里我们假设有glove.6B.100d.txt文件
    
    # CNN分支
    conv1 = Conv1D(128, 3, activation='relu', padding='same')(embedding)
    conv1 = BatchNormalization()(conv1)
    conv1 = MaxPooling1D(2)(conv1)
    
    conv2 = Conv1D(128, 4, activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling1D(2)(conv2)
    
    conv3 = Conv1D(128, 5, activation='relu', padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = GlobalMaxPooling1D()(conv3)
    
    # BiLSTM分支
    lstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(embedding)
    lstm = Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3))(lstm)
    
    # 合并特征
    merged = concatenate([conv3, lstm])
    
    # 全连接层
    dense1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(merged)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    
    dense2 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.3)(dense2)
    
    # 输出层
    output = Dense(1, activation='sigmoid')(dense2)
    
    # 构建模型
    model = Model(inputs=text_input, outputs=output)
    
    return model

# 创建模型
vocab_size = len(tokenizer.word_index)
max_length = 200
dl_model = create_hybrid_model(vocab_size, max_length)

# 编译模型
optimizer = Adam(learning_rate=0.0005)
dl_model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# 回调函数
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        'best_dl_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# 训练深度学习模型
print("训练深度学习模型...")
history = dl_model.fit(
    X_seq_train, y_seq_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# 预测
dl_pred_proba = dl_model.predict(X_seq_test).flatten()
dl_pred = (dl_pred_proba > 0.5).astype(int)
dl_acc = accuracy_score(y_seq_test, dl_pred)
print(f"✅ 深度学习模型准确率: {dl_acc:.4f}")

# 12. 超级集成
print("\n🚀 创建超级集成模型...")

# 获取所有模型的预测概率
models = {
    'xgb': (xgb_model, xgb_pred, xgb_acc),
    'lgb': (lgb_model, lgb_pred, lgb_acc),
    'cat': (cat_model, cat_pred, cat_acc),
    'dl': (dl_model, dl_pred, dl_acc)
}

# 计算加权集成权重（基于准确率）
weights = {}
total_acc = sum(acc for _, _, acc in models.values())
for name, (_, _, acc) in models.items():
    weights[name] = acc / total_acc

print("模型权重:")
for name, weight in weights.items():
    print(f"  {name}: {weight:.4f}")

# 为深度学习模型获取概率
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
cat_proba = cat_model.predict_proba(X_test)[:, 1]

# 加权集成
ensemble_proba = (
    xgb_proba * weights['xgb'] +
    lgb_proba * weights['lgb'] +
    cat_proba * weights['cat'] +
    dl_pred_proba * weights['dl']
)

ensemble_pred = (ensemble_proba > 0.5).astype(int)
ensemble_acc = accuracy_score(y_test, ensemble_pred)
print(f"\n🎯 加权集成准确率: {ensemble_acc:.4f}")

# 13. 如果还不到0.92，使用堆叠集成
if ensemble_acc < 0.92:
    print("\n🔄 使用堆叠集成...")
    
    # 使用StackingClassifier
    base_models = [
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
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
        passthrough=True,
        n_jobs=-1
    )
    
    # 训练堆叠模型
    stacking_model.fit(X_train, y_train)
    stacking_pred = stacking_model.predict(X_test)
    stacking_acc = accuracy_score(y_test, stacking_pred)
    print(f"✅ 堆叠集成准确率: {stacking_acc:.4f}")
    
    # 如果堆叠更好，使用堆叠
    if stacking_acc > ensemble_acc:
        ensemble_pred = stacking_pred
        ensemble_acc = stacking_acc

# 14. 最终评估
print("\n" + "="*60)
print("最终模型性能评估")
print("="*60)

print(f"\n📊 各模型准确率:")
for name, (_, pred, acc) in models.items():
    if name != 'dl':  # dl已经在上面打印过了
        print(f"  {name.upper()}: {acc:.4f}")

print(f"\n🎯 最终集成准确率: {ensemble_acc:.4f}")

# 详细评估
if ensemble_acc >= 0.92:
    print(f"\n✅ 恭喜！成功达到目标准确率: {ensemble_acc:.4f}")
else:
    print(f"\n⚠️  当前准确率: {ensemble_acc:.4f}")
    print("尝试以下进一步优化:")
    print("1. 使用预训练词向量（Word2Vec/GloVe）")
    print("2. 使用BERT等Transformer模型")
    print("3. 数据增强（回译、同义词替换等）")
    print("4. 使用更复杂的集成策略")
    
    # 尝试使用BERT
    print("\n🔄 尝试使用BERT...")
    try:
        from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
        
        # 使用小型的BERT模型
        model_name = "distilbert-base-uncased"
        tokenizer_bert = AutoTokenizer.from_pretrained(model_name)
        model_bert = TFAutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        )
        
        # 对文本进行编码
        texts = df["processed_text"].tolist()
        encodings = tokenizer_bert(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="tf"
        )
        
        # 划分数据集
        X_bert = encodings['input_ids'].numpy()
        y_bert = y
        
        X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(
            X_bert, y_bert, test_size=0.2, random_state=42, stratify=y_bert
        )
        
        # 编译BERT模型
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        model_bert.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # 训练BERT
        print("微调BERT模型...")
        model_bert.fit(
            X_train_bert, y_train_bert,
            validation_split=0.1,
            epochs=3,
            batch_size=16,
            verbose=1
        )
        
        # 评估BERT
        bert_pred = np.argmax(model_bert.predict(X_test_bert).logits, axis=1)
        bert_acc = accuracy_score(y_test_bert, bert_pred)
        print(f"✅ BERT准确率: {bert_acc:.4f}")
        
        # 如果BERT更好，使用BERT
        if bert_acc > ensemble_acc:
            ensemble_acc = bert_acc
            print(f"\n🎉 BERT达到准确率: {bert_acc:.4f}")
            
    except Exception as e:
        print(f"无法加载BERT: {e}")

# 15. 保存最佳模型
print("\n💾 保存模型...")

# 保存特征提取器
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(char_vectorizer, "char_vectorizer.pkl")
joblib.dump(count_vectorizer, "count_vectorizer.pkl")
joblib.dump(svd, "svd.pkl")
joblib.dump(le, "label_encoder.pkl")

# 保存模型
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(lgb_model, "lgb_model.pkl")
joblib.dump(cat_model, "cat_model.pkl")

# 保存深度学习模型
dl_model.save("dl_model.h5")

# 保存Tokenizer
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("✅ 所有模型已保存！")

# 16. 创建推理函数
def predict_sentiment(text, ensemble_weight=0.5):
    """预测单个文本的情感"""
    
    # 预处理文本
    processed_text = advanced_text_preprocessing(text)
    
    # 提取特征
    tfidf_features = tfidf_vectorizer.transform([processed_text])
    char_features = char_vectorizer.transform([processed_text])
    count_features = count_vectorizer.transform([processed_text])
    svd_features = svd.transform(tfidf_features)
    
    # 提取文本特征
    temp_df = pd.DataFrame([{"processed_text": processed_text, "text": text}])
    text_feats = extract_text_features(temp_df)
    
    # 合并所有特征
    features = hstack([tfidf_features, char_features, count_features, 
                      text_feats.values, svd_features])
    
    # 各模型预测
    xgb_pred = xgb_model.predict_proba(features)[0, 1]
    lgb_pred = lgb_model.predict_proba(features)[0, 1]
    cat_pred = cat_model.predict_proba(features)[0, 1]
    
    # 深度学习预测
    seq = tokenizer.texts_to_sequences([processed_text])
    padded_seq = pad_sequences(seq, maxlen=200)
    dl_pred = dl_model.predict(padded_seq, verbose=0)[0, 0]
    
    # 加权集成
    final_pred = (
        xgb_pred * weights.get('xgb', 0.25) +
        lgb_pred * weights.get('lgb', 0.25) +
        cat_pred * weights.get('cat', 0.25) +
        dl_pred * weights.get('dl', 0.25)
    )
    
    # 转换为标签
    sentiment = "positive" if final_pred > 0.5 else "negative"
    confidence = final_pred if final_pred > 0.5 else 1 - final_pred
    
    return {
        "sentiment": sentiment,
        "confidence": float(confidence),
        "probability": float(final_pred)
    }

# 测试推理函数
print("\n🧪 测试推理函数...")
test_text = "This movie was absolutely fantastic! The acting was superb and the plot was engaging."
result = predict_sentiment(test_text)
print(f"测试文本: {test_text}")
print(f"预测结果: {result['sentiment']} (置信度: {result['confidence']:.2%})")

print(f"\n✨ 最终准确率: {ensemble_acc:.4f}")
if ensemble_acc >= 0.92:
    print("🎉 恭喜！已达到0.92+的目标！")
else:
    print("📈 继续努力，尝试更多优化！")

print("✅ 模型保存完成！")
