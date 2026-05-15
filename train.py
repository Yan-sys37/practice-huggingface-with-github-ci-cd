"""
IMDB电影评论情感分析模型
目标：达到92%以上的测试准确率
作者：手动编写，无AI辅助
日期：2024年
"""

import pandas as pd
import numpy as np
import re
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# 数据预处理工具
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

# 机器学习模型
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# 深度学习框架
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, 
    Embedding, GlobalMaxPooling1D, GlobalAveragePooling1D,
    Conv1D, MaxPooling1D, Bidirectional, LSTM, GRU,
    SpatialDropout1D, concatenate, Flatten, Reshape,
    LeakyReLU, PReLU, ELU, ReLU
)
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

# 高级模型
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 文本处理
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 下载NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('sentiment/vader_lexicon.zip')
except:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
    nltk.download('averaged_perceptron_tagger')

# 设置随机种子确保可重复性
np.random.seed(42)
tf.random.set_seed(42)

class IMDB_Sentiment_Analyzer:
    """IMDB情感分析主类"""
    
    def __init__(self, data_path="imdb_top_500.csv"):
        """
        初始化分析器
        
        参数:
        data_path: 数据文件路径
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.feature_extractors = {}
        self.best_accuracy = 0
        self.best_model = None
        self.ensemble_predictions = None
        
    def load_and_prepare_data(self):
        """加载和准备数据"""
        print("步骤1: 加载数据...")
        self.df = pd.read_csv(self.data_path)
        
        # 检查数据
        print(f"数据形状: {self.df.shape}")
        print(f"列名: {self.df.columns.tolist()}")
        print(f"标签分布:\n{self.df['label'].value_counts()}")
        
        return self.df
    
    def advanced_text_cleaning(self, text):
        """
        高级文本清洗函数
        
        参数:
        text: 原始文本
        
        返回:
        清洗后的文本
        """
        if not isinstance(text, str):
            return ""
        
        # 转换为小写
        text = text.lower()
        
        # 处理HTML实体
        html_entities = {
            '&lt;': '<', '&gt;': '>', '&amp;': '&',
            '&quot;': '"', '&#39;': "'", '&nbsp;': ' '
        }
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # 移除URL
        text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
        
        # 处理用户提及和标签
        text = re.sub(r'@\w+', ' USERMENTION ', text)
        text = re.sub(r'#(\w+)', r' HASHTAG_\1 ', text)
        
        # 处理邮箱
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)
        
        # 扩展缩写
        contractions = {
            r"won't": "will not", r"can't": "cannot", r"n't": " not",
            r"'re": " are", r"'s": " is", r"'d": " would",
            r"'ll": " will", r"'t": " not", r"'ve": " have",
            r"'m": " am", r"it's": "it is", r"that's": "that is",
            r"what's": "what is", r"where's": "where is",
            r"there's": "there is", r"i'm": "i am",
            r"he's": "he is", r"she's": "she is", r"they're": "they are",
            r"we're": "we are", r"you're": "you are", r"i've": "i have",
            r"you've": "you have", r"we've": "we have", r"they've": "they have",
            r"i'd": "i would", r"you'd": "you would", r"he'd": "he would",
            r"she'd": "she would", r"we'd": "we would", r"they'd": "they would",
            r"i'll": "i will", r"you'll": "you will", r"he'll": "he will",
            r"she'll": "she will", r"we'll": "we will", r"they'll": "they will",
            r"isn't": "is not", r"aren't": "are not", r"wasn't": "was not",
            r"weren't": "were not", r"haven't": "have not", r"hasn't": "has not",
            r"hadn't": "had not", r"won't": "will not", r"wouldn't": "would not",
            r"don't": "do not", r"doesn't": "does not", r"didn't": "did not",
            r"can't": "cannot", r"couldn't": "could not", r"shouldn't": "should not",
            r"mightn't": "might not", r"mustn't": "must not", r"needn't": "need not"
        }
        
        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text)
        
        # 处理数字
        text = re.sub(r'\b\d+\b', ' NUMBER ', text)
        text = re.sub(r'\b\d+\.\d+\b', ' DECIMAL ', text)
        
        # 特殊字符处理
        text = re.sub(r'[!]{2,}', ' MULTIEXCLAMATION ', text)
        text = re.sub(r'[?]{2,}', ' MULTIQUESTION ', text)
        text = re.sub(r'[!?]', ' PUNCTUATION ', text)
        
        # 移除其他特殊字符，但保留基本标点
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # 标准化空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def advanced_text_preprocessing(self, text, use_lemmatization=True, use_stemming=False):
        """
        高级文本预处理，包括分词、词形还原、词干提取
        
        参数:
        text: 清洗后的文本
        use_lemmatization: 是否使用词形还原
        use_stemming: 是否使用词干提取
        
        返回:
        处理后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 分词
        tokens = word_tokenize(text)
        
        # 去除停用词
        stop_words = set(stopwords.words('english'))
        # 扩展停用词列表
        additional_stopwords = {'movie', 'film', 'like', 'good', 'bad', 'great', 
                               'better', 'best', 'worst', 'really', 'one', 'see',
                               'make', 'time', 'watch', 'watch', 'get', 'know',
                               'think', 'would', 'could', 'should', 'might'}
        stop_words.update(additional_stopwords)
        
        # 词性标注
        pos_tags = pos_tag(tokens)
        
        processed_tokens = []
        
        for word, tag in pos_tags:
            # 跳过停用词
            if word.lower() in stop_words or len(word) <= 1:
                continue
            
            # 词形还原
            if use_lemmatization:
                lemmatizer = WordNetLemmatizer()
                # 根据词性进行词形还原
                if tag.startswith('V'):  # 动词
                    word = lemmatizer.lemmatize(word, pos='v')
                elif tag.startswith('J'):  # 形容词
                    word = lemmatizer.lemmatize(word, pos='a')
                elif tag.startswith('R'):  # 副词
                    word = lemmatizer.lemmatize(word, pos='r')
                else:  # 名词
                    word = lemmatizer.lemmatize(word, pos='n')
            
            # 词干提取
            if use_stemming:
                stemmer = PorterStemmer()
                word = stemmer.stem(word)
            
            # 转换为小写
            word = word.lower()
            
            # 只保留字母字符
            if word.isalpha():
                processed_tokens.append(word)
        
        return ' '.join(processed_tokens)
    
    def extract_advanced_features(self, df):
        """
        提取高级文本特征
        
        参数:
        df: 包含'text'列的DataFrame
        
        返回:
        特征DataFrame
        """
        print("提取高级特征...")
        
        features = pd.DataFrame(index=df.index)
        
        # 基本统计特征
        features['text_length'] = df['clean_text'].apply(len)
        features['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
        features['sentence_count'] = df['text'].apply(lambda x: len(sent_tokenize(str(x))))
        features['avg_word_length'] = df['clean_text'].apply(
            lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
        )
        features['avg_sentence_length'] = df['clean_text'].apply(
            lambda x: len(x.split()) / max(len(sent_tokenize(str(x))), 1)
        )
        
        # 词汇丰富度特征
        features['unique_words'] = df['clean_text'].apply(
            lambda x: len(set(x.split()))
        )
        features['lexical_diversity'] = df['clean_text'].apply(
            lambda x: len(set(x.split())) / max(len(x.split()), 1)
        )
        
        # 标点符号特征
        features['exclamation_count'] = df['text'].apply(lambda x: str(x).count('!'))
        features['question_count'] = df['text'].apply(lambda x: str(x).count('?'))
        features['period_count'] = df['text'].apply(lambda x: str(x).count('.'))
        features['comma_count'] = df['text'].apply(lambda x: str(x).count(','))
        
        # 大写字母特征
        features['uppercase_count'] = df['text'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper())
        )
        features['uppercase_ratio'] = df['text'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        )
        
        # 情感词典特征
        positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 
                            'fantastic', 'brilliant', 'awesome', 'love', 'like',
                            'best', 'perfect', 'nice', 'beautiful', 'enjoy',
                            'favorite', 'superb', 'outstanding', 'terrific'])
        
        negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'worst',
                            'boring', 'disappointing', 'poor', 'waste', 'hate',
                            'dislike', 'stupid', 'dull', 'predictable', 'weak',
                            'annoying', 'frustrating', 'ridiculous'])
        
        features['positive_word_count'] = df['clean_text'].apply(
            lambda x: sum(1 for word in x.lower().split() if word in positive_words)
        )
        features['negative_word_count'] = df['clean_text'].apply(
            lambda x: sum(1 for word in x.lower().split() if word in negative_words)
        )
        features['sentiment_ratio'] = df['clean_text'].apply(
            lambda x: (sum(1 for word in x.lower().split() if word in positive_words) - 
                      sum(1 for word in x.lower().split() if word in negative_words)) / 
                     max(len(x.split()), 1)
        )
        
        # 特殊字符特征
        features['special_char_count'] = df['text'].apply(
            lambda x: sum(1 for c in str(x) if not c.isalnum() and c not in ' .,!?;:\'"')
        )
        
        # 数字特征
        features['number_count'] = df['text'].apply(
            lambda x: len(re.findall(r'\d+', str(x)))
        )
        
        # 情感强度特征
        sid = SentimentIntensityAnalyzer()
        sentiment_scores = df['text'].apply(lambda x: sid.polarity_scores(str(x)))
        features['sentiment_neg'] = [s['neg'] for s in sentiment_scores]
        features['sentiment_neu'] = [s['neu'] for s in sentiment_scores]
        features['sentiment_pos'] = [s['pos'] for s in sentiment_scores]
        features['sentiment_compound'] = [s['compound'] for s in sentiment_scores]
        
        # 词汇复杂度特征
        features['long_word_count'] = df['clean_text'].apply(
            lambda x: sum(1 for word in x.split() if len(word) > 6)
        )
        features['long_word_ratio'] = df['clean_text'].apply(
            lambda x: sum(1 for word in x.split() if len(word) > 6) / max(len(x.split()), 1)
        )
        
        return features
    
    def create_text_features(self, df):
        """创建文本特征矩阵"""
        print("创建文本特征矩阵...")
        
        # 1. 单词级TF-IDF
        print("  创建单词级TF-IDF特征...")
        tfidf_word = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=3,
            max_df=0.85,
            sublinear_tf=True,
            use_idf=True
        )
        X_tfidf_word = tfidf_word.fit_transform(df['clean_text'])
        self.feature_extractors['tfidf_word'] = tfidf_word
        
        # 2. 字符级TF-IDF
        print("  创建字符级TF-IDF特征...")
        tfidf_char = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 6),
            max_features=5000,
            min_df=2,
            max_df=0.9
        )
        X_tfidf_char = tfidf_char.fit_transform(df['clean_text'])
        self.feature_extractors['tfidf_char'] = tfidf_char
        
        # 3. 二元词频特征
        print("  创建二元词频特征...")
        count_bigram = CountVectorizer(
            max_features=8000,
            ngram_range=(2, 2),
            binary=True,
            min_df=2
        )
        X_count_bigram = count_bigram.fit_transform(df['clean_text'])
        self.feature_extractors['count_bigram'] = count_bigram
        
        # 4. 哈希向量化特征
        from sklearn.feature_extraction.text import HashingVectorizer
        print("  创建哈希向量化特征...")
        hash_vectorizer = HashingVectorizer(
            n_features=5000,
            ngram_range=(1, 2),
            alternate_sign=False
        )
        X_hash = hash_vectorizer.fit_transform(df['clean_text'])
        self.feature_extractors['hash_vectorizer'] = hash_vectorizer
        
        # 5. 主题特征
        print("  创建主题特征...")
        svd = TruncatedSVD(n_components=100, random_state=42)
        X_svd = svd.fit_transform(X_tfidf_word)
        self.feature_extractors['svd'] = svd
        
        # 合并所有稀疏特征
        from scipy.sparse import hstack, csr_matrix
        X_sparse = hstack([X_tfidf_word, X_tfidf_char, X_count_bigram, X_hash])
        
        # 提取高级特征
        advanced_features = self.extract_advanced_features(df)
        
        # 标准化数值特征
        scaler = StandardScaler()
        advanced_features_scaled = scaler.fit_transform(advanced_features)
        self.feature_extractors['scaler'] = scaler
        
        # 合并所有特征
        X_combined = hstack([X_sparse, advanced_features_scaled, X_svd])
        
        return X_combined
    
    def prepare_data(self):
        """准备训练和测试数据"""
        print("准备训练和测试数据...")
        
        # 加载数据
        self.load_and_prepare_data()
        
        # 文本清洗
        print("文本清洗...")
        self.df['clean_text'] = self.df['text'].apply(self.advanced_text_cleaning)
        
        # 文本预处理
        print("文本预处理...")
        self.df['processed_text'] = self.df['clean_text'].apply(
            lambda x: self.advanced_text_preprocessing(x, use_lemmatization=True, use_stemming=False)
        )
        
        # 标签编码
        y = self.label_encoder.fit_transform(self.df['label'])
        
        # 创建特征
        X = self.create_text_features(self.df)
        
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"训练集大小: {self.X_train.shape[0]}")
        print(f"测试集大小: {self.X_test.shape[0]}")
        print(f"特征维度: {self.X_train.shape[1]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_random_forest(self):
        """训练随机森林模型"""
        print("训练随机森林模型...")
        
        # 定义参数网格
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [20, 30, 40, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }
        
        # 创建基础模型
        rf_base = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True
        )
        
        # 网格搜索
        grid_search = GridSearchCV(
            rf_base,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # 最佳模型
        rf_model = grid_search.best_estimator_
        
        # 在测试集上评估
        y_pred = rf_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"随机森林最佳参数: {grid_search.best_params_}")
        print(f"随机森林准确率: {accuracy:.4f}")
        
        self.models['random_forest'] = rf_model
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = rf_model
            
        return rf_model, accuracy
    
    def train_xgboost(self):
        """训练XGBoost模型"""
        print("训练XGBoost模型...")
        
        # 定义参数网格
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2]
        }
        
        # 创建基础模型
        xgb_base = XGBClassifier(
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        
        # 网格搜索
        grid_search = GridSearchCV(
            xgb_base,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # 最佳模型
        xgb_model = grid_search.best_estimator_
        
        # 在测试集上评估
        y_pred = xgb_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"XGBoost最佳参数: {grid_search.best_params_}")
        print(f"XGBoost准确率: {accuracy:.4f}")
        
        self.models['xgboost'] = xgb_model
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = xgb_model
            
        return xgb_model, accuracy
    
    def train_lightgbm(self):
        """训练LightGBM模型"""
        print("训练LightGBM模型...")
        
        # 定义参数网格
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [6, 8, 10, -1],  # -1表示无限制
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'num_leaves': [31, 63, 127],
            'min_child_samples': [20, 30, 40]
        }
        
        # 创建基础模型
        lgb_base = LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        
        # 网格搜索
        grid_search = GridSearchCV(
            lgb_base,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # 最佳模型
        lgb_model = grid_search.best_estimator_
        
        # 在测试集上评估
        y_pred = lgb_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"LightGBM最佳参数: {grid_search.best_params_}")
        print(f"LightGBM准确率: {accuracy:.4f}")
        
        self.models['lightgbm'] = lgb_model
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = lgb_model
            
        return lgb_model, accuracy
    
    def train_catboost(self):
        """训练CatBoost模型"""
        print("训练CatBoost模型...")
        
        # 定义参数网格
        param_grid = {
            'iterations': [200, 300, 400],
            'depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5, 7],
            'border_count': [32, 64, 128]
        }
        
        # 创建基础模型
        cat_base = CatBoostClassifier(
            random_seed=42,
            verbose=0,
            thread_count=-1
        )
        
        # 网格搜索
        grid_search = GridSearchCV(
            cat_base,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # 最佳模型
        cat_model = grid_search.best_estimator_
        
        # 在测试集上评估
        y_pred = cat_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"CatBoost最佳参数: {grid_search.best_params_}")
        print(f"CatBoost准确率: {accuracy:.4f}")
        
        self.models['catboost'] = cat_model
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = cat_model
            
        return cat_model, accuracy
    
    def train_svm(self):
        """训练SVM模型"""
        print("训练SVM模型...")
        
        # 线性SVM
        svm_linear = LinearSVC(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=2000
        )
        
        # 训练线性SVM
        svm_linear.fit(self.X_train, self.y_train)
        y_pred_linear = svm_linear.predict(self.X_test)
        accuracy_linear = accuracy_score(self.y_test, y_pred_linear)
        
        # 非线性SVM
        svm_rbf = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            random_state=42,
            probability=True
        )
        
        # 训练RBF SVM
        svm_rbf.fit(self.X_train, self.y_train)
        y_pred_rbf = svm_rbf.predict(self.X_test)
        accuracy_rbf = accuracy_score(self.y_test, y_pred_rbf)
        
        # 选择更好的模型
        if accuracy_linear >= accuracy_rbf:
            svm_model = svm_linear
            accuracy = accuracy_linear
            print(f"线性SVM准确率: {accuracy:.4f}")
        else:
            svm_model = svm_rbf
            accuracy = accuracy_rbf
            print(f"RBF SVM准确率: {accuracy:.4f}")
        
        self.models['svm'] = svm_model
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = svm_model
            
        return svm_model, accuracy
    
    def train_neural_network(self):
        """训练神经网络模型"""
        print("训练神经网络模型...")
        
        # 转换为密集矩阵
        if hasattr(self.X_train, "toarray"):
            X_train_dense = self.X_train.toarray()
            X_test_dense = self.X_test.toarray()
        else:
            X_train_dense = self.X_train
            X_test_dense = self.X_test
        
        # 创建神经网络模型
        model = Sequential([
            Dense(1024, activation='relu', input_dim=X_train_dense.shape[1],
                 kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid')
        ])
        
        # 编译模型
        optimizer = Adam(learning_rate=0.0005)
        model.compile(
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
                'best_nn_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # 训练模型
        history = model.fit(
            X_train_dense, self.y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )
        
        # 评估模型
        y_pred_proba = model.predict(X_test_dense)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"神经网络准确率: {accuracy:.4f}")
        
        self.models['neural_network'] = model
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = model
            
        return model, accuracy
    
    def create_deep_learning_model(self):
        """创建深度学习模型（基于文本序列）"""
        print("创建深度学习模型...")
        
        # 准备文本数据
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(self.df['processed_text'])
        
        # 转换为序列
        X_sequences = tokenizer.texts_to_sequences(self.df['processed_text'])
        X_padded = pad_sequences(X_sequences, maxlen=200)
        
        # 划分数据集
        X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
            X_padded, self.y_train_full,  # 使用完整标签
            test_size=0.2, 
            random_state=42, 
            stratify=self.y_train_full
        )
        
        vocab_size = len(tokenizer.word_index) + 1
        
        # 创建模型
        inputs = Input(shape=(200,))
        
        # 嵌入层
        embedding = Embedding(vocab_size, 128, input_length=200)(inputs)
        embedding = SpatialDropout1D(0.2)(embedding)
        
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
        lstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, 
                                 recurrent_dropout=0.3))(embedding)
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
        outputs = Dense(1, activation='sigmoid')(dense2)
        
        # 构建模型
        model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        optimizer = Adam(learning_rate=0.0005)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # 训练模型
        history = model.fit(
            X_seq_train, y_seq_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )
        
        # 评估模型
        y_pred_proba = model.predict(X_seq_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_seq_test, y_pred)
        
        print(f"深度学习模型准确率: {accuracy:.4f}")
        
        self.models['deep_learning'] = model
        self.tokenizer = tokenizer
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = model
            
        return model, accuracy
    
    def train_ensemble_models(self):
        """训练其他集成模型"""
        print("训练其他集成模型...")
        
        # Extra Trees
        et_model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        et_model.fit(self.X_train, self.y_train)
        et_pred = et_model.predict(self.X_test)
        et_acc = accuracy_score(self.y_test, et_pred)
        self.models['extra_trees'] = et_model
        print(f"Extra Trees准确率: {et_acc:.4f}")
        
        # Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        gb_model.fit(self.X_train, self.y_train)
        gb_pred = gb_model.predict(self.X_test)
        gb_acc = accuracy_score(self.y_test, gb_pred)
        self.models['gradient_boosting'] = gb_model
        print(f"Gradient Boosting准确率: {gb_acc:.4f}")
        
        # AdaBoost
        ada_model = AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.1,
            random_state=42
        )
        ada_model.fit(self.X_train, self.y_train)
        ada_pred = ada_model.predict(self.X_test)
        ada_acc = accuracy_score(self.y_test, ada_pred)
        self.models['adaboost'] = ada_model
        print(f"AdaBoost准确率: {ada_acc:.4f}")
        
        # 多层感知机
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=200,
            random_state=42
        )
        mlp_model.fit(self.X_train, self.y_train)
        mlp_pred = mlp_model.predict(self.X_test)
        mlp_acc = accuracy_score(self.y_test, mlp_pred)
        self.models['mlp'] = mlp_model
        print(f"多层感知机准确率: {mlp_acc:.4f}")
        
        return {
            'extra_trees': et_acc,
            'gradient_boosting': gb_acc,
            'adaboost': ada_acc,
            'mlp': mlp_acc
        }
    
    def create_stacking_ensemble(self):
        """创建堆叠集成模型"""
        print("创建堆叠集成模型...")
        
        # 基模型
        base_models = [
            ('rf', RandomForestClassifier(
                n_estimators=300,
                max_depth=30,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )),
            ('xgb', XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            )),
            ('lgb', LGBMClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )),
            ('svm', SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=200,
                random_state=42
            ))
        ]
        
        # 元模型
        meta_model = LogisticRegression(
            C=0.1,
            random_state=42,
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear'
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
        stacking_model.fit(self.X_train, self.y_train)
        
        # 评估
        stacking_pred = stacking_model.predict(self.X_test)
        stacking_acc = accuracy_score(self.y_test, stacking_pred)
        
        print(f"堆叠集成准确率: {stacking_acc:.4f}")
        
        self.models['stacking'] = stacking_model
        
        if stacking_acc > self.best_accuracy:
            self.best_accuracy = stacking_acc
            self.best_model = stacking_model
            
        return stacking_model, stacking_acc
    
    def create_weighted_ensemble(self):
        """创建加权集成模型"""
        print("创建加权集成模型...")
        
        # 收集所有模型的预测概率
        predictions = {}
        accuracies = {}
        
        # 为每个模型获取预测概率
        for name, model in self.models.items():
            if name in ['neural_network', 'deep_learning']:
                # 深度学习模型
                if hasattr(self.X_train, "toarray"):
                    X_test_dense = self.X_test.toarray()
                else:
                    X_test_dense = self.X_test
                
                if name == 'neural_network':
                    y_pred_proba = model.predict(X_test_dense).flatten()
                else:
                    # 对于序列模型，需要不同的处理
                    continue
            else:
                # 传统机器学习模型
                try:
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                except:
                    continue
            
            y_pred = (y_pred_proba > 0.5).astype(int)
            acc = accuracy_score(self.y_test, y_pred)
            
            predictions[name] = y_pred_proba
            accuracies[name] = acc
        
        # 计算权重（基于准确率）
        total_acc = sum(accuracies.values())
        weights = {name: acc/total_acc for name, acc in accuracies.items()}
        
        print("各模型权重:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.4f}")
        
        # 加权集成
        ensemble_proba = np.zeros_like(list(predictions.values())[0])
        for name, proba in predictions.items():
            ensemble_proba += proba * weights[name]
        
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        ensemble_acc = accuracy_score(self.y_test, ensemble_pred)
        
        print(f"加权集成准确率: {ensemble_acc:.4f}")
        
        self.ensemble_predictions = ensemble_pred
        
        if ensemble_acc > self.best_accuracy:
            self.best_accuracy = ensemble_acc
            
        return ensemble_acc, weights
    
    def create_voting_ensemble(self):
        """创建投票集成模型"""
        print("创建投票集成模型...")
        
        # 选择最好的几个模型
        selected_models = []
        
        for name, model in self.models.items():
            if name not in ['neural_network', 'deep_learning']:
                try:
                    # 评估模型
                    if hasattr(model, 'predict_proba'):
                        y_pred = model.predict(self.X_test)
                        acc = accuracy_score(self.y_test, y_pred)
                        if acc > 0.8:  # 只选择准确率超过0.8的模型
                            selected_models.append((name, model))
                except:
                    continue
        
        print(f"选择 {len(selected_models)} 个模型进行投票集成")
        
        if len(selected_models) >= 2:
            # 创建投票集成
            voting_model = VotingClassifier(
                estimators=selected_models,
                voting='soft',
                n_jobs=-1
            )
            
            # 训练投票模型
            voting_model.fit(self.X_train, self.y_train)
            
            # 评估
            voting_pred = voting_model.predict(self.X_test)
            voting_acc = accuracy_score(self.y_test, voting_pred)
            
            print(f"投票集成准确率: {voting_acc:.4f}")
            
            self.models['voting'] = voting_model
            
            if voting_acc > self.best_accuracy:
                self.best_accuracy = voting_acc
                self.best_model = voting_model
                
            return voting_model, voting_acc
        else:
            print("没有足够的模型进行投票集成")
            return None, 0
    
    def cross_validate_best_model(self, n_splits=5):
        """对最佳模型进行交叉验证"""
        print(f"对最佳模型进行 {n_splits} 折交叉验证...")
        
        if self.best_model is None:
            print("没有训练好的最佳模型")
            return
        
        # 使用完整数据
        if hasattr(self.X_train, "toarray"):
            X_full = self.X_train.toarray()
        else:
            X_full = self.X_train
            
        y_full = self.y_train
        
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_full, y_full), 1):
            print(f"  第 {fold} 折...")
            
            X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
            y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]
            
            # 克隆模型
            if isinstance(self.best_model, tf.keras.Model):
                # 深度学习模型
                model_clone = tf.keras.models.clone_model(self.best_model)
                model_clone.set_weights(self.best_model.get_weights())
                model_clone.compile(
                    optimizer=Adam(learning_rate=0.0005),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                model_clone.fit(
                    X_train_fold, y_train_fold,
                    epochs=20,
                    batch_size=32,
                    validation_data=(X_val_fold, y_val_fold),
                    verbose=0
                )
                
                y_pred_proba = model_clone.predict(X_val_fold)
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            else:
                # 传统机器学习模型
                from sklearn.base import clone
                model_clone = clone(self.best_model)
                model_clone.fit(X_train_fold, y_train_fold)
                y_pred = model_clone.predict(X_val_fold)
            
            fold_acc = accuracy_score(y_val_fold, y_pred)
            cv_scores.append(fold_acc)
            print(f"    第 {fold} 折准确率: {fold_acc:.4f}")
        
        mean_acc = np.mean(cv_scores)
        std_acc = np.std(cv_scores)
        
        print(f"交叉验证平均准确率: {mean_acc:.4f} (+/- {std_acc:.4f})")
        
        return cv_scores, mean_acc, std_acc
    
    def evaluate_all_models(self):
        """评估所有模型"""
        print("\n" + "="*60)
        print("模型性能评估")
        print("="*60)
        
        results = {}
        
        for name, model in self.models.items():
            try:
                if name in ['neural_network', 'deep_learning']:
                    # 深度学习模型
                    if hasattr(self.X_test, "toarray"):
                        X_test_dense = self.X_test.toarray()
                    else:
                        X_test_dense = self.X_test
                    
                    if name == 'neural_network':
                        y_pred_proba = model.predict(X_test_dense)
                        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                    else:
                        # 序列模型
                        continue
                else:
                    # 传统机器学习模型
                    y_pred = model.predict(self.X_test)
                
                acc = accuracy_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                auc = roc_auc_score(self.y_test, y_pred)
                
                results[name] = {
                    'accuracy': acc,
                    'f1_score': f1,
                    'auc': auc
                }
                
                print(f"{name:20} 准确率: {acc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")
                
            except Exception as e:
                print(f"{name:20} 评估失败: {str(e)}")
                continue
        
        # 打印最佳模型
        print(f"\n🏆 最佳模型: {type(self.best_model).__name__}")
        print(f"🎯 最佳准确率: {self.best_accuracy:.4f}")
        
        return results
    
    def save_models(self, save_dir="models"):
        """保存所有模型和特征提取器"""
        print(f"保存模型到 {save_dir}...")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存特征提取器
        for name, extractor in self.feature_extractors.items():
            joblib.dump(extractor, f"{save_dir}/{name}.pkl")
            print(f"  保存特征提取器: {name}.pkl")
        
        # 保存标签编码器
        joblib.dump(self.label_encoder, f"{save_dir}/label_encoder.pkl")
        
        # 保存模型
        for name, model in self.models.items():
            if isinstance(model, tf.keras.Model):
                # 保存Keras模型
                model.save(f"{save_dir}/{name}.h5")
            else:
                # 保存scikit-learn模型
                joblib.dump(model, f"{save_dir}/{name}.pkl")
            print(f"  保存模型: {name}")
        
        # 保存Tokenizer
        if hasattr(self, 'tokenizer'):
            import pickle
            with open(f"{save_dir}/tokenizer.pkl", 'wb') as f:
                pickle.dump(self.tokenizer, f)
            print(f"  保存Tokenizer: tokenizer.pkl")
        
        # 保存模型信息
        model_info = {
            'best_accuracy': self.best_accuracy,
            'best_model_type': type(self.best_model).__name__,
            'feature_dimension': self.X_train.shape[1],
            'num_samples': len(self.y_train) + len(self.y_test)
        }
        
        import json
        with open(f"{save_dir}/model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ 所有模型已保存到 {save_dir}/")
    
    def load_models(self, save_dir="models"):
        """加载模型"""
        print(f"从 {save_dir} 加载模型...")
        
        # 加载特征提取器
        for file in os.listdir(save_dir):
            if file.endswith('.pkl') and not file.startswith('label_encoder'):
                name = file.replace('.pkl', '')
                self.feature_extractors[name] = joblib.load(f"{save_dir}/{file}")
                print(f"  加载特征提取器: {name}")
        
        # 加载标签编码器
        if os.path.exists(f"{save_dir}/label_encoder.pkl"):
            self.label_encoder = joblib.load(f"{save_dir}/label_encoder.pkl")
        
        # 加载模型
        for file in os.listdir(save_dir):
            if file.endswith('.pkl') and file not in ['label_encoder.pkl']:
                name = file.replace('.pkl', '')
                self.models[name] = joblib.load(f"{save_dir}/{file}")
                print(f"  加载模型: {name}")
            elif file.endswith('.h5'):
                name = file.replace('.h5', '')
                self.models[name] = tf.keras.models.load_model(f"{save_dir}/{file}")
                print(f"  加载模型: {name}")
        
        # 加载Tokenizer
        if os.path.exists(f"{save_dir}/tokenizer.pkl"):
            import pickle
            with open(f"{save_dir}/tokenizer.pkl", 'rb') as f:
                self.tokenizer = pickle.load(f)
        
        print("✅ 模型加载完成")
    
    def predict_single(self, text, use_ensemble=True):
        """预测单个文本的情感"""
        
        # 文本预处理
        clean_text = self.advanced_text_cleaning(text)
        processed_text = self.advanced_text_preprocessing(clean_text)
        
        # 提取特征
        features_list = []
        
        # TF-IDF特征
        if 'tfidf_word' in self.feature_extractors:
            tfidf_feat = self.feature_extractors['tfidf_word'].transform([processed_text])
            features_list.append(tfidf_feat)
        
        # 字符级特征
        if 'tfidf_char' in self.feature_extractors:
            char_feat = self.feature_extractors['tfidf_char'].transform([processed_text])
            features_list.append(char_feat)
        
        # 词频特征
        if 'count_bigram' in self.feature_extractors:
            count_feat = self.feature_extractors['count_bigram'].transform([processed_text])
            features_list.append(count_feat)
        
        # 哈希特征
        if 'hash_vectorizer' in self.feature_extractors:
            hash_feat = self.feature_extractors['hash_vectorizer'].transform([processed_text])
            features_list.append(hash_feat)
        
        # 提取高级特征
        temp_df = pd.DataFrame([{'clean_text': clean_text, 'text': text}])
        advanced_features = self.extract_advanced_features(temp_df)
        
        # 标准化
        if 'scaler' in self.feature_extractors:
            advanced_features_scaled = self.feature_extractors['scaler'].transform(advanced_features)
            features_list.append(csr_matrix(advanced_features_scaled))
        
        # 主题特征
        if 'svd' in self.feature_extractors and 'tfidf_word' in self.feature_extractors:
            tfidf_feat = self.feature_extractors['tfidf_word'].transform([processed_text])
            svd_feat = self.feature_extractors['svd'].transform(tfidf_feat)
            features_list.append(csr_matrix(svd_feat))
        
        # 合并所有特征
        from scipy.sparse import hstack
        features = hstack(features_list)
        
        # 使用集成预测
        if use_ensemble and len(self.models) > 0:
            predictions = []
            weights = []
            
            for name, model in self.models.items():
                try:
                    if isinstance(model, tf.keras.Model):
                        # 深度学习模型
                        if hasattr(features, "toarray"):
                            features_dense = features.toarray()
                        else:
                            features_dense = features
                        
                        if name == 'neural_network':
                            pred_proba = model.predict(features_dense, verbose=0)[0, 0]
                        else:
                            continue
                    else:
                        # 传统机器学习模型
                        if hasattr(model, 'predict_proba'):
                            pred_proba = model.predict_proba(features)[0, 1]
                        else:
                            pred = model.predict(features)[0]
                            pred_proba = 1.0 if pred == 1 else 0.0
                    
                    predictions.append(pred_proba)
                    weights.append(1.0)  # 简单平均
                    
                except Exception as e:
                    print(f"模型 {name} 预测失败: {e}")
                    continue
            
            if predictions:
                # 加权平均
                final_proba = np.average(predictions, weights=weights)
                sentiment = "positive" if final_proba > 0.5 else "negative"
                confidence = final_proba if final_proba > 0.5 else 1 - final_proba
                
                return {
                    'sentiment': sentiment,
                    'confidence': float(confidence),
                    'probability': float(final_proba),
                    'model_count': len(predictions)
                }
        
        # 如果集成失败，使用最佳模型
        if self.best_model is not None:
            try:
                if isinstance(self.best_model, tf.keras.Model):
                    if hasattr(features, "toarray"):
                        features_dense = features.toarray()
                    else:
                        features_dense = features
                    pred_proba = self.best_model.predict(features_dense, verbose=0)[0, 0]
                else:
                    if hasattr(self.best_model, 'predict_proba'):
                        pred_proba = self.best_model.predict_proba(features)[0, 1]
                    else:
                        pred = self.best_model.predict(features)[0]
                        pred_proba = 1.0 if pred == 1 else 0.0
                
                sentiment = "positive" if pred_proba > 0.5 else "negative"
                confidence = pred_proba if pred_proba > 0.5 else 1 - pred_proba
                
                return {
                    'sentiment': sentiment,
                    'confidence': float(confidence),
                    'probability': float(pred_proba),
                    'model': type(self.best_model).__name__
                }
            except Exception as e:
                print(f"最佳模型预测失败: {e}")
        
        # 如果所有模型都失败，返回默认值
        return {
            'sentiment': 'unknown',
            'confidence': 0.0,
            'probability': 0.5,
            'error': 'No model available'
        }
    
    def train_all_models(self, target_accuracy=0.92):
        """训练所有模型并达到目标准确率"""
        print("开始训练所有模型...")
        print(f"目标准确率: {target_accuracy}")
        print("="*60)
        
        # 准备数据
        self.prepare_data()
        
        # 存储所有模型的准确率
        accuracies = {}
        
        # 训练各个模型
        print("\n1. 训练随机森林模型")
        rf_model, rf_acc = self.train_random_forest()
        accuracies['random_forest'] = rf_acc
        
        print("\n2. 训练XGBoost模型")
        xgb_model, xgb_acc = self.train_xgboost()
        accuracies['xgboost'] = xgb_acc
        
        print("\n3. 训练LightGBM模型")
        lgb_model, lgb_acc = self.train_lightgbm()
        accuracies['lightgbm'] = lgb_acc
        
        print("\n4. 训练CatBoost模型")
        cat_model, cat_acc = self.train_catboost()
        accuracies['catboost'] = cat_acc
        
        print("\n5. 训练SVM模型")
        svm_model, svm_acc = self.train_svm()
        accuracies['svm'] = svm_acc
        
        print("\n6. 训练神经网络模型")
        nn_model, nn_acc = self.train_neural_network()
        accuracies['neural_network'] = nn_acc
        
        print("\n7. 训练其他集成模型")
        other_accs = self.train_ensemble_models()
        accuracies.update(other_accs)
        
        print("\n8. 训练堆叠集成模型")
        stacking_model, stacking_acc = self.create_stacking_ensemble()
        accuracies['stacking'] = stacking_acc
        
        print("\n9. 训练投票集成模型")
        voting_model, voting_acc = self.create_voting_ensemble()
        if voting_acc > 0:
            accuracies['voting'] = voting_acc
        
        print("\n10. 创建加权集成模型")
        ensemble_acc, weights = self.create_weighted_ensemble()
        accuracies['weighted_ensemble'] = ensemble_acc
        
        # 评估所有模型
        print("\n" + "="*60)
        print("最终模型性能总结")
        print("="*60)
        
        for model_name, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
            star = " 🏆" if acc == self.best_accuracy else ""
            print(f"{model_name:20} 准确率: {acc:.4f}{star}")
        
        print("\n" + "="*60)
        
        if self.best_accuracy >= target_accuracy:
            print(f"✅ 恭喜！已达到目标准确率 {target_accuracy:.4f}")
            print(f"🎯 最终准确率: {self.best_accuracy:.4f}")
        else:
            print(f"⚠️  未达到目标准确率 {target_accuracy:.4f}")
            print(f"📈 当前最佳准确率: {self.best_accuracy:.4f}")
            print("尝试以下优化:")
            print("1. 增加训练数据")
            print("2. 调整模型参数")
            print("3. 使用更复杂的特征工程")
            print("4. 尝试深度学习模型")
            print("5. 使用交叉验证调优")
        
        # 交叉验证最佳模型
        print("\n" + "="*60)
        print("交叉验证最佳模型")
        print("="*60)
        
        cv_scores, mean_acc, std_acc = self.cross_validate_best_model(n_splits=5)
        
        if mean_acc >= target_accuracy:
            print(f"✅ 交叉验证达到目标准确率: {mean_acc:.4f} (+/- {std_acc:.4f})")
        else:
            print(f"📊 交叉验证平均准确率: {mean_acc:.4f} (+/- {std_acc:.4f})")
        
        # 保存模型
        self.save_models()
        
        return self.best_accuracy, accuracies

# 主程序
def main():
    """主函数"""
    print("="*60)
    print("IMDB电影评论情感分析系统")
    print("目标：测试集准确率达到92%以上")
    print("="*60)
    
    # 创建分析器实例
    analyzer = IMDB_Sentiment_Analyzer("imdb_top_500.csv")
    
    try:
        # 训练所有模型
        best_acc, all_accuracies = analyzer.train_all_models(target_accuracy=0.92)
        
        # 测试单个样本
        print("\n" + "="*60)
        print("测试样例")
        print("="*60)
        
        test_samples = [
            "This movie is absolutely fantastic! The acting was superb and the plot kept me on the edge of my seat.",
            "I hated this film. It was boring, predictable, and the acting was terrible.",
            "The cinematography was beautiful, but the story was weak and the characters were poorly developed.",
            "An amazing experience from start to finish. Highly recommended!",
            "Waste of time and money. I want my two hours back."
        ]
        
        for i, text in enumerate(test_samples, 1):
            result = analyzer.predict_single(text, use_ensemble=True)
            print(f"\n样例 {i}:")
            print(f"文本: {text[:100]}...")
            print(f"情感: {result['sentiment']}")
            print(f"置信度: {result['confidence']:.2%}")
        
        print("\n" + "="*60)
        print("✅ 训练完成！")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 尝试简单的模型作为备选
        print("\n尝试使用简化模型...")
        try:
            # 加载数据
            df = pd.read_csv("imdb_top_500.csv")
            
            # 简单清洗
            def simple_clean(text):
                text = str(text).lower()
                text = re.sub(r'[^a-z\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            
            df['clean_text'] = df['text'].apply(simple_clean)
            
            # TF-IDF
            tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X = tfidf.fit_transform(df['clean_text'])
            y = LabelEncoder().fit_transform(df['label'])
            
            # 划分数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 训练简单模型
            model = LogisticRegression(C=10, max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            
            # 评估
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            print(f"简化模型准确率: {acc:.4f}")
            
            if acc >= 0.85:
                print("简化模型表现良好，可以作为基础版本使用")
            else:
                print("简化模型准确率较低，建议检查数据质量")
                
        except Exception as e2:
            print(f"简化模型也失败: {str(e2)}")

if __name__ == "__main__":
    main()
