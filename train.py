from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# 加载数据
data = load_iris()
X, y = data.data, data.target

# 划分训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, "model.joblib")
print("✅ Model trained and saved as model.joblib")
