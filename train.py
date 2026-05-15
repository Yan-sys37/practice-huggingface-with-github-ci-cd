"""
生成GitHub Pages可用的报告
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def generate_visualizations():
    """生成可视化图表"""
    
    # 读取结果
    with open('models/results.json', 'r') as f:
        results = json.load(f)
    
    # 创建数据框
    models = []
    accuracies = []
    
    for model_name, data in results.items():
        if 'accuracy' in data:
            models.append(model_name)
            accuracies.append(data['accuracy'])
    
    df = pd.DataFrame({'Model': models, 'Accuracy': accuracies})
    df = df.sort_values('Accuracy', ascending=False)
    
    # 创建图表目录
    os.makedirs('reports/plots', exist_ok=True)
    
    # 1. 准确率条形图
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df['Model'], df['Accuracy'], color=['#2E86AB' if acc >= 0.8 else '#F24236' for acc in df['Accuracy']])
    plt.xlabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xlim([0, 1])
    
    # 添加数值标签
    for bar, acc in zip(bars, df['Accuracy']):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{acc:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig('reports/plots/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 目标达成图表
    target_acc = 0.92
    best_acc = df['Accuracy'].max()
    
    plt.figure(figsize=(8, 4))
    colors = ['#F24236', '#2E86AB', '#4CB944']
    labels = ['Below 0.8', '0.8-0.92', 'Above 0.92']
    
    categories = []
    counts = []
    
    for i, label in enumerate(labels):
        if i == 0:
            count = (df['Accuracy'] < 0.8).sum()
        elif i == 1:
            count = ((df['Accuracy'] >= 0.8) & (df['Accuracy'] < target_acc)).sum()
        else:
            count = (df['Accuracy'] >= target_acc).sum()
        
        categories.append(label)
        counts.append(count)
    
    plt.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%')
    plt.title('Model Accuracy Distribution')
    plt.savefig('reports/plots/accuracy_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成Markdown报告
    with open('reports/README.md', 'w') as f:
        f.write(f"""# IMDB Sentiment Analysis Report

## Summary
- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Target Accuracy**: {target_acc:.4f}
- **Best Accuracy Achieved**: {best_acc:.4f}
- **Status**: {'✅ PASS' if best_acc >= target_acc else '❌ FAIL'}

## Model Performance

| Model | Accuracy | Status |
|-------|----------|---------|
""")
        
        for _, row in df.iterrows():
            status = "✅" if row['Accuracy'] >= target_acc else "⚠️" if row['Accuracy'] >= 0.8 else "❌"
            f.write(f"| {row['Model']} | {row['Accuracy']:.4f} | {status} |\n")
        
        f.write(f"""

## Visualizations

![Accuracy Comparison](plots/accuracy_comparison.png)
![Accuracy Distribution](plots/accuracy_distribution.png)

## Recommendations
""")
        
        if best_acc >= target_acc:
            f.write("✅ The model has successfully achieved the target accuracy. Good job!\n")
        else:
            f.write("""
⚠️ The model did not reach the target accuracy. Consider:

1. **Increase training data** - Add more labeled examples
2. **Feature engineering** - Create more informative features
3. **Model tuning** - Perform hyperparameter optimization
4. **Try different architectures** - Experiment with deep learning models
5. **Ensemble methods** - Combine multiple models
""")
    
    print("✅ Visualizations and reports generated successfully")

if __name__ == "__main__":
    generate_visualizations()
