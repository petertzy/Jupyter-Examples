import pandas as pd
from sklearn.datasets import make_classification

# 生成示范数据
X, y = make_classification(
    n_samples=500,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# 转成 DataFrame
df = pd.DataFrame(X, columns=['feature1', 'feature2'])
df['target'] = y

# 保存成 CSV
df.to_csv('data.csv', index=False)
print("示范数据已保存到 data.csv")
