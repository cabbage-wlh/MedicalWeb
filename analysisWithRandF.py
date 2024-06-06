import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

plt.rcParams["font.sans-serif"] = ["SimSun"]


def df_init(df):
    # 将分类变量转换为数值变量
    label_encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = label_encoder.fit_transform(df[col])
    return df

def get_X_y(df, selected_cols=None):
    if selected_cols is None:
        selected_df = df.drop(columns=["生存状态"]).sample(n=5, axis=1)
    else:
        selected_df = df[selected_cols]
        # selected_df = df.drop(columns=["生存状态"])
    X = selected_df
    y = df["生存状态"]
    return X, y


def get_importance(df, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf = RandomForestClassifier(n_estimators=1024, random_state=42)
    rf.fit(X_train, y_train)
    feature_importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    return feature_importance_df


def get_correlation(X):
    correlation_matrix = X.corr()
    return correlation_matrix


# 生成示例数据
#df = pd.read_csv("filled_dataset.csv")
# df = df_init(df)
# X, y = get_X_y(df)
# print(X, y)
# feature_importance_df = get_importance(df, X, y)
# correlation_matrix = get_correlation(X)

# print(df['生存状态'].value_counts())
# data['Target'] = np.random.randint(0, 2, size=n_samples)
# df = pd.DataFrame(data)


# print(feature_importance_df, type(feature_importance_df))
# # 绘制特征重要性条形图
# plt.figure(figsize=(10, 6))
# sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
# plt.title("Feature Importance")
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.savefig("feature_importance.png")
# plt.show()

# 绘制特征之间的相关性热力图
# plt.figure(figsize=(10, 8))
# print(correlation_matrix, type(correlation_matrix))
# sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
# plt.title("Feature Correlation Heatmap")
# plt.savefig("feature_correlation.png")
# plt.show()
