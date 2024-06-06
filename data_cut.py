import pandas as pd

# 加载CSV文件
df = pd.read_csv('D:/tcga-brca/data_final.csv')

# 筛选出列名为中文的列
# 假设我们通过检查列名是否包含中文字符来确定是否为中文列
# 这里使用了Unicode范围来检查中文字符
chinese_columns = [col for col in df.columns if any(ord(c) >= 0x4e00 and ord(c) <= 0x9fff for c in col)]

# 使用筛选出的列名来创建一个新的DataFrame
df_chinese_columns = df[chinese_columns]

# 显示前几行，以检查结果
print(df_chinese_columns.head())

# 如果需要，可以将结果保存到新的CSV文件
df_chinese_columns.to_csv('D:/tcga-brca/data_all.csv', index=False)