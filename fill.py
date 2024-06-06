import traceback
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
from tqdm import tqdm


def transform_df(df):
    mapping_dicts = defaultdict(dict)
    reversed_mapping_dicts = defaultdict(dict)
    str_columns = []
    # 遍历每一列
    for col in df.columns:
        # 统计该列中各个字符串出现的次数
        value_counts = df[col].value_counts()
        # 创建映射字典
        mapping_dict = {value: i for i, value in enumerate(value_counts.index)}
        # 创建反向映射字典
        reversed_mapping_dict = {i: value for value, i in mapping_dict.items()}
        # 更新映射字典和反向映射字典
        mapping_dicts[col] = mapping_dict
        reversed_mapping_dicts[col] = reversed_mapping_dict
        # 将字符串替换为对应的数字
        if df[col].dtype == "object":
            str_columns.append(col)
            df[col] = df[col].map(mapping_dict)

    # for col in df.columns:
    #     if df[col].dtype == "object":
    #         print(col)
    #     print(reversed_mapping_dicts[col][0])
    # 输出映射字典和反向映射字典
    # print(mapping_dicts)
    # print(reversed_mapping_dicts)
    return df, reversed_mapping_dicts, str_columns


def fill_nans(df, reversed_mapping_dicts, str_columns):
    missing_cols = [col for col in df.columns if df[col].isnull().any()]
    full_cols=[col for col in df.columns if col not in missing_cols]
    # print(missing_cols)
    df_full = df.drop(columns=missing_cols)
    df_nan = df.loc[:, missing_cols]
    for col in tqdm(missing_cols):
        try:
            nan_rows = df.loc[:, col].isnull()
            # print(col,nan_rows)
            Xtrain = df_full.loc[~nan_rows]
            Ytrain = df_nan.loc[~nan_rows, col]
            Xtest = df_full.loc[nan_rows]
            rfc = RandomForestRegressor(n_estimators=1024)
            rfc = rfc.fit(Xtrain, Ytrain)
            Ypredict = rfc.predict(Xtest)
            max_len = len(reversed_mapping_dicts[col])
            if col in str_columns:
                Ypredict = [
                    round(i) if round(i) < max_len - 1 else max_len - 1
                    for i in Ypredict
                ]
            # print(Ypredict)
            # print(df_nan.loc[nan_rows,col])
            df.loc[nan_rows, col] = Ypredict
            if col in str_columns:
                df[col] = df[col].astype(int)
                df[col] = df[col].map(reversed_mapping_dicts[col])
            # print(df_nan.loc[nan_rows,col])
        except:
            print(col, max_len)
            traceback.print_exc()
    for col in full_cols:
        if col in str_columns:
            df[col] = df[col].astype(int)
            df[col] = df[col].map(reversed_mapping_dicts[col])
    return df


# 读取数据
df = pd.read_csv("D:/tcga-brca/data_all.csv")
df, reversed_mapping_dicts, str_columns = transform_df(df)
# for col in df.columns:
#     print(col,len(reversed_mapping_dicts[col]))
#     for k,v in reversed_mapping_dicts[col].items():
#         print(k,v)
df = fill_nans(df, reversed_mapping_dicts, str_columns)
if df.isnull().any().any():
    print("No!!!")
else:
    print("Nice!!!")
df.to_csv("filled_dataset.csv", index=False)
