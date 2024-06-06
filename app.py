import json
import os

from flask import Flask, render_template, request,redirect,session,url_for
import pickle
from analysisWithRandF import *
from data import SourceData
from dataConnect import *
from predict import *
from predict_1 import *
app = Flask(__name__)
app.secret_key = os.urandom(24)
df = pd.read_csv("filled_dataset.csv")
#roc_data={}
def save_pkl(path,data):
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def load_pkl(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data
@app.route('/')
def hello_name():
    return render_template('index.html')


# @app.route('/view')
# def home():
#     return render_template('view.html',form=data)
@app.route('/predict')
def predict():
    predict_T()

    #run_predict()
    print(type(results["MLP"]["FPR"]))
    print(results)
    return render_template('Predict.html',form=results)
@app.route('/choose')
def choose():
    return render_template('choose.html')


@app.route('/view')
def age_page():
    data = SourceData()
    return render_template('view.html', form=data, title=data.title)


@app.route('/submit', methods=['POST', 'GET'])
def submit_form():
    if request.method == 'POST':
        # 获取 JSON 数据
        data = request.json
        selected_options = data.get('selected_options', [])  # 获取名为 'selected_options' 的字段，如果不存在则返回空列表
        print(selected_options)
        # 处理数据（例如，保存到数据库或进行其他逻辑处理）
        #标签编码
        df1 = df_init(df)
        X, y = get_X_y(df1, selected_options)
        feature_importance_df = get_importance(df1, X, y)
        correlation_matrix = get_correlation(X)
        correlation_matrix_values = correlation_matrix.values.tolist()
        correlation_matrix_values = np.nan_to_num(correlation_matrix_values).tolist()
        indices = correlation_matrix.index.to_list()
        importance_values=feature_importance_df.values.tolist()
        importance_index = [item[0] for item in importance_values]
        importance_values=[item[1] for item in importance_values]
        save_pkl('data/correlation_matrix.pkl',correlation_matrix_values)
        save_pkl('data/indices.pkl',indices)
        save_pkl('data/importance_index.pkl',importance_index)
        save_pkl('data/importance_values.pkl',importance_values)
        # return render_template('correlation_heatmap.html', data=correlation_matrix_values, indices=indices)
        return redirect('/submit')
    else:
        # data = session.get('data')
        data=load_pkl('data/correlation_matrix.pkl')
        indices = load_pkl('data/indices.pkl')
        importance_index=load_pkl('data/importance_index.pkl')
        importance_values=load_pkl('data/importance_values.pkl')
        return render_template('correlation_heatmap.html',
                               data=data,
                               indices=indices,
                               importance_index=importance_index,
                               importance_values=importance_values)



def viewData():
    return render_template('')

if __name__ == '__main__':
    app.run(debug=True)
