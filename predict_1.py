import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve,accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from analysisWithRandF import df_init
from sklearn.svm import SVC
from icecream import ic
results = {}
# 加载和预处理数据
def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    data = df_init(data)
    X = data.drop(columns=["生存状态", "去世间隔天数", '随访']).values  # 特征列
    y = (1 - data['生存状态']).values  # 标签列
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


# 随机森林模型定义和训练
class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=10, n_samples=None):#决策树数量，决策树最大高度，拆分内部节点所需的最小样本数，
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_samples = n_samples
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [self._most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

    def _bootstrap_sample(self, X, y):
        n_samples = self.n_samples or X.shape[0]
        idxs = np.random.choice(X.shape[0], n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict_proba(self, X):
        tree_probs = np.array([tree.predict_proba(X) for tree in self.trees])
        # print("tree_probs.shape", tree_probs.shape)
        #单棵树决策为正的概率
        return np.mean(tree_probs, axis=0)


# Transformer 模型定义和训练
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_encoder_layers=4, dim_feedforward=512,
                 dropout=0.25):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.transformer_encoder(x)
        x = self.fc(x.squeeze(0))
        return x


# MLP 模型定义和训练
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        hidden_states = self.relu(x)
        hidden_states = self.layers(hidden_states)
        x = self.fc2(hidden_states)
        return x


# 训练模型
def train_transformer(model, criterion, optimizer, train_loader, epochs=20):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        # print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')


def train_mlp(model, criterion, optimizer, train_loader, epochs=20):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        # print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')
def train_SVM(name, model, X_train, X_test, y_train, y_test):
    # Fit model
    model.fit(X_train, y_train)
    # Predict
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    ic(y_proba_test.shape)
    ic(len(y_test))
    # ic(len(y_proba_test))
    # Compute metrics
    ic(f'{name} scoore')
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    auc = roc_auc_score(y_test, y_proba_test)
    ic(accuracy, precision, recall, f1, auc)
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba_test)
    auc_SVM= roc_auc_score(y_test, y_proba_test)
    results[name]={'FPR':fpr.tolist(),'TPR':tpr.tolist(),'AUC':auc_SVM.tolist()}


# 模型评估
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        probabilities = torch.softmax(outputs,dim=1)[:,1].numpy()  # 获取正类的预测概率

    y_test_numpy = y_test.numpy()
    predictions = np.argmax(outputs.numpy(), axis=1)
    fpr, tpr, _ = roc_curve(y_test_numpy, probabilities)
    auc = roc_auc_score(y_test_numpy,probabilities)
    accuracy = accuracy_score(y_test_numpy, predictions)
    precision = precision_score(y_test_numpy, predictions)
    recall = recall_score(y_test_numpy, predictions)
    f1 = f1_score(y_test_numpy, predictions)
    ic(round(accuracy,4), round(precision,4),round(recall,4),round(f1,4),round(auc,4))
    return fpr,tpr,auc


# 主函数
def predict_T():
    file_path = 'filled_dataset.csv'
    X, y = load_data_from_csv(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)
    ic(len(X_train),len(X_test),len(y_test),len(y_train))
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    input_dim = X_train.shape[1]#训练集的样本数，用在MLP和transformer中
    num_classes = len(np.unique(y_train))#二分类,num_classes=2


    # Random Forest
    forest = RandomForest(n_trees=50, max_depth=10)
    forest.fit(X_train, y_train)
    forest_predictions = forest.predict(X_test)
    forest_predictions_proba = forest.predict_proba(X_test)[:,1]
    # print("forest_predictions_proba:",forest_predictions_proba.shape)
    # print("y_test:",y_test.shape)
    # # print("forest_predictions_proba:",forest_predictions_proba)
    fpr_rf, tpr_rf,_= roc_curve(y_test, forest_predictions_proba)
    auc_rf = roc_auc_score(y_test, forest_predictions_proba)
    accuracy_rf = accuracy_score(y_test, forest_predictions)
    precision_rf = precision_score(y_test, forest_predictions)
    recall_rf = recall_score(y_test, forest_predictions)
    f1_rf = f1_score(y_test, forest_predictions)
    ic("random forest score:")
    ic(round(accuracy_rf,4),round(precision_rf,4),round(recall_rf,4),round(f1_rf,4),round(auc_rf,4),round(accuracy_rf,4))
    results['RandomForest'] = {'FPR': fpr_rf.tolist(), 'TPR': tpr_rf.tolist(),'AUC': auc_rf.tolist()}

    # Transformer
    transformer_model = TransformerClassifier(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(transformer_model.parameters(), lr=0.001)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    train_transformer(transformer_model, criterion, optimizer, train_loader, epochs=10)
    ic("transformer score")
    fpr_trans, tpr_trans,auc_trans= evaluate_model(transformer_model, X_test_tensor, y_test_tensor)
    #print(type(fpr_trans.tolist()))
    results['Transformer'] =  {'FPR': fpr_trans.tolist(), 'TPR': tpr_trans.tolist(),'AUC':auc_trans.tolist()}

    # MLP
    mlp_model = MLP(input_dim, hidden_dim=128, output_dim=num_classes)
    optimizer = optim.AdamW(mlp_model.parameters(), lr=0.005)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    train_mlp(mlp_model, criterion, optimizer, train_loader, epochs=10)
    ic("mlp score")
    fpr_mlp, tpr_mlp,auc_mlp = evaluate_model(mlp_model, X_test_tensor, y_test_tensor)
    results['MLP'] = {'FPR': fpr_mlp.tolist(), 'TPR': tpr_mlp.tolist(),'AUC':auc_mlp}
    #print(results)

    #SVM
    SVM_liner= SVC(kernel="linear", probability=True)  # Linear
    SVM_poly= SVC(kernel="poly", probability=True)  # Polynomial
    SVM_RBF=SVC(kernel="rbf", probability=True)  # Radial basis function
    print(type(X_train))
    train_SVM("SVM_Linear", SVM_liner,X_train,X_test,y_train,y_test)
    train_SVM("SVM_Poly", SVM_poly, X_train, X_test, y_train, y_test)
    train_SVM("SVM_RBF", SVM_RBF, X_train, X_test, y_train, y_test)
    #ic(results)
    return results
predict_T()