import os
import shap
from sklearn.calibration import label_binarize
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
import joblib

warnings.filterwarnings('ignore')

# 数据读取并自动划分
def load_and_split_data(file_path, test_size=0.3, random_state=42):
    # 读取 Excel 文件
    data = pd.read_excel(file_path, header=0)  # header=0 表示第一行是列名

    # 假设最后一列为目标标签，其他列为特征
    X = data.iloc[:, :-1]  # 所有行，除了最后一列的特征
    y = data.iloc[:, -1]   # 目标标签（最后一列）

    # 使用LabelEncoder将分类标签转换为数值标签
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # 将字符串标签转为数值

    # 自动划分数据为训练集和测试集（70% 训练集，30% 测试集）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test, label_encoder

# 特征工程
def create_features(df):
    feature_columns = [
        'blink_count', 'fixations', 'saccades', 'static_entropy',
        'transition_entropy', 'std_diff_left', 'std_diff_right', 'Positive', 'Neutral', 'Negative'
    ]
    X = df[feature_columns].copy()
    # 添加多项式特征
    for col in X.columns:
        X[f"{col}_squared"] = X[col] ** 2
        X[f"{col}_cubed"] = X[col] ** 3
    # 添加交互特征
    for i in range(len(feature_columns)):
        for j in range(i + 1, len(feature_columns)):
            X[f"{feature_columns[i]}_{feature_columns[j]}_interaction"] = (
                X[feature_columns[i]] * X[feature_columns[j]]
            )
    # 添加统计特征
    X['feature_mean'] = X[feature_columns].mean(axis=1)
    X['feature_std'] = X[feature_columns].std(axis=1)
    X['feature_sum'] = X[feature_columns].sum(axis=1)
    X['feature_min'] = X[feature_columns].min(axis=1)
    X['feature_max'] = X[feature_columns].max(axis=1)
    return X

# 数据预处理和数据增强
def preprocess_data(X_train, X_test, y_train, max_features):
    # 数据归一化
    scaler = RobustScaler()  # 使用RobustScaler进行归一化，减少异常值的影响
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 特征选择
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        max_features=max_features
    )
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    selected_features = X_train.columns[selector.get_support()]
    return X_train_selected, X_test_selected, selected_features

# 模型初始化
def initialize_models():
    return {
        'RF': RandomForestClassifier(),
        'SVC': SVC(),
        'Logistic Regression': LogisticRegression(),
        'XGBoost': XGBClassifier()
    }

# 参数搜索空间
def get_param_grid():
    param_grid = {
        'RF': {'max_depth': [5], 'max_features': ['sqrt'], 'min_samples_leaf': [1], 'min_samples_split': [5], 'n_estimators': [100], 'random_state': [42]},
        'SVC': {'C': [0.1], 'kernel': ['linear'], 'gamma': ['scale']},
        'Logistic Regression': {'C': [0.1], 'solver': ['saga']},
        'XGBoost': {'n_estimators': [100], 'max_depth': [3], 'learning_rate':[ 0.01], 'subsample': [0.8], 'colsample_bytree': [0.8]}
        # 'RF': {  # 随机森林的参数搜索空间
        #     'n_estimators': [100, 200, 300],
        #     'max_depth': [5, 10, 15],
        #     'min_samples_split': [2, 5],
        #     'min_samples_leaf': [1, 2],
        #     'max_features': ['sqrt', 'log2'],
        #     'random_state': [42]
        # },
        # 'SVC': {
        #     'C': [0.1, 1.0, 10.0],
        #     'kernel': ['linear', 'rbf'],
        #     'gamma': ['scale', 'auto']
        # },
        # 'Logistic Regression': {
        #     'C': [0.1, 1.0, 10.0],
        #     'solver': ['liblinear', 'saga']
        # },
        # 'XGBoost': {
        #     'n_estimators': [100, 200],
        #     'max_depth': [3, 5],
        #     'learning_rate': [0.01, 0.05],
        #     'subsample': [0.8, 0.9],
        #     'colsample_bytree': [0.8, 0.9]
        # }
    }
    return param_grid

# 使用 GridSearchCV 自动调参
def tune_hyperparameters(models, param_grid, X_train, y_train, k_fold):
    tuned_models = {}
    for name, model in models.items():
        print(f"\n开始调参 {name} 模型...")
        # 创建 GridSearchCV 对象，verbose为1 隐藏调参过程 为2输出调参过程
        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=param_grid[name], 
            cv=k_fold, 
            scoring='accuracy',  # 使用准确率作为评估标准
            n_jobs=-1, 
            verbose=1
        )
        
        # 执行网格搜索
        grid_search.fit(X_train, y_train)
        
        # 获取最佳参数和最佳模型
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        print(f"{name} 最佳参数: {best_params}")
        print(f"{name} 最佳得分: {grid_search.best_score_:.4f}")
        
        tuned_models[name] = best_model
        
    return tuned_models

# 计算评估指标
def calculate_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Classification Report': classification_report(y_true, y_pred),
        'Confusion Matrix': confusion_matrix(y_true, y_pred)
    }

# 更新模型训练和评估函数，使用调参后的模型
def train_and_evaluate_with_cv(models, X_train, X_test, y_train, y_test, k_fold):
    all_results = {}
    kfold = KFold(n_splits=k_fold, shuffle=True, random_state=42)  # 使用n折交叉验证
    for name, model in models.items():
        print(f"\n开始训练{name}模型...")
        
        # 使用交叉验证评估模型
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)
        
        # 输出每一折的准确率评分
        print(f"{name} 每折的准确率: {cv_results}")
        
        # 保存平均得分
        mean_cv_score = np.mean(cv_results)
        
        # 训练模型并获取预测结果
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # 计算评估指标
        train_metrics = calculate_metrics(y_train, y_train_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)
        
        # 保存训练好的模型
        save_model(model, name)
        
        all_results[name] = {
            'cv_scores': cv_results,
            'mean_accuracy': mean_cv_score,
            'train_pred': y_train_pred,
            'test_pred': y_test_pred,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_true': y_train,  # 存储真实标签
            'test_true': y_test    # 存储真实标签
        }
        
        print(f"{name} - 平均交叉验证准确率: {mean_cv_score:.4f}")
        print(f"{name} - 训练集评估指标：")
        for metric, value in train_metrics.items():
            print(f"{metric}: {value}")
        print(f"{name} - 测试集评估指标：")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value}")
    return all_results

# 保存模型
def save_model(model, model_name):
    # 创建 trained_models 文件夹（如果它不存在）
    folder_path = 'trained_models'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # 创建文件夹

    # 保存训练好的模型到指定路径
    model_path = os.path.join(folder_path, f"{model_name}_model.pkl")
    joblib.dump(model, model_path)
    print(f"{model_name} 模型已保存到 {model_path}")

def save_results_to_excel(all_results, selected_features,models, file_name="模型比较结果.xlsx"):
    results_summary = []
    
    # 循环遍历每个模型
    for model_name, results in all_results.items():
        # 获取训练集结果
        train_metrics = results['train_metrics']
        y_train_pred = results['train_pred']
        y_train_true = results['train_true']  # 假设你有训练集真实标签
        
        # 计算 precision, recall, f1-score
        precision, recall, f1, support = precision_recall_fscore_support(y_train_true, y_train_pred, average='weighted')
        accuracy = accuracy_score(y_train_true, y_train_pred)
        
        # 添加训练集结果
        results_summary.append({
            'Model': model_name, 'Dataset': '训练集', 'Accuracy': accuracy,
            'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'Support': support
        })
        
        # 获取测试集结果
        test_metrics = results['test_metrics']
        y_test_pred = results['test_pred']
        y_test_true = results['test_true']  # 假设你有测试集真实标签
        
        # 计算 precision, recall, f1-score
        precision, recall, f1, support = precision_recall_fscore_support(y_test_true, y_test_pred, average='weighted')
        accuracy = accuracy_score(y_test_true, y_test_pred)
        
        # 添加测试集结果
        results_summary.append({
            'Model': model_name, 'Dataset': '测试集', 'Accuracy': accuracy,
            'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'Support': support
        })
    
    # 将结果保存到DataFrame
    results_df = pd.DataFrame(results_summary)
    
    # 保存到Excel文件
    with pd.ExcelWriter(file_name) as writer:
        results_df.to_excel(writer, sheet_name='模型评估指标', index=False)

        # 遍历所有模型，处理特征重要性
        for name, model in models.items():
            # 确保模型已经训练
            if hasattr(model, 'feature_importances_'):  # 检查模型是否有feature_importances_
                if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
                    # 获取特征重要性并保存
                    feature_importance = pd.DataFrame({
                        'feature': selected_features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    # 保存到Excel
                    feature_importance.to_excel(
                        writer, sheet_name=f'{name}_特征重要性', index=False
                    )

# 可视化模型的准确度对比
def plot_accuracy_comparison(all_results):
    models = list(all_results.keys())
    train_accuracies = [results['train_metrics']['Accuracy'] for results in all_results.values()]
    test_accuracies = [results['test_metrics']['Accuracy'] for results in all_results.values()]

    # 创建准确度对比图
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(models))

    # 使用seaborn样式
    sns.set(style="whitegrid")
    
    # 绘制训练集和测试集的准确度
    bars_train = plt.bar(index, train_accuracies, bar_width, label='Train Accuracy', color='#1f77b4')  # 蓝色
    bars_test = plt.bar(index + bar_width, test_accuracies, bar_width, label='Test Accuracy', color='#ff7f0e')  # 橙色

    # 添加数值标签
    for bar in bars_train:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 2), ha='center', va='bottom', fontsize=12, weight='bold')
    
    for bar in bars_test:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 2), ha='center', va='bottom', fontsize=12, weight='bold')

    # 添加标题和标签
    plt.xlabel('Models', fontsize=14, weight='bold')
    plt.ylabel('Accuracy', fontsize=14, weight='bold')
    plt.title('Model Accuracy Comparison', fontsize=16, weight='bold')
    plt.xticks(index + bar_width / 2, models, rotation=0, ha="center", fontsize=12, weight='bold')
    plt.yticks(fontsize=12)
    plt.legend(loc='best', fontsize=12)
    
    # 显示图表
    plt.tight_layout()
    file_path = os.path.join('results', f'accurarcy_comparison.png')
    plt.savefig(file_path, dpi=500, bbox_inches='tight')
    plt.close()

# 可视化混淆矩阵
def plot_confusion_matrix(y_true, y_pred, model_name, labels=None):
    # 定义数字标签到情绪标签的映射
    label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    # 将数字标签转换为情绪标签
    y_true_labels = [label_mapping[label] for label in y_true]
    y_pred_labels = [label_mapping[label] for label in y_pred]
    
   
    if labels is None:
        labels = ['negative', 'neutral', 'positive']  # 默认标签
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)

    # 使用seaborn样式
    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, 
                cbar_kws={'label': 'Number of Samples'}, annot_kws={'size': 14, 'weight': 'bold'})

    # 图表美化
    plt.title(f'{model_name} Confusion Matrix', fontsize=16, weight='bold')
    plt.xlabel('Predicted Labels', fontsize=14, weight='bold')
    plt.ylabel('True Labels', fontsize=14, weight='bold')
    plt.xticks(fontsize=12, weight='bold')
    plt.yticks(fontsize=12, weight='bold')
    plt.tight_layout()

    # 创建results文件夹（如果不存在）
    if not os.path.exists('results'):
        os.makedirs('results')

    # 保存混淆矩阵图像到results文件夹
    confusion_matrix_file_path = os.path.join('results', f'{model_name.lower()}_confusion_matrix.png')
    plt.savefig(confusion_matrix_file_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_pred, model_name, n_classes):
    # 对多分类标签进行二进制化
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    # 用于存储每个类别的 FPR, TPR 和 AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 获取每个类别的概率预测值
    if len(y_pred.shape) == 1:  # 只有一列（类别预测），需要转化为概率
        y_pred = label_binarize(y_pred, classes=np.arange(n_classes))

    # 为每个类别计算 ROC 曲线
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 使用seaborn样式
    sns.set(style="whitegrid")

    # 绘制所有类别的 ROC 曲线
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    # 绘制对角线，表示随机分类器的 ROC 曲线
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    # 图表美化
    plt.xlabel('False Positive Rate', fontsize=14, weight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, weight='bold')
    plt.title(f'ROC Curve for {model_name}', fontsize=16, weight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()

    # 创建results文件夹（如果不存在）
    if not os.path.exists('results'):
        os.makedirs('results')

    # 保存ROC曲线到results文件夹
    roc_file_path = os.path.join('results', f'{model_name.lower()}_roc_curve.png')
    plt.savefig(roc_file_path, dpi=300, bbox_inches='tight')
    plt.close()


# 可视化所有模型的评估结果
def visualize_results(all_results, y_train, y_test, n_classes):
    # 1. 准确度对比
    plot_accuracy_comparison(all_results)

    # 2. 混淆矩阵和ROC曲线
    for model_name, results in all_results.items():
        # 混淆矩阵
        print(f"绘制 {model_name} 模型的混淆矩阵...")
        plot_confusion_matrix(y_test, results['test_pred'], model_name)

        # ROC曲线
        print(f"绘制 {model_name} 模型的ROC曲线...")
        plot_roc_curve(y_test, results['test_pred'], model_name, n_classes)

# SHAP值计算与可视化
def calculate_shap_values(models, X_train_selected, selected_features, y_train):
    for name, model in models.items():
        print(f"\n正在计算 {name} 模型的 SHAP 值...")
        try:
            # 确保模型已训练
            if not hasattr(model, 'coef_') and not hasattr(model, 'feature_importances_'):
                model.fit(X_train_selected, y_train)
            
            # 对于树结构的模型（如 RF 和 XGBoost），使用 TreeExplainer
            if hasattr(model, 'feature_importances_') or name in ['RF', 'XGBoost']:
                # 使用 TreeExplainer 计算 SHAP 值
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_train_selected)
            
            # 对于线性模型（如 Ridge、Lasso、SVM），使用 KernelExplainer
            else:
                explainer = shap.KernelExplainer(model.predict, X_train_selected)
                shap_values = explainer.shap_values(X_train_selected)
            # 绘制 SHAP 值汇总图
            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                shap_values, X_train_selected, feature_names=selected_features,
                max_display=10, show=False
            )
            plt.title(f'{name} Model SHAP Summary Plot')
            plt.tight_layout()
            
            shap_file_path = os.path.join('./results', f'{name.lower()}_shap_summary.png')
            
            plt.savefig(shap_file_path, dpi=500, bbox_inches='tight')
            plt.close()

            # 打印特征数量验证
            print(f"使用的特征总数: {len(selected_features)}")
            print("选择的特征如下:")
            for i, feat in enumerate(selected_features):
                print(f"{i+1}. {feat}")
        except Exception as e:
            print(f"无法计算 {name} 模型的 SHAP 值。错误信息: {e}")
            
# 主函数
def main():
    # 参数
    max_features = 6  # 选择的特征数量
    k_folds = 4     # K折交叉验证的折数
    
    # 文件路径
    file_path = r"C:\Users\Lhtooo\Desktop\data\总结.xlsx"
    
    # 数据加载并划分为训练集和测试集
    X_train, X_test, y_train, y_test, label_encoder = load_and_split_data(file_path)
    # 数据预处理
    X_train = create_features(X_train)
    X_test = create_features(X_test)

    # 特征选择与缩放
    X_train_selected, X_test_selected, selected_features = preprocess_data(X_train, X_test, y_train, max_features)
    # 模型初始化
    models = initialize_models()
    # 获取参数网格
    param_grid = get_param_grid()
    # 自动调参
    tuned_models = tune_hyperparameters(models, param_grid, X_train_selected, y_train, k_folds)
    # 模型训练与评估
    all_results = train_and_evaluate_with_cv(tuned_models, X_train_selected, X_test_selected, y_train, y_test, k_folds)
    # 保存结果到Excel
    save_results_to_excel(all_results, selected_features, models)
    print("所有模型训练和评估完成，结果已保存。")
    # 可视化结果
    visualize_results(all_results, y_train, y_test,3)
    # 计算SHAP值
    calculate_shap_values(tuned_models, X_train_selected, selected_features, y_train)
    
if __name__ == "__main__":
    main()
