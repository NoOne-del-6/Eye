import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib  # 用于加载已保存的模型

class EmotionPrediction:
    def __init__(self, features):
        # 初始化特征数据
        self.features = features
        # 加载已训练好的模型
        self.models = self.load_trained_models()

    def load_trained_models(self):
        # 从trained_models文件夹加载已训练好的模型
        models = {
            'RF': joblib.load('trained_models/RF_model.pkl'),
            'SVC': joblib.load('trained_models/SVC_model.pkl'),
            'Logistic Regression': joblib.load('trained_models/Logistic Regression_model.pkl'),
            'XGBoost': joblib.load('trained_models/XGBoost_model.pkl')
        }
        return models

    def preprocess_features(self, features):
        # 提取数值特征以供预测使用
        feature_data = [
            features['Positive'],  # Positive 特征
            features['Neutral'],   # Neutral 特征
            features['Negative'],  # Negative 特征
            features['blink_count'], 
            features['fixations'], 
            features['saccades'],
            features['static_entropy'],
            features['transition_entropy'],
            features['std_diff_left'],
            features['std_diff_right']
        ]
        
        # 将特征转换为模型可接受的格式
        return np.array(feature_data).reshape(1, -1)

    def predict(self):
        # 预处理输入特征
        X_new = self.preprocess_features(self.features)

        # 标准化特征数据
        scaler = RobustScaler()
        X_new_scaled = scaler.fit_transform(X_new)

        # 存储所有模型的预测结果
        predictions = {}

        # 对每个加载的模型进行预测
        for model_name, model in self.models.items():
            # 使用加载的训练好的模型进行预测
            prediction = model.predict(X_new_scaled)  # 获取预测结果
            
            # 将预测结果映射为情绪标签
            predicted_label = self.get_emotion_label(prediction[0])  # 获取情绪预测标签
            predictions[model_name] = predicted_label  # 保存预测结果
            print(f"{model_name} 情绪预测结果: {predicted_label}")  # 显示预测结果

        return predictions

    def get_emotion_label(self, predicted_class):
        # 将数字预测标签转换为情绪标签
        if predicted_class == 0:
            return 'negative'  # 负面情绪
        elif predicted_class == 1:
            return 'neutral'   # 中性情绪
        elif predicted_class == 2:
            return 'positive'  # 正面情绪
        else:
            return 'unknown'   # 未知标签

# 示例使用：

# 假设你有一个包含特征的字典
new_features = {
    'Positive': 0.32749465974977116,
    'Neutral': 0.4366188587122368,
    'Negative': 0.23588648153799208,
    'blink_count': 7,
    'fixations': 211,
    'saccades': 603,
    'static_entropy': 1.541,
    'transition_entropy': 71.3861,
    'std_diff_left': 0.1755,
    'std_diff_right': 0.2656
}

# 创建情绪预测对象
emotion_predictor = EmotionPrediction(new_features)

# 执行预测
predictions = emotion_predictor.predict()

# 打印所有模型的预测结果
print(predictions)
