import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


class EmotionPrediction:
    def __init__(self, features):
        self.features = features
        # Initialize models
        self.models = self.initialize_models()
        # Initialize a dictionary to store the trained models
        self.fitted_models = {}

    def initialize_models(self):
        # Initialize models with predefined parameters
        models = {
            'RF': RandomForestClassifier(max_depth=5, max_features='sqrt', min_samples_leaf=1, min_samples_split=5, n_estimators=100, random_state=42),
            'SVC': SVC(C=0.1, kernel='linear', gamma='scale', probability=True),
            'Logistic Regression': LogisticRegression(C=0.1, solver='saga', multi_class='ovr'),
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8)
        }
        return models

    def preprocess_features(self, features):
        # Extract numerical features for prediction
        feature_data = [
            features['Positive'],  # Positive feature
            features['Neutral'],   # Neutral feature
            features['Negative'],  # Negative feature
            features['blink_count'], 
            features['fixations'], 
            features['saccades'],
            features['static_entropy'],
            features['transition_entropy'],
            features['std_diff_left'],
            features['std_diff_right']
        ]
        
        # Format features to be compatible with model input
        return np.array(feature_data).reshape(1, -1)

    def predict(self):
        # Preprocess the input features
        X_new = self.preprocess_features(self.features)

        # Standardize the feature data
        scaler = RobustScaler()
        X_new_scaled = scaler.fit_transform(X_new)

        # Store predictions for all models
        predictions = {}

        # Predict with each model
        for model_name, model in self.models.items():
            # Check if the model has already been trained
            if model_name not in self.fitted_models:
                # If not, fit the model with the data
                model.fit(X_new_scaled, [self.features['Positive'], self.features['Neutral'], self.features['Negative']])
                # Save the trained model
                self.fitted_models[model_name] = model
            
            # Make predictions with the trained model
            prediction = model.predict(X_new_scaled)  # Get prediction results
            
            # Map the prediction to the emotion label
            predicted_label = self.get_emotion_label(prediction[0])  # Get the predicted emotion label
            predictions[model_name] = predicted_label  # Save the prediction result
            print(f"{model_name} 情绪预测结果: {predicted_label}")  # Display the result

        return predictions

    def get_emotion_label(self, predicted_class):
        # Map the numerical prediction to emotion labels
        if predicted_class == 0:
            return 'negative'  # Negative emotion
        elif predicted_class == 1:
            return 'neutral'   # Neutral emotion
        elif predicted_class == 2:
            return 'positive'  # Positive emotion
        else:
            return 'unknown'   # In case of an unknown label

# Example usage:

# Assuming you have a dictionary of features
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

# Create emotion prediction object
emotion_predictor = EmotionPrediction(new_features)

# Perform prediction
predictions = emotion_predictor.predict()

# Print all model predictions
print(predictions)

