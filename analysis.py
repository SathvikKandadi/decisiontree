import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.metrics import mean_squared_error, r2_score

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

class StudentPerformancePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.explainer = None
        
    def train(self, df):
        # Separate features and target
        X = df.drop('Overall', axis=1)
        y = df['Overall']
        
        # Save original feature names
        self.original_features = X.columns.tolist()
        
        # Prepare categorical encoders
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Train model
        self.model = DecisionTreeRegressor(random_state=42, max_depth=5)
        self.model.fit(X_scaled, y)
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        # Save feature names
        self.feature_names = X.columns
        
        return self
    
    def predict_student_performance(self, student_data):
        """
        Predict performance for a single student and provide detailed analysis
        """
        # Convert student data to DataFrame if it's a dictionary
        if isinstance(student_data, dict):
            student_data = pd.DataFrame([student_data])
        
        # Encode categorical variables
        student_processed = student_data.copy()
        for col, encoder in self.label_encoders.items():
            student_processed[col] = encoder.transform(student_processed[col])
        
        # Scale features
        student_scaled = self.scaler.transform(student_processed)
        
        # Make prediction
        prediction = self.model.predict(student_scaled)[0]
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(student_scaled)
        
        # Get feature importance for this prediction
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values[0])
        }).sort_values('importance', ascending=False)
        
        # Determine strengths and weaknesses
        positive_impact = []
        negative_impact = []
        
        for idx, row in feature_importance.head(5).iterrows():
            feature = row['feature']
            shap_value = shap_values[0][idx]
            if shap_value > 0:
                positive_impact.append((feature, shap_value))
            else:
                negative_impact.append((feature, abs(shap_value)))
        
        # Generate analysis report
        analysis = {
            'predicted_grade': round(prediction, 2),
            'performance_level': self._get_performance_level(prediction),
            'strengths': positive_impact,
            'areas_for_improvement': negative_impact,
            'feature_importance': feature_importance
        }
        
        return analysis
    
    def _get_performance_level(self, grade):
        if grade >= 3.7:
            return "Excellent"
        elif grade >= 3.3:
            return "Very Good"
        elif grade >= 3.0:
            return "Good"
        elif grade >= 2.7:
            return "Satisfactory"
        else:
            return "Needs Improvement"
    
    def save_model(self, path='student_predictor.joblib'):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'original_features': self.original_features
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load_model(cls, path='student_predictor.joblib'):
        """Load a trained model"""
        predictor = cls()
        model_data = joblib.load(path)
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.label_encoders = model_data['label_encoders']
        predictor.feature_names = model_data['feature_names']
        predictor.original_features = model_data['original_features']
        predictor.explainer = shap.TreeExplainer(predictor.model)
        return predictor

def visualize_prediction(analysis):
    """Create visualization for the prediction analysis"""
    plt.switch_backend('Agg')
    
    # Feature importance plot
    plt.figure(figsize=(10, 6))
    importance_df = analysis['feature_importance'].head(10)
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Top 10 Factors Influencing the Prediction')
    plt.tight_layout()
    plt.savefig('student_analysis.png')
    plt.close()

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Initialize and train the model
    model = DecisionTreeRegressor(random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'Train R²': r2_score(y_train, y_train_pred),
        'Test R²': r2_score(y_test, y_test_pred),
        'Train Accuracy': 1 - (np.abs(y_train - y_train_pred) / y_train).mean(),
        'Test Accuracy': 1 - (np.abs(y_test - y_test_pred) / y_test).mean()
    }
    
    # Save the model and metrics
    joblib.dump(model, 'trained_model.joblib')
    joblib.dump(metrics, 'model_metrics.joblib')
    
    return model, metrics, (y_test, y_test_pred)

def main():
    # Load and prepare data
    print("Loading dataset and training model...")
    df = pd.read_csv('dataset.csv')
    
    # Preprocess data
    X = df.drop('Overall', axis=1)
    y = df['Overall']
    
    # Convert categorical variables to numeric
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate model
    model, metrics, _ = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    # Initialize and train predictor
    predictor = StudentPerformancePredictor()
    predictor.train(df)
    
    # Save the trained model
    predictor.save_model()
    
    # Print model performance
    print("\n=== Model Performance ===")
    print(f"Training Accuracy: {metrics['Train Accuracy']:.2%}")
    print(f"Testing Accuracy: {metrics['Test Accuracy']:.2%}")
    print(f"Training RMSE: {metrics['Train RMSE']:.3f}")
    print(f"Testing RMSE: {metrics['Test RMSE']:.3f}")
    print(f"Training R²: {metrics['Train R²']:.3f}")
    print(f"Testing R²: {metrics['Test R²']:.3f}")
    
    # Save metrics
    joblib.dump(metrics, 'model_metrics.joblib')
    
    print("\nModel and metrics saved successfully!")

if __name__ == "__main__":
    main()