import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Add this at the top after imports
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def explore_dataset(df):
    print("\n=== Dataset Exploration ===")
    
    # Basic information
    print("\nBasic Information:")
    print(df.info())
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Value distributions for categorical variables
    print("\nCategorical Variable Distributions:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col} distribution:")
        print(df[col].value_counts(normalize=True).round(3))
    
    # Correlation analysis for numerical variables
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix of Numerical Features")
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

# Preprocess the data
def preprocess_data(df):
    # Separate features and target
    X = df.drop('Overall', axis=1)
    y = df['Overall']
    
    # Convert categorical variables to numeric using one-hot encoding
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

# Train the model
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
    
    # Visualize predictions vs actual values
    plt.figure(figsize=(10, 5))
    
    # Training set
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Training Set: Predicted vs Actual')
    
    # Test set
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Test Set: Predicted vs Actual')
    
    plt.tight_layout()
    plt.savefig('prediction_accuracy.png')
    plt.close()
    
    return model, metrics, (y_test, y_test_pred)

# Calculate and visualize SHAP values
def analyze_shap(model, X_train, X_test, feature_names):
    # Set the backend to avoid potential M1/M2 issues
    plt.switch_backend('Agg')
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()
    
    # Feature importance bar plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar Plot)")
    plt.tight_layout()
    plt.savefig('shap_importance.png')
    plt.close()
    
    # Calculate and return average absolute SHAP values for feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    return feature_importance, explainer, shap_values

# Analyze specific predictions
def analyze_predictions(model, explainer, X_test, feature_names):
    # Set the backend to avoid potential M1/M2 issues
    plt.switch_backend('Agg')
    
    # Get SHAP values for a single prediction
    sample_idx = 0
    shap_values = explainer.shap_values(X_test[sample_idx:sample_idx+1])
    
    # Force plot for single prediction
    plt.figure(figsize=(15, 3))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_test[sample_idx:sample_idx+1],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title("SHAP Force Plot for Single Prediction")
    plt.tight_layout()
    plt.savefig('shap_force_plot.png')
    plt.close()

def main():
    # Load data
    print("Loading and exploring dataset...")
    df = pd.read_csv('dataset.csv')
    
    # Explore dataset
    explore_dataset(df)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    
    # Train and evaluate model
    print("\nTraining and evaluating model...")
    model, metrics, (y_test, y_test_pred) = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    # Print model performance metrics
    print("\n=== Model Performance ===")
    print("\nAccuracy Metrics:")
    print(f"Training Accuracy: {metrics['Train Accuracy']:.2%}")
    print(f"Testing Accuracy: {metrics['Test Accuracy']:.2%}")
    
    print("\nRegression Metrics:")
    print(f"Training RMSE: {metrics['Train RMSE']:.3f}")
    print(f"Testing RMSE: {metrics['Test RMSE']:.3f}")
    print(f"Training R²: {metrics['Train R²']:.3f}")
    print(f"Testing R²: {metrics['Test R²']:.3f}")
    
    # Calculate and display error distribution
    errors = np.abs(y_test - y_test_pred)
    print("\nError Distribution:")
    print(f"Mean Absolute Error: {errors.mean():.3f}")
    print(f"Median Absolute Error: {np.median(errors):.3f}")
    print(f"90th Percentile Error: {np.percentile(errors, 90):.3f}")
    
    # SHAP analysis
    print("\nPerforming SHAP analysis...")
    feature_importance, explainer, shap_values = analyze_shap(model, X_train, X_test, feature_names)
    
    # Print top 10 most important features
    print("\n=== Top 10 Most Important Features ===")
    print(feature_importance.head(10))
    
    # Analyze specific predictions
    analyze_predictions(model, explainer, X_test, feature_names)
    
    print("\nAnalysis complete! Generated visualization files:")
    print("- correlation_matrix.png")
    print("- prediction_accuracy.png")
    print("- shap_summary.png")
    print("- shap_importance.png")
    print("- shap_force_plot.png")

if __name__ == "__main__":
    main() 