import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

class StudentPredictor:
    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
    
    def predict_and_explain(self, student_data):
        # Scale the features using the provided scaler
        student_scaled = self.scaler.transform(student_data)
        
        # Make prediction
        prediction = self.model.predict(student_scaled)[0]
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(student_scaled)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values[0])
        }).sort_values('importance', ascending=False)
        
        # Determine strengths and weaknesses with explanations
        positive_impact = []
        negative_impact = []
        
        for idx, row in feature_importance.head(5).iterrows():
            feature = row['feature']
            shap_value = shap_values[0][idx]
            
            # Add explanations based on the feature
            if feature == 'Gaming':
                if shap_value < 0:
                    explanation = f"Consider balancing gaming hours with study time"
                else:
                    explanation = f"Current gaming balance is good"
                
            elif feature == 'Attendance':
                if shap_value < 0:
                    explanation = f"Try to improve attendance"
                else:
                    explanation = f"Good attendance level"
                    
            elif feature == 'Preparation':
                if shap_value < 0:
                    explanation = f"Consider increasing study hours"
                else:
                    explanation = f"Good study habits"
                    
            elif feature == 'Last':
                if shap_value < 0:
                    explanation = f"Previous semester performance needs improvement"
                else:
                    explanation = f"Good previous performance"
                    
            else:
                explanation = f"Impact: {abs(shap_value):.3f}"
            
            if shap_value > 0:
                positive_impact.append((feature, abs(shap_value), explanation))
            else:
                negative_impact.append((feature, abs(shap_value), explanation))
        
        return {
            'predicted_grade': round(prediction, 2),
            'performance_level': self.get_performance_level(prediction),
            'strengths': positive_impact,
            'areas_for_improvement': negative_impact,
            'feature_importance': feature_importance
        }
    
    @staticmethod
    def get_performance_level(grade):
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

def get_user_input():
    print("\nPlease enter student information:")
    student = {}
    
    # Department options from dataset
    departments = [
        'Business Administration',
        'Computer Science and Engineering',
        'Economics',
        'Electrical and Electronic Engineering',
        'English',
        'Journalism, Communication and Media Studies',
        'Law and Human Rights',
        'Political Science',
        'Public Health',
        'Sociology'
    ]
    print("\nDepartment Options:")
    for i, dept in enumerate(departments, 1):
        print(f"{i}: {dept}")
    dept_choice = int(input("Choose department (1-10): "))
    student['Department'] = departments[dept_choice-1]
    
    # Gender
    gender_options = ['Male', 'Female']
    print("\nGender Options:")
    for i, gender in enumerate(gender_options, 1):
        print(f"{i}: {gender}")
    gender_choice = int(input("Choose gender (1-2): "))
    student['Gender'] = gender_options[gender_choice-1]
    
    # Income with exact format from dataset
    income_options = [
        'Low (Below 15,000)',
        'Lower middle (15,000-30,000)',
        'Upper middle (30,000-50,000)',
        'High (Above 50,000)'
    ]
    print("\nIncome Level Options:")
    for i, income in enumerate(income_options, 1):
        print(f"{i}: {income}")
    income_choice = int(input("Choose income level (1-4): "))
    student['Income'] = income_options[income_choice-1]
    
    # Hometown
    hometown_options = ['City', 'Village']
    print("\nHometown Options:")
    for i, hometown in enumerate(hometown_options, 1):
        print(f"{i}: {hometown}")
    hometown_choice = int(input("Choose hometown (1-2): "))
    student['Hometown'] = hometown_options[hometown_choice-1]
    
    # Study hours (Preparation)
    prep_options = ['0-1 Hour', '2-3 Hours', 'More than 3 Hours']
    print("\nStudy Hours Options:")
    for i, opt in enumerate(prep_options, 1):
        print(f"{i}: {opt}")
    prep_choice = int(input("Choose study hours (1-3): "))
    student['Preparation'] = prep_options[prep_choice-1]
    
    # Gaming hours
    gaming_options = ['0-1 Hour', '2-3 Hours', 'More than 3 Hours']
    print("\nGaming Hours Options:")
    for i, opt in enumerate(gaming_options, 1):
        print(f"{i}: {opt}")
    gaming_choice = int(input("Choose gaming hours (1-3): "))
    student['Gaming'] = gaming_options[gaming_choice-1]
    
    # Attendance
    attendance_options = ['Below 40%', '40%-59%', '60%-79%', '80%-100%']
    print("\nAttendance Options:")
    for i, opt in enumerate(attendance_options, 1):
        print(f"{i}: {opt}")
    attendance_choice = int(input("Choose attendance (1-4): "))
    student['Attendance'] = attendance_options[attendance_choice-1]
    
    # Job
    job_options = ['Yes', 'No']
    print("\nJob Status:")
    for i, opt in enumerate(job_options, 1):
        print(f"{i}: {opt}")
    job_choice = int(input("Has job? (1-2): "))
    student['Job'] = job_options[job_choice-1]
    
    # Extra curricular activities
    extra_options = ['Yes', 'No']
    print("\nExtra Curricular Activities:")
    for i, opt in enumerate(extra_options, 1):
        print(f"{i}: {opt}")
    extra_choice = int(input("Participates in extra activities? (1-2): "))
    student['Extra'] = extra_options[extra_choice-1]
    
    # Semester
    semester_options = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th', '11th', '12th']
    print("\nSemester Options:")
    for i, sem in enumerate(semester_options, 1):
        print(f"{i}: {sem}")
    sem_choice = int(input("Choose semester (1-12): "))
    student['Semester'] = semester_options[sem_choice-1]
    
    # Numerical inputs with range validation
    while True:
        try:
            hsc = float(input("\nHSC Score (0-5): "))
            if 0 <= hsc <= 5:
                student['HSC'] = hsc
                break
            print("Score must be between 0 and 5")
        except ValueError:
            print("Please enter a valid number")
    
    while True:
        try:
            ssc = float(input("SSC Score (0-5): "))
            if 0 <= ssc <= 5:
                student['SSC'] = ssc
                break
            print("Score must be between 0 and 5")
        except ValueError:
            print("Please enter a valid number")
    
    while True:
        try:
            computer = int(input("Computer Skills (1-5): "))
            if 1 <= computer <= 5:
                student['Computer'] = computer
                break
            print("Skills must be between 1 and 5")
        except ValueError:
            print("Please enter a valid number")
    
    while True:
        try:
            english = int(input("English Skills (1-5): "))
            if 1 <= english <= 5:
                student['English'] = english
                break
            print("Skills must be between 1 and 5")
        except ValueError:
            print("Please enter a valid number")
    
    while True:
        try:
            last = float(input("Last Semester GPA (0-4): "))
            if 0 <= last <= 4:
                student['Last'] = last
                break
            print("GPA must be between 0 and 4")
        except ValueError:
            print("Please enter a valid number")
    
    return student

def main():
    # Load the original dataset to get the feature names and scaler
    df = pd.read_csv('dataset.csv')
    X = df.drop('Overall', axis=1)
    
    # Load the trained model data
    model_data = joblib.load('student_predictor.joblib')
    metrics = joblib.load('model_metrics.joblib')
    
    # Extract the actual model and scaler from the loaded data
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    # Print model performance
    print("\n=== Model Performance ===")
    print(f"Training Accuracy: {metrics['Train Accuracy']:.2%}")
    print(f"Testing Accuracy: {metrics['Test Accuracy']:.2%}")
    print(f"Training RMSE: {metrics['Train RMSE']:.3f}")
    print(f"Testing RMSE: {metrics['Test RMSE']:.3f}")
    print(f"Training R²: {metrics['Train R²']:.3f}")
    print(f"Testing R²: {metrics['Test R²']:.3f}")
    
    # Get student data from user
    student_data = get_user_input()
    
    # Convert to DataFrame and ensure column order matches training data
    student_df = pd.DataFrame([student_data])
    
    # Get the original column order from the dataset
    original_columns = X.columns.tolist()
    
    # Reorder the columns to match the training data
    student_df = student_df[original_columns]
    
    # Process categorical variables
    for col in model_data['label_encoders'].keys():
        encoder = model_data['label_encoders'][col]
        student_df[col] = encoder.transform(student_df[col])
    
    # Initialize predictor with the actual model
    predictor = StudentPredictor(model, scaler, feature_names)
    
    # Get prediction and analysis
    analysis = predictor.predict_and_explain(student_df)
    
    # Print results
    print("\n=== Student Performance Analysis ===")
    print(f"\nPredicted Grade: {analysis['predicted_grade']}")
    print(f"Performance Level: {analysis['performance_level']}")
    
    print("\nStrengths:")
    for feature, impact, explanation in analysis['strengths']:
        print(f"- {feature}: {explanation}")
    
    print("\nAreas for Improvement:")
    for feature, impact, explanation in analysis['areas_for_improvement']:
        print(f"- {feature}: {explanation}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    importance_df = analysis['feature_importance'].head(10)
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Top 10 Factors Influencing the Prediction')
    plt.tight_layout()
    plt.savefig('student_analysis.png')
    plt.close()
    
    print("\nVisualization saved as 'student_analysis.png'")

if __name__ == "__main__":
    main() 