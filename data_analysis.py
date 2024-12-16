import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class BigMarketAnalysis:
    def __init__(self, file_path='bigmarket.csv', output_dir='results'):
        """
        Initialize the analysis with data loading and initial setup
        """
        # Data Loading and Initial Setup
        self.df = pd.read_csv(file_path)
        self.original_columns = self.df.columns.tolist()
        
        # Suppress Warnings
        import warnings
        warnings.filterwarnings('ignore')
        
        # Set Seaborn Style
        sns.set_style('whitegrid')
    
    def data_overview(self):
        """
        Provide comprehensive overview of the dataset
        """
        print("Dataset Shape:", self.df.shape)
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nMissing Values:")
        print(self.df.isnull().sum())
    
    def handle_missing_values(self):
        """
        Comprehensive method to handle missing values
        """
        # Handle Item Weight Missing Values
        self.df['Item_Weight'].fillna(self.df['Item_Weight'].mean(), inplace=True)
        
        # Handle Outlet Size Missing Values
        mode_of_outlet_size = self.df.pivot_table(
            values='Outlet_Size', 
            columns='Outlet_Type', 
            aggfunc=(lambda x: x.mode()[0])
        )
        
        missing_outlet_size = self.df['Outlet_Size'].isnull()
        self.df.loc[missing_outlet_size, 'Outlet_Size'] = \
            self.df.loc[missing_outlet_size, 'Outlet_Type'].apply(lambda x: mode_of_outlet_size[x])
        
        print("\nMissing Values After Handling:")
        print(self.df.isnull().sum())
    
    def encode_categorical_features(self):
        """
        Encode categorical features
        """
        # Label Encoding for categorical columns
        categorical_columns = [
            'Item_Identifier', 'Item_Fat_Content', 'Item_Type', 
            'Outlet_Identifier', 'Outlet_Size', 
            'Outlet_Location_Type', 'Outlet_Type'
        ]
        
        label_encoder = LabelEncoder()
        for col in categorical_columns:
            self.df[col] = label_encoder.fit_transform(self.df[col])
        
        return self.df
    
    def prepare_data_for_ml(self):
        """
        Prepare data for Machine Learning
        """
        # Separate features and target
        X = self.df.drop('Item_Outlet_Sales', axis=1)
        y = self.df['Item_Outlet_Sales']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        Train Random Forest Regressor
        """
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        return rf_model
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the model's performance
        """
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\nModel Performance:")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared Score: {r2}")
        
        return y_pred
    
    def visualize_results(self, y_test, y_pred, model):
        """
        Create comprehensive visualizations and display them
        """
        # 1. Actual vs Predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title('Actual vs Predicted Sales')
        plt.xlabel('Actual Sales')
        plt.ylabel('Predicted Sales')
        plt.show()  # Display the plot
        
        # 2. Feature Importance
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0])
        
        plt.figure(figsize=(8, 6))
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, [self.original_columns[i] for i in sorted_idx])
        plt.title('Feature Importance')
        plt.show()  # Display the plot
        
        # 3. Error Distribution
        errors = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.hist(errors, bins=30, color="purple", alpha=0.7)
        plt.title('Error Distribution')
        plt.xlabel('Errors')
        plt.ylabel('Frequency')
        plt.show()  # Display the plot
        
        # 4. Residual Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, errors, alpha=0.5, color="orange")
        plt.title('Residual Plot')
        plt.xlabel('Predicted Sales')
        plt.ylabel('Errors')
        plt.show()  # Display the plot
    
    def run_full_analysis(self):
        """
        Run complete analysis pipeline
        """
        print("Starting Big Market Sales Prediction Analysis")
        
        # Data Overview
        self.data_overview()
        
        # Handle Missing Values
        self.handle_missing_values()
        
        # Encode Categorical Features
        self.encode_categorical_features()
        
        # Prepare Data for ML
        X_train, X_test, y_train, y_test = self.prepare_data_for_ml()
        
        # Train Model
        model = self.train_model(X_train, y_train)
        
        # Evaluate Model
        y_pred = self.evaluate_model(model, X_test, y_test)
        
        # Visualize Results
        self.visualize_results(y_test, y_pred, model)

# Main Execution
if __name__ == "__main__":
    market_analysis = BigMarketAnalysis()
    market_analysis.run_full_analysis()
