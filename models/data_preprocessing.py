"""
Data Preprocessing Module
Handles data loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Advanced data preprocessing for pricing optimization"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = []
        
    def load_data(self, filepath):
        """Load data from CSV or Excel"""
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        else:
            return pd.read_excel(filepath)
    
    def get_preview(self, data):
        """Get data preview for frontend"""
        preview = data.head(10).to_dict('records')
        stats = {
            'total_rows': len(data),
            'missing_values': data.isnull().sum().to_dict(),
            'dtypes': data.dtypes.astype(str).to_dict()
        }
        return {'data': preview, 'stats': stats}
    
    def preprocess(self, data):
        """Clean and preprocess data"""
        df = data.copy()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        if len(numeric_columns) > 0:
            df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])
        
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        # Encode categorical variables
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        # Remove outliers using IQR method
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        
        return df
    
    def engineer_features(self, data):
        """Create advanced features for pricing optimization"""
        df = data.copy()
        
        # Identify required columns (flexible naming)
        price_col = self._find_column(df, ['price', 'selling_price', 'unit_price', 'current_price'])
        cost_col = self._find_column(df, ['cost', 'cost_price', 'purchase_price', 'buying_price'])
        quantity_col = self._find_column(df, ['quantity', 'units_sold', 'sales_quantity', 'qty'])
        revenue_col = self._find_column(df, ['revenue', 'total_sales', 'sales_amount'])
        
        # Calculate profit metrics
        if price_col and cost_col:
            df['profit_per_unit'] = df[price_col] - df[cost_col]
            df['profit_margin'] = (df['profit_per_unit'] / df[price_col] * 100).clip(0, 100)
            df['profit_margin_normalized'] = df['profit_margin'] / 100
        
        # Calculate demand indicators
        if quantity_col:
            df['demand_level'] = pd.qcut(df[quantity_col].rank(method='first'), 
                                         q=5, labels=False, duplicates='drop') / 4
            df['demand_score'] = df['demand_level'] * 100
        
        # Calculate revenue efficiency
        if revenue_col and quantity_col:
            df['revenue_per_unit'] = df[revenue_col] / (df[quantity_col] + 1)
            df['revenue_efficiency'] = df['revenue_per_unit'] / df['revenue_per_unit'].max()
        
        # Customer satisfaction proxy
        if quantity_col and revenue_col:
            df['satisfaction_score'] = (
                (df[quantity_col] / df[quantity_col].max() * 0.4) +
                (df[revenue_col] / df[revenue_col].max() * 0.3) +
                (np.random.uniform(0.7, 1.0, len(df)) * 0.3)
            ) * 100
            df['satisfaction_score'] = df['satisfaction_score'].clip(0, 100)
        
        # Market competition indicators
        if price_col:
            df['price_competitiveness'] = 1 - (df[price_col] / df[price_col].max())
            df['price_position'] = pd.qcut(df[price_col].rank(method='first'),
                                          q=3, labels=['Low', 'Medium', 'High'])
        
        # Add default values if columns are missing
        if 'demand_level' not in df.columns:
            df['demand_level'] = 0.5
        
        if 'profit_margin_normalized' not in df.columns:
            df['profit_margin_normalized'] = 0.3
        
        if 'satisfaction_score' not in df.columns:
            df['satisfaction_score'] = 75.0
        
        if 'demand_score' not in df.columns:
            df['demand_score'] = 50.0
        
        return df
    
    def _find_column(self, df, possible_names):
        """Find column by possible names"""
        for name in possible_names:
            if name in df.columns:
                return name
            matches = [col for col in df.columns if col.lower() == name.lower()]
            if matches:
                return matches[0]
        return None