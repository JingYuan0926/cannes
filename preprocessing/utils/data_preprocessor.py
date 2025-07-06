import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    LabelEncoder, OneHotEncoder, 
    QuantileTransformer, PowerTransformer,
    PolynomialFeatures
)
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, chi2, f_classif, 
    mutual_info_classif, mutual_info_regression
)
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Comprehensive data preprocessing utility"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.transformers = {}
        self.preprocessing_history = []
    
    def standardize_numerical_features(self, df, columns='all', method='z_score', **kwargs):
        """Standardize numerical features using various methods"""
        df_copy = df.copy()
        
        if columns == 'all':
            columns = df_copy.select_dtypes(include=[np.number]).columns
        elif columns == 'numerical':
            columns = df_copy.select_dtypes(include=[np.number]).columns
        elif isinstance(columns, str):
            columns = [columns]
        
        # Filter out non-existent columns
        columns = [col for col in columns if col in df_copy.columns]
        
        if not columns:
            return df_copy
        
        try:
            if method == 'z_score' or method == 'standard':
                scaler = StandardScaler()
            elif method == 'min_max':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            elif method == 'quantile_uniform':
                scaler = QuantileTransformer(output_distribution='uniform')
            elif method == 'quantile_normal':
                scaler = QuantileTransformer(output_distribution='normal')
            else:
                scaler = StandardScaler()
            
            df_copy[columns] = scaler.fit_transform(df_copy[columns])
            self.scalers[f'{method}_scaler'] = scaler
            
            self.preprocessing_history.append({
                'operation': 'standardize_numerical_features',
                'method': method,
                'columns': columns,
                'parameters': kwargs
            })
            
        except Exception as e:
            print(f"Error in standardize_numerical_features: {str(e)}")
        
        return df_copy
    
    def normalize_categorical_features(self, df, columns='all', method='one_hot_encoding', **kwargs):
        """Normalize categorical features using various encoding methods"""
        df_copy = df.copy()
        
        if columns == 'all':
            columns = df_copy.select_dtypes(include=['object', 'category']).columns
        elif columns == 'categorical':
            columns = df_copy.select_dtypes(include=['object', 'category']).columns
        elif isinstance(columns, str):
            columns = [columns]
        
        # Filter out non-existent columns
        columns = [col for col in columns if col in df_copy.columns]
        
        if not columns:
            return df_copy
        
        try:
            for col in columns:
                if method == 'label_encoding':
                    le = LabelEncoder()
                    df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                    self.encoders[f'{col}_label_encoder'] = le
                
                elif method == 'one_hot_encoding':
                    # Get dummies and concatenate
                    dummies = pd.get_dummies(df_copy[col], prefix=col, dummy_na=True)
                    df_copy = pd.concat([df_copy.drop(col, axis=1), dummies], axis=1)
                
                elif method == 'frequency_encoding':
                    freq_map = df_copy[col].value_counts().to_dict()
                    df_copy[col] = df_copy[col].map(freq_map)
                
                elif method == 'binary_encoding':
                    # Simple binary encoding for binary categorical variables
                    if df_copy[col].nunique() == 2:
                        unique_vals = df_copy[col].unique()
                        df_copy[col] = df_copy[col].map({unique_vals[0]: 0, unique_vals[1]: 1})
                
                elif method == 'target_encoding':
                    # Placeholder for target encoding (requires target variable)
                    # For now, use frequency encoding as fallback
                    freq_map = df_copy[col].value_counts().to_dict()
                    df_copy[col] = df_copy[col].map(freq_map)
            
            self.preprocessing_history.append({
                'operation': 'normalize_categorical_features',
                'method': method,
                'columns': columns,
                'parameters': kwargs
            })
            
        except Exception as e:
            print(f"Error in normalize_categorical_features: {str(e)}")
        
        return df_copy
    
    def handle_text_features(self, df, columns='all', method='tfidf', max_features=100, **kwargs):
        """Handle text features using various NLP techniques"""
        df_copy = df.copy()
        
        if columns == 'all':
            # Detect text columns (object columns with high average string length)
            text_columns = []
            for col in df_copy.select_dtypes(include=['object']).columns:
                avg_length = df_copy[col].astype(str).str.len().mean()
                if avg_length > 20:  # Threshold for text data
                    text_columns.append(col)
            columns = text_columns
        elif isinstance(columns, str):
            columns = [columns]
        
        # Filter out non-existent columns
        columns = [col for col in columns if col in df_copy.columns]
        
        if not columns:
            return df_copy
        
        try:
            for col in columns:
                if method == 'tfidf':
                    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
                    text_features = vectorizer.fit_transform(df_copy[col].astype(str))
                    feature_names = [f'{col}_tfidf_{i}' for i in range(text_features.shape[1])]
                    text_df = pd.DataFrame(text_features.toarray(), columns=feature_names, index=df_copy.index)
                    df_copy = pd.concat([df_copy.drop(col, axis=1), text_df], axis=1)
                    self.transformers[f'{col}_tfidf'] = vectorizer
                
                elif method == 'count_vectorizer':
                    vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
                    text_features = vectorizer.fit_transform(df_copy[col].astype(str))
                    feature_names = [f'{col}_count_{i}' for i in range(text_features.shape[1])]
                    text_df = pd.DataFrame(text_features.toarray(), columns=feature_names, index=df_copy.index)
                    df_copy = pd.concat([df_copy.drop(col, axis=1), text_df], axis=1)
                    self.transformers[f'{col}_count'] = vectorizer
                
                elif method == 'text_preprocessing':
                    # Basic text preprocessing
                    df_copy[col] = df_copy[col].astype(str).str.lower()
                    df_copy[col] = df_copy[col].str.replace(r'[^\w\s]', '', regex=True)
                    df_copy[col] = df_copy[col].str.strip()
            
            self.preprocessing_history.append({
                'operation': 'handle_text_features',
                'method': method,
                'columns': columns,
                'parameters': kwargs
            })
            
        except Exception as e:
            print(f"Error in handle_text_features: {str(e)}")
        
        return df_copy
    
    def engineer_datetime_features(self, df, columns='all', extract=['year', 'month', 'day'], **kwargs):
        """Engineer features from datetime columns"""
        df_copy = df.copy()
        
        if columns == 'all':
            # Auto-detect datetime columns
            datetime_columns = []
            for col in df_copy.select_dtypes(include=['object']).columns:
                try:
                    pd.to_datetime(df_copy[col].dropna().head(10))
                    datetime_columns.append(col)
                except:
                    pass
            columns = datetime_columns
        elif isinstance(columns, str):
            columns = [columns]
        
        # Filter out non-existent columns
        columns = [col for col in columns if col in df_copy.columns]
        
        if not columns:
            return df_copy
        
        try:
            for col in columns:
                # Convert to datetime
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                
                # Extract features
                if 'year' in extract:
                    df_copy[f'{col}_year'] = df_copy[col].dt.year
                if 'month' in extract:
                    df_copy[f'{col}_month'] = df_copy[col].dt.month
                if 'day' in extract:
                    df_copy[f'{col}_day'] = df_copy[col].dt.day
                if 'hour' in extract:
                    df_copy[f'{col}_hour'] = df_copy[col].dt.hour
                if 'weekday' in extract:
                    df_copy[f'{col}_weekday'] = df_copy[col].dt.weekday
                if 'quarter' in extract:
                    df_copy[f'{col}_quarter'] = df_copy[col].dt.quarter
                if 'is_weekend' in extract:
                    df_copy[f'{col}_is_weekend'] = df_copy[col].dt.weekday >= 5
                if 'days_since_epoch' in extract:
                    df_copy[f'{col}_days_since_epoch'] = (df_copy[col] - pd.Timestamp('1970-01-01')).dt.days
                
                # Drop original column
                df_copy = df_copy.drop(col, axis=1)
            
            self.preprocessing_history.append({
                'operation': 'engineer_datetime_features',
                'extract': extract,
                'columns': columns,
                'parameters': kwargs
            })
            
        except Exception as e:
            print(f"Error in engineer_datetime_features: {str(e)}")
        
        return df_copy
    
    def create_polynomial_features(self, df, columns='all', degree=2, interaction_only=False, include_bias=False, **kwargs):
        """Create polynomial features"""
        df_copy = df.copy()
        
        if columns == 'all':
            columns = df_copy.select_dtypes(include=[np.number]).columns
        elif columns == 'numerical':
            columns = df_copy.select_dtypes(include=[np.number]).columns
        elif isinstance(columns, str):
            columns = [columns]
        
        # Filter out non-existent columns and limit to reasonable number
        columns = [col for col in columns if col in df_copy.columns][:5]  # Limit to 5 columns to avoid explosion
        
        if not columns or len(columns) == 0:
            return df_copy
        
        try:
            poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
            poly_features = poly.fit_transform(df_copy[columns])
            
            # Create feature names
            feature_names = poly.get_feature_names_out(columns)
            
            # Create DataFrame with polynomial features
            poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df_copy.index)
            
            # Drop original columns and add polynomial features
            df_copy = df_copy.drop(columns, axis=1)
            df_copy = pd.concat([df_copy, poly_df], axis=1)
            
            self.transformers['polynomial_features'] = poly
            
            self.preprocessing_history.append({
                'operation': 'create_polynomial_features',
                'degree': degree,
                'columns': columns,
                'parameters': kwargs
            })
            
        except Exception as e:
            print(f"Error in create_polynomial_features: {str(e)}")
        
        return df_copy
    
    def apply_feature_selection(self, df, columns='all', method='variance_threshold', k=10, **kwargs):
        """Apply feature selection techniques"""
        df_copy = df.copy()
        
        if columns == 'all':
            columns = df_copy.select_dtypes(include=[np.number]).columns
        elif columns == 'numerical':
            columns = df_copy.select_dtypes(include=[np.number]).columns
        elif isinstance(columns, str):
            columns = [columns]
        
        # Filter out non-existent columns
        columns = [col for col in columns if col in df_copy.columns]
        
        if not columns or len(columns) <= 1:
            return df_copy
        
        try:
            if method == 'variance_threshold':
                selector = VarianceThreshold(threshold=0.01)
                selected_features = selector.fit_transform(df_copy[columns])
                selected_columns = [col for col, selected in zip(columns, selector.get_support()) if selected]
                df_copy = df_copy.drop(columns, axis=1)
                selected_df = pd.DataFrame(selected_features, columns=selected_columns, index=df_copy.index)
                df_copy = pd.concat([df_copy, selected_df], axis=1)
                
            elif method == 'correlation_threshold':
                # Remove highly correlated features
                corr_matrix = df_copy[columns].corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
                df_copy = df_copy.drop(to_drop, axis=1)
            
            self.preprocessing_history.append({
                'operation': 'apply_feature_selection',
                'method': method,
                'columns': columns,
                'parameters': kwargs
            })
            
        except Exception as e:
            print(f"Error in apply_feature_selection: {str(e)}")
        
        return df_copy
    
    def detect_and_handle_outliers(self, df, columns='all', method='isolation_forest', **kwargs):
        """Detect and handle outliers using various methods"""
        df_copy = df.copy()
        
        if columns == 'all':
            columns = df_copy.select_dtypes(include=[np.number]).columns
        elif columns == 'numerical':
            columns = df_copy.select_dtypes(include=[np.number]).columns
        elif isinstance(columns, str):
            columns = [columns]
        
        # Filter out non-existent columns
        columns = [col for col in columns if col in df_copy.columns]
        
        if not columns:
            return df_copy
        
        try:
            if method == 'isolation_forest':
                clf = IsolationForest(contamination=0.1, random_state=42)
                outliers = clf.fit_predict(df_copy[columns])
                df_copy = df_copy[outliers == 1]
                
            elif method == 'local_outlier_factor':
                clf = LocalOutlierFactor(contamination=0.1)
                outliers = clf.fit_predict(df_copy[columns])
                df_copy = df_copy[outliers == 1]
                
            elif method == 'one_class_svm':
                clf = OneClassSVM(nu=0.1)
                outliers = clf.fit_predict(df_copy[columns])
                df_copy = df_copy[outliers == 1]
                
            elif method == 'elliptic_envelope':
                clf = EllipticEnvelope(contamination=0.1)
                outliers = clf.fit_predict(df_copy[columns])
                df_copy = df_copy[outliers == 1]
            
            self.preprocessing_history.append({
                'operation': 'detect_and_handle_outliers',
                'method': method,
                'columns': columns,
                'parameters': kwargs
            })
            
        except Exception as e:
            print(f"Error in detect_and_handle_outliers: {str(e)}")
        
        return df_copy
    
    def apply_dimensionality_reduction(self, df, columns='all', method='pca', n_components=2, **kwargs):
        """Apply dimensionality reduction techniques"""
        df_copy = df.copy()
        
        if columns == 'all':
            columns = df_copy.select_dtypes(include=[np.number]).columns
        elif columns == 'numerical':
            columns = df_copy.select_dtypes(include=[np.number]).columns
        elif isinstance(columns, str):
            columns = [columns]
        
        # Filter out non-existent columns
        columns = [col for col in columns if col in df_copy.columns]
        
        if not columns or len(columns) <= n_components:
            return df_copy
        
        try:
            if method == 'pca':
                reducer = PCA(n_components=n_components)
                reduced_features = reducer.fit_transform(df_copy[columns])
                feature_names = [f'pca_{i}' for i in range(n_components)]
                
            elif method == 'factor_analysis':
                reducer = FactorAnalysis(n_components=n_components)
                reduced_features = reducer.fit_transform(df_copy[columns])
                feature_names = [f'factor_{i}' for i in range(n_components)]
            
            elif method == 'tsne':
                reducer = TSNE(n_components=n_components, random_state=42)
                reduced_features = reducer.fit_transform(df_copy[columns])
                feature_names = [f'tsne_{i}' for i in range(n_components)]
            
            # Replace original columns with reduced features
            df_copy = df_copy.drop(columns, axis=1)
            reduced_df = pd.DataFrame(reduced_features, columns=feature_names, index=df_copy.index)
            df_copy = pd.concat([df_copy, reduced_df], axis=1)
            
            self.transformers[f'{method}_reducer'] = reducer
            
            self.preprocessing_history.append({
                'operation': 'apply_dimensionality_reduction',
                'method': method,
                'n_components': n_components,
                'columns': columns,
                'parameters': kwargs
            })
            
        except Exception as e:
            print(f"Error in apply_dimensionality_reduction: {str(e)}")
        
        return df_copy
    
    def create_interaction_features(self, df, columns='all', method='multiplicative', **kwargs):
        """Create interaction features between variables"""
        df_copy = df.copy()
        
        if columns == 'all':
            columns = df_copy.select_dtypes(include=[np.number]).columns
        elif columns == 'numerical':
            columns = df_copy.select_dtypes(include=[np.number]).columns
        elif isinstance(columns, str):
            columns = [columns]
        
        # Filter out non-existent columns and limit to reasonable number
        columns = [col for col in columns if col in df_copy.columns][:5]  # Limit to avoid explosion
        
        if not columns or len(columns) < 2:
            return df_copy
        
        try:
            for i, col1 in enumerate(columns):
                for col2 in columns[i+1:]:
                    if method == 'multiplicative':
                        df_copy[f'{col1}_{col2}_mult'] = df_copy[col1] * df_copy[col2]
                    elif method == 'additive':
                        df_copy[f'{col1}_{col2}_add'] = df_copy[col1] + df_copy[col2]
                    elif method == 'ratio_features':
                        # Avoid division by zero
                        df_copy[f'{col1}_{col2}_ratio'] = df_copy[col1] / (df_copy[col2] + 1e-8)
                    elif method == 'polynomial':
                        df_copy[f'{col1}_{col2}_poly'] = df_copy[col1] * df_copy[col2]
            
            self.preprocessing_history.append({
                'operation': 'create_interaction_features',
                'method': method,
                'columns': columns,
                'parameters': kwargs
            })
            
        except Exception as e:
            print(f"Error in create_interaction_features: {str(e)}")
        
        return df_copy
    
    def handle_imbalanced_data(self, df, target_column=None, method='smote', **kwargs):
        """Handle imbalanced datasets (requires target column)"""
        if target_column is None or target_column not in df.columns:
            return df
        
        df_copy = df.copy()
        
        try:
            X = df_copy.drop(target_column, axis=1)
            y = df_copy[target_column]
            
            # Only apply to numerical features for simplicity
            X_numerical = X.select_dtypes(include=[np.number])
            
            if len(X_numerical.columns) == 0:
                return df_copy
            
            if method == 'smote':
                sampler = SMOTE(random_state=42)
            elif method == 'adasyn':
                sampler = ADASYN(random_state=42)
            elif method == 'random_oversample':
                sampler = RandomOverSampler(random_state=42)
            elif method == 'random_undersample':
                sampler = RandomUnderSampler(random_state=42)
            elif method == 'tomek_links':
                sampler = TomekLinks()
            else:
                return df_copy
            
            X_resampled, y_resampled = sampler.fit_resample(X_numerical, y)
            
            # Combine resampled data
            resampled_df = pd.DataFrame(X_resampled, columns=X_numerical.columns)
            resampled_df[target_column] = y_resampled
            
            # Add back non-numerical columns (this is simplified)
            for col in X.columns:
                if col not in X_numerical.columns:
                    resampled_df[col] = X[col].iloc[0]  # Simplified approach
            
            self.preprocessing_history.append({
                'operation': 'handle_imbalanced_data',
                'method': method,
                'target_column': target_column,
                'parameters': kwargs
            })
            
            return resampled_df
            
        except Exception as e:
            print(f"Error in handle_imbalanced_data: {str(e)}")
            return df_copy
    
    def get_preprocessing_summary(self):
        """Get summary of all preprocessing operations performed"""
        return {
            'operations_performed': len(self.preprocessing_history),
            'history': self.preprocessing_history,
            'scalers_fitted': list(self.scalers.keys()),
            'encoders_fitted': list(self.encoders.keys()),
            'transformers_fitted': list(self.transformers.keys())
        } 