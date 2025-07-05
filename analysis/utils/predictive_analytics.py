"""
Predictive Analytics Module

This module implements predictive analytics algorithms to answer "What will happen?"
Includes regression algorithms (Linear, Ridge, Decision Tree) and classification algorithms 
(Logistic Regression, Random Forest, SVM).

Created: 2025-01-21
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class PredictiveAnalytics:
    """
    Predictive analytics for understanding what will happen
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def perform_analysis(self, df: pd.DataFrame, algorithm: str, goal: str) -> Dict[str, Any]:
        """Main entry point for predictive analytics"""
        try:
            if algorithm == 'linear_regression':
                return self.perform_linear_regression(df, goal)
            elif algorithm == 'ridge_regression':
                return self.perform_ridge_regression(df, goal)
            elif algorithm == 'decision_tree_regression':
                return self.perform_decision_tree_regression(df, goal)
            elif algorithm == 'logistic_regression':
                return self.perform_logistic_regression(df, goal)
            elif algorithm == 'random_forest_classification':
                return self.perform_random_forest_classification(df, goal)
            elif algorithm == 'svm_classification':
                return self.perform_svm_classification(df, goal)
            else:
                return {'error': f'Unknown algorithm: {algorithm}'}
        except Exception as e:
            logger.error(f"Error in predictive analysis {algorithm}: {e}")
            return {'error': str(e)}
    
    def prepare_data_for_regression(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
        """Prepare data for regression analysis"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            raise ValueError("Need at least 2 numerical columns for regression")
        
        # Use the last numerical column as target, others as features
        target_col = numerical_cols[-1]
        feature_cols = numerical_cols[:-1]
        
        # Handle missing values
        df_clean = df[feature_cols + [target_col]].fillna(df[feature_cols + [target_col]].mean())
        
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values
        
        return X, y, feature_cols, target_col
    
    def prepare_data_for_classification(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
        """Prepare data for classification analysis"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(categorical_cols) == 0:
            raise ValueError("Need at least 1 categorical column for classification")
        
        # Use the first categorical column as target
        target_col = categorical_cols[0]
        feature_cols = numerical_cols
        
        if len(feature_cols) == 0:
            raise ValueError("Need at least 1 numerical feature for classification")
        
        # Handle missing values
        df_clean = df[feature_cols + [target_col]].copy()
        df_clean[feature_cols] = df_clean[feature_cols].fillna(df_clean[feature_cols].mean())
        df_clean[target_col] = df_clean[target_col].fillna(df_clean[target_col].mode()[0])
        
        X = df_clean[feature_cols].values
        
        # Encode categorical target
        y = self.label_encoder.fit_transform(df_clean[target_col])
        
        return X, y, feature_cols, target_col
    
    def perform_linear_regression(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform linear regression analysis"""
        try:
            X, y, feature_cols, target_col = self.prepare_data_for_regression(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            
            # Feature importance (coefficients)
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'coefficient': model.coef_,
                'abs_coefficient': np.abs(model.coef_)
            }).sort_values('abs_coefficient', ascending=False)
            
            # Generate insights
            insights = self.generate_regression_insights(
                r2, rmse, mae, cv_scores, feature_importance, target_col, goal
            )
            
            # Create visualizations
            graphs = self.create_regression_visualizations(
                y_test, y_pred, feature_importance, target_col
            )
            
            return {
                'algorithm': 'Linear Regression',
                'results': {
                    'r2_score': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'cv_mean_r2': cv_scores.mean(),
                    'cv_std_r2': cv_scores.std(),
                    'feature_importance': feature_importance.to_dict('records'),
                    'target_column': target_col,
                    'feature_columns': feature_cols
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in linear regression: {e}")
            return {'error': str(e)}
    
    def perform_ridge_regression(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform ridge regression analysis"""
        try:
            X, y, feature_cols, target_col = self.prepare_data_for_regression(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model with different alpha values
            alphas = [0.1, 1.0, 10.0, 100.0]
            best_alpha = None
            best_score = -np.inf
            
            for alpha in alphas:
                model = Ridge(alpha=alpha)
                scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_alpha = alpha
            
            # Train final model
            model = Ridge(alpha=best_alpha)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'coefficient': model.coef_,
                'abs_coefficient': np.abs(model.coef_)
            }).sort_values('abs_coefficient', ascending=False)
            
            # Generate insights
            insights = self.generate_regression_insights(
                r2, rmse, mae, [best_score], feature_importance, target_col, goal
            )
            insights.append(f"Optimal regularization parameter (alpha): {best_alpha}")
            
            # Create visualizations
            graphs = self.create_regression_visualizations(
                y_test, y_pred, feature_importance, target_col
            )
            
            return {
                'algorithm': 'Ridge Regression',
                'results': {
                    'r2_score': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'best_alpha': best_alpha,
                    'cv_best_score': best_score,
                    'feature_importance': feature_importance.to_dict('records'),
                    'target_column': target_col,
                    'feature_columns': feature_cols
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in ridge regression: {e}")
            return {'error': str(e)}
    
    def perform_decision_tree_regression(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform decision tree regression analysis"""
        try:
            X, y, feature_cols, target_col = self.prepare_data_for_regression(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = DecisionTreeRegressor(random_state=42, max_depth=10)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Generate insights
            insights = self.generate_tree_regression_insights(
                r2, rmse, mae, feature_importance, target_col, goal
            )
            
            # Create visualizations
            graphs = self.create_tree_regression_visualizations(
                y_test, y_pred, feature_importance, target_col
            )
            
            return {
                'algorithm': 'Decision Tree Regression',
                'results': {
                    'r2_score': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'feature_importance': feature_importance.to_dict('records'),
                    'target_column': target_col,
                    'feature_columns': feature_cols
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in decision tree regression: {e}")
            return {'error': str(e)}
    
    def perform_logistic_regression(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform logistic regression analysis"""
        try:
            X, y, feature_cols, target_col = self.prepare_data_for_classification(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            
            # Feature importance (coefficients)
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'coefficient': model.coef_[0] if model.coef_.shape[0] == 1 else model.coef_[0],
                'abs_coefficient': np.abs(model.coef_[0] if model.coef_.shape[0] == 1 else model.coef_[0])
            }).sort_values('abs_coefficient', ascending=False)
            
            # Generate insights
            insights = self.generate_classification_insights(
                accuracy, precision, recall, f1, cv_scores, feature_importance, target_col, goal
            )
            
            # Create visualizations
            graphs = self.create_classification_visualizations(
                y_test, y_pred, y_pred_proba, feature_importance, target_col
            )
            
            return {
                'algorithm': 'Logistic Regression',
                'results': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean_accuracy': cv_scores.mean(),
                    'cv_std_accuracy': cv_scores.std(),
                    'feature_importance': feature_importance.to_dict('records'),
                    'target_column': target_col,
                    'feature_columns': feature_cols,
                    'class_labels': self.label_encoder.classes_.tolist()
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in logistic regression: {e}")
            return {'error': str(e)}
    
    def perform_random_forest_classification(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform random forest classification analysis"""
        try:
            X, y, feature_cols, target_col = self.prepare_data_for_classification(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Generate insights
            insights = self.generate_tree_classification_insights(
                accuracy, precision, recall, f1, cv_scores, feature_importance, target_col, goal
            )
            
            # Create visualizations
            graphs = self.create_tree_classification_visualizations(
                y_test, y_pred, y_pred_proba, feature_importance, target_col
            )
            
            return {
                'algorithm': 'Random Forest Classification',
                'results': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean_accuracy': cv_scores.mean(),
                    'cv_std_accuracy': cv_scores.std(),
                    'feature_importance': feature_importance.to_dict('records'),
                    'target_column': target_col,
                    'feature_columns': feature_cols,
                    'class_labels': self.label_encoder.classes_.tolist()
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in random forest classification: {e}")
            return {'error': str(e)}
    
    def perform_svm_classification(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform SVM classification analysis"""
        try:
            X, y, feature_cols, target_col = self.prepare_data_for_classification(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            model = SVC(kernel='rbf', random_state=42, probability=True)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            
            # Generate insights
            insights = self.generate_svm_classification_insights(
                accuracy, precision, recall, f1, cv_scores, target_col, goal
            )
            
            # Create visualizations
            graphs = self.create_svm_classification_visualizations(
                y_test, y_pred, y_pred_proba, target_col
            )
            
            return {
                'algorithm': 'SVM Classification',
                'results': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean_accuracy': cv_scores.mean(),
                    'cv_std_accuracy': cv_scores.std(),
                    'target_column': target_col,
                    'feature_columns': feature_cols,
                    'class_labels': self.label_encoder.classes_.tolist()
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in SVM classification: {e}")
            return {'error': str(e)}
    
    def generate_regression_insights(self, r2: float, rmse: float, mae: float, 
                                   cv_scores: List[float], feature_importance: pd.DataFrame,
                                   target_col: str, goal: str) -> List[str]:
        """Generate insights from regression analysis"""
        insights = []
        
        # Model performance insights
        insights.append(f"Model explains {r2*100:.1f}% of variance in {target_col}")
        insights.append(f"Average prediction error: {rmse:.2f} (RMSE), {mae:.2f} (MAE)")
        
        if len(cv_scores) > 1:
            insights.append(f"Cross-validation R² score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        # Feature importance insights
        top_feature = feature_importance.iloc[0]
        insights.append(f"Most important feature: {top_feature['feature']} (coefficient: {top_feature['coefficient']:.3f})")
        
        # Model quality assessment
        if r2 > 0.8:
            insights.append("Excellent model performance - high predictive accuracy")
        elif r2 > 0.6:
            insights.append("Good model performance - moderate predictive accuracy")
        elif r2 > 0.4:
            insights.append("Fair model performance - limited predictive accuracy")
        else:
            insights.append("Poor model performance - consider feature engineering")
        
        # Goal-specific insights
        if 'predict' in goal.lower() or 'forecast' in goal.lower():
            insights.append("Model can be used for making predictions on new data")
        
        return insights
    
    def generate_tree_regression_insights(self, r2: float, rmse: float, mae: float,
                                        feature_importance: pd.DataFrame, target_col: str, 
                                        goal: str) -> List[str]:
        """Generate insights from tree regression analysis"""
        insights = []
        
        insights.append(f"Decision tree explains {r2*100:.1f}% of variance in {target_col}")
        insights.append(f"Average prediction error: {rmse:.2f} (RMSE), {mae:.2f} (MAE)")
        
        # Feature importance insights
        top_feature = feature_importance.iloc[0]
        insights.append(f"Most important feature: {top_feature['feature']} (importance: {top_feature['importance']:.3f})")
        
        # Tree-specific insights
        insights.append("Decision tree provides interpretable rules for predictions")
        insights.append("Model captures non-linear relationships in the data")
        
        return insights
    
    def generate_classification_insights(self, accuracy: float, precision: float, recall: float,
                                       f1: float, cv_scores: List[float], feature_importance: pd.DataFrame,
                                       target_col: str, goal: str) -> List[str]:
        """Generate insights from classification analysis"""
        insights = []
        
        # Model performance insights
        insights.append(f"Classification accuracy: {accuracy*100:.1f}%")
        insights.append(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")
        
        if len(cv_scores) > 1:
            insights.append(f"Cross-validation accuracy: {np.mean(cv_scores)*100:.1f}% ± {np.std(cv_scores)*100:.1f}%")
        
        # Feature importance insights
        top_feature = feature_importance.iloc[0]
        insights.append(f"Most predictive feature: {top_feature['feature']} (coefficient: {top_feature['coefficient']:.3f})")
        
        # Model quality assessment
        if accuracy > 0.9:
            insights.append("Excellent classification performance")
        elif accuracy > 0.8:
            insights.append("Good classification performance")
        elif accuracy > 0.7:
            insights.append("Fair classification performance")
        else:
            insights.append("Poor classification performance - consider feature engineering")
        
        return insights
    
    def generate_tree_classification_insights(self, accuracy: float, precision: float, recall: float,
                                            f1: float, cv_scores: List[float], feature_importance: pd.DataFrame,
                                            target_col: str, goal: str) -> List[str]:
        """Generate insights from tree classification analysis"""
        insights = []
        
        insights.append(f"Random Forest classification accuracy: {accuracy*100:.1f}%")
        insights.append(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")
        
        if len(cv_scores) > 1:
            insights.append(f"Cross-validation accuracy: {np.mean(cv_scores)*100:.1f}% ± {np.std(cv_scores)*100:.1f}%")
        
        # Feature importance insights
        top_feature = feature_importance.iloc[0]
        insights.append(f"Most important feature: {top_feature['feature']} (importance: {top_feature['importance']:.3f})")
        
        # Tree-specific insights
        insights.append("Random Forest provides robust predictions with feature importance ranking")
        insights.append("Model is less prone to overfitting than single decision trees")
        
        return insights
    
    def generate_svm_classification_insights(self, accuracy: float, precision: float, recall: float,
                                           f1: float, cv_scores: List[float], target_col: str, 
                                           goal: str) -> List[str]:
        """Generate insights from SVM classification analysis"""
        insights = []
        
        insights.append(f"SVM classification accuracy: {accuracy*100:.1f}%")
        insights.append(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")
        
        if len(cv_scores) > 1:
            insights.append(f"Cross-validation accuracy: {np.mean(cv_scores)*100:.1f}% ± {np.std(cv_scores)*100:.1f}%")
        
        # SVM-specific insights
        insights.append("SVM finds optimal decision boundary using support vectors")
        insights.append("Model works well with high-dimensional data and complex patterns")
        
        return insights
    
    def create_regression_visualizations(self, y_test: np.ndarray, y_pred: np.ndarray,
                                       feature_importance: pd.DataFrame, target_col: str) -> List[Dict[str, Any]]:
        """Create visualizations for regression analysis"""
        graphs = []
        
        # 1. Actual vs Predicted scatter plot
        fig = go.Figure()
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        # Actual vs predicted points
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(size=8, opacity=0.6)
        ))
        
        fig.update_layout(
            title=f'Actual vs Predicted {target_col}',
            xaxis_title=f'Actual {target_col}',
            yaxis_title=f'Predicted {target_col}',
            showlegend=True
        )
        
        graphs.append({
            'title': 'Actual vs Predicted',
            'type': 'scatter',
            'data': fig.to_json()
        })
        
        # 2. Feature importance bar chart
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=feature_importance['feature'],
            y=feature_importance['abs_coefficient'],
            name='Feature Importance',
            marker_color='lightblue'
        ))
        
        fig2.update_layout(
            title='Feature Importance (Absolute Coefficients)',
            xaxis_title='Features',
            yaxis_title='Absolute Coefficient',
            showlegend=True
        )
        
        graphs.append({
            'title': 'Feature Importance',
            'type': 'bar',
            'data': fig2.to_json()
        })
        
        # 3. Residuals plot
        residuals = y_test - y_pred
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(size=6, opacity=0.6)
        ))
        
        # Add horizontal line at y=0
        fig3.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig3.update_layout(
            title='Residuals vs Predicted Values',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals',
            showlegend=True
        )
        
        graphs.append({
            'title': 'Residuals Analysis',
            'type': 'scatter',
            'data': fig3.to_json()
        })
        
        return graphs
    
    def create_tree_regression_visualizations(self, y_test: np.ndarray, y_pred: np.ndarray,
                                            feature_importance: pd.DataFrame, target_col: str) -> List[Dict[str, Any]]:
        """Create visualizations for tree regression analysis"""
        graphs = []
        
        # 1. Actual vs Predicted scatter plot
        fig = go.Figure()
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        # Actual vs predicted points
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(size=8, opacity=0.6)
        ))
        
        fig.update_layout(
            title=f'Decision Tree: Actual vs Predicted {target_col}',
            xaxis_title=f'Actual {target_col}',
            yaxis_title=f'Predicted {target_col}',
            showlegend=True
        )
        
        graphs.append({
            'title': 'Decision Tree Predictions',
            'type': 'scatter',
            'data': fig.to_json()
        })
        
        # 2. Feature importance bar chart
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=feature_importance['feature'],
            y=feature_importance['importance'],
            name='Feature Importance',
            marker_color='lightgreen'
        ))
        
        fig2.update_layout(
            title='Decision Tree Feature Importance',
            xaxis_title='Features',
            yaxis_title='Importance Score',
            showlegend=True
        )
        
        graphs.append({
            'title': 'Feature Importance',
            'type': 'bar',
            'data': fig2.to_json()
        })
        
        return graphs
    
    def create_classification_visualizations(self, y_test: np.ndarray, y_pred: np.ndarray,
                                           y_pred_proba: np.ndarray, feature_importance: pd.DataFrame,
                                           target_col: str) -> List[Dict[str, Any]]:
        """Create visualizations for classification analysis"""
        graphs = []
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f'Predicted {i}' for i in range(len(cm))],
            y=[f'Actual {i}' for i in range(len(cm))],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16}
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Class',
            yaxis_title='Actual Class'
        )
        
        graphs.append({
            'title': 'Confusion Matrix',
            'type': 'heatmap',
            'data': fig.to_json()
        })
        
        # 2. Feature importance bar chart
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=feature_importance['feature'],
            y=feature_importance['abs_coefficient'],
            name='Feature Importance',
            marker_color='lightcoral'
        ))
        
        fig2.update_layout(
            title='Feature Importance (Absolute Coefficients)',
            xaxis_title='Features',
            yaxis_title='Absolute Coefficient',
            showlegend=True
        )
        
        graphs.append({
            'title': 'Feature Importance',
            'type': 'bar',
            'data': fig2.to_json()
        })
        
        # 3. Prediction probabilities (if binary classification)
        if y_pred_proba.shape[1] == 2:
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(
                x=y_pred_proba[:, 1],
                nbinsx=20,
                name='Prediction Probabilities',
                marker_color='lightblue'
            ))
            
            fig3.update_layout(
                title='Distribution of Prediction Probabilities',
                xaxis_title='Probability of Positive Class',
                yaxis_title='Frequency',
                showlegend=True
            )
            
            graphs.append({
                'title': 'Prediction Probabilities',
                'type': 'histogram',
                'data': fig3.to_json()
            })
        
        return graphs
    
    def create_tree_classification_visualizations(self, y_test: np.ndarray, y_pred: np.ndarray,
                                                y_pred_proba: np.ndarray, feature_importance: pd.DataFrame,
                                                target_col: str) -> List[Dict[str, Any]]:
        """Create visualizations for tree classification analysis"""
        graphs = []
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f'Predicted {i}' for i in range(len(cm))],
            y=[f'Actual {i}' for i in range(len(cm))],
            colorscale='Greens',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16}
        ))
        
        fig.update_layout(
            title='Random Forest Confusion Matrix',
            xaxis_title='Predicted Class',
            yaxis_title='Actual Class'
        )
        
        graphs.append({
            'title': 'Confusion Matrix',
            'type': 'heatmap',
            'data': fig.to_json()
        })
        
        # 2. Feature importance bar chart
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=feature_importance['feature'],
            y=feature_importance['importance'],
            name='Feature Importance',
            marker_color='lightgreen'
        ))
        
        fig2.update_layout(
            title='Random Forest Feature Importance',
            xaxis_title='Features',
            yaxis_title='Importance Score',
            showlegend=True
        )
        
        graphs.append({
            'title': 'Feature Importance',
            'type': 'bar',
            'data': fig2.to_json()
        })
        
        return graphs
    
    def create_svm_classification_visualizations(self, y_test: np.ndarray, y_pred: np.ndarray,
                                               y_pred_proba: np.ndarray, target_col: str) -> List[Dict[str, Any]]:
        """Create visualizations for SVM classification analysis"""
        graphs = []
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f'Predicted {i}' for i in range(len(cm))],
            y=[f'Actual {i}' for i in range(len(cm))],
            colorscale='Oranges',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16}
        ))
        
        fig.update_layout(
            title='SVM Confusion Matrix',
            xaxis_title='Predicted Class',
            yaxis_title='Actual Class'
        )
        
        graphs.append({
            'title': 'Confusion Matrix',
            'type': 'heatmap',
            'data': fig.to_json()
        })
        
        # 2. Prediction confidence distribution
        max_proba = np.max(y_pred_proba, axis=1)
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=max_proba,
            nbinsx=20,
            name='Prediction Confidence',
            marker_color='orange'
        ))
        
        fig2.update_layout(
            title='SVM Prediction Confidence Distribution',
            xaxis_title='Maximum Probability',
            yaxis_title='Frequency',
            showlegend=True
        )
        
        graphs.append({
            'title': 'Prediction Confidence',
            'type': 'histogram',
            'data': fig2.to_json()
        })
        
        return graphs 