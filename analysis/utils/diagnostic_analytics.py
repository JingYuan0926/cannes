"""
Diagnostic Analytics Module

This module implements diagnostic analytics algorithms to answer "Why did it happen?"
Includes decision trees for feature importance analysis and causal inference models
for understanding cause-effect relationships.

Created: 2025-01-21
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DiagnosticAnalytics:
    """
    Diagnostic analytics for understanding why things happened
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
    
    def perform_analysis(self, df: pd.DataFrame, algorithm: str, goal: str) -> Dict[str, Any]:
        """Main entry point for diagnostic analytics"""
        try:
            if algorithm == 'decision_tree_analysis':
                return self.perform_decision_tree_analysis(df, goal)
            elif algorithm == 'feature_importance_analysis':
                return self.perform_feature_importance_analysis(df, goal)
            elif algorithm == 'causal_inference':
                return self.perform_causal_inference(df, goal)
            elif algorithm == 'correlation_analysis':
                return self.perform_correlation_analysis(df, goal)
            elif algorithm == 'anomaly_detection':
                return self.perform_anomaly_detection(df, goal)
            elif algorithm == 'root_cause_analysis':
                return self.perform_root_cause_analysis(df, goal)
            else:
                return {'error': f'Unknown algorithm: {algorithm}'}
        except Exception as e:
            logger.error(f"Error in diagnostic analysis {algorithm}: {e}")
            return {'error': str(e)}
    
    def perform_decision_tree_analysis(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform decision tree analysis for interpretable rules"""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(numerical_cols) == 0:
                raise ValueError("Need at least 1 numerical column for decision tree analysis")
            
            # Determine if this is classification or regression
            if len(categorical_cols) > 0:
                # Classification
                target_col = categorical_cols[0]
                feature_cols = numerical_cols
                
                # Prepare data
                X = df[feature_cols].fillna(df[feature_cols].mean())
                y = self.label_encoder.fit_transform(df[target_col].fillna(df[target_col].mode()[0]))
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train decision tree
                tree = DecisionTreeClassifier(max_depth=5, random_state=42)
                tree.fit(X_train, y_train)
                
                # Make predictions
                y_pred = tree.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Extract rules
                tree_rules = export_text(tree, feature_names=feature_cols, max_depth=3)
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': tree.feature_importances_
                }).sort_values('importance', ascending=False)
                
                analysis_type = 'classification'
                performance_metric = accuracy
                
            else:
                # Regression
                target_col = numerical_cols[-1]
                feature_cols = numerical_cols[:-1]
                
                if len(feature_cols) == 0:
                    raise ValueError("Need at least 2 numerical columns for regression analysis")
                
                # Prepare data
                X = df[feature_cols].fillna(df[feature_cols].mean())
                y = df[target_col].fillna(df[target_col].mean())
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train decision tree
                tree = DecisionTreeRegressor(max_depth=5, random_state=42)
                tree.fit(X_train, y_train)
                
                # Make predictions
                y_pred = tree.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                
                # Extract rules
                tree_rules = export_text(tree, feature_names=feature_cols, max_depth=3)
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': tree.feature_importances_
                }).sort_values('importance', ascending=False)
                
                analysis_type = 'regression'
                performance_metric = r2
            
            # Generate insights
            insights = self.generate_decision_tree_insights(
                tree_rules, feature_importance, analysis_type, performance_metric, target_col, goal
            )
            
            # Create visualizations
            graphs = self.create_decision_tree_visualizations(
                feature_importance, tree, X_train, y_train, feature_cols, target_col
            )
            
            return {
                'algorithm': 'Decision Tree Analysis',
                'results': {
                    'analysis_type': analysis_type,
                    'performance_metric': performance_metric,
                    'target_column': target_col,
                    'feature_columns': feature_cols,
                    'feature_importance': feature_importance.to_dict('records'),
                    'tree_rules': tree_rules,
                    'tree_depth': tree.get_depth(),
                    'n_leaves': tree.get_n_leaves()
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in decision tree analysis: {e}")
            return {'error': str(e)}
    
    def perform_feature_importance_analysis(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform comprehensive feature importance analysis"""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(numerical_cols) == 0:
                raise ValueError("Need at least 1 numerical column for feature importance analysis")
            
            # Check if we have enough samples
            if len(df) < 5:
                raise ValueError(f"Need at least 5 samples for feature importance analysis, but got {len(df)} samples")
            
            # Prepare target and features
            if len(categorical_cols) > 0:
                target_col = categorical_cols[0]
                feature_cols = numerical_cols
                target_type = 'categorical'
                
                X = df[feature_cols].fillna(df[feature_cols].mean())
                y = self.label_encoder.fit_transform(df[target_col].fillna(df[target_col].mode()[0]))
                
                # Mutual information for classification
                mi_scores = mutual_info_classif(X, y, random_state=42)
                
                # Random Forest importance
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, y)
                rf_importance = rf.feature_importances_
                
            else:
                target_col = numerical_cols[-1]
                feature_cols = numerical_cols[:-1]
                target_type = 'numerical'
                
                if len(feature_cols) == 0:
                    raise ValueError("Need at least 2 numerical columns")
                
                X = df[feature_cols].fillna(df[feature_cols].mean())
                y = df[target_col].fillna(df[target_col].mean())
                
                # Mutual information for regression
                mi_scores = mutual_info_regression(X, y, random_state=42)
                
                # Random Forest importance
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                rf_importance = rf.feature_importances_
            
            # Correlation analysis
            correlations = []
            for col in feature_cols:
                if target_type == 'categorical':
                    # Point-biserial correlation for categorical target
                    corr, p_value = pearsonr(X[col], y)
                else:
                    # Pearson correlation for numerical target
                    corr, p_value = pearsonr(X[col], y)
                correlations.append({'feature': col, 'correlation': corr, 'p_value': p_value})
            
            # Combine all importance measures
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'mutual_info': mi_scores,
                'random_forest': rf_importance,
                'correlation': [abs(c['correlation']) for c in correlations]
            })
            
            # Normalize and combine scores
            importance_df['mi_normalized'] = importance_df['mutual_info'] / importance_df['mutual_info'].max()
            importance_df['rf_normalized'] = importance_df['random_forest'] / importance_df['random_forest'].max()
            importance_df['corr_normalized'] = importance_df['correlation'] / importance_df['correlation'].max()
            
            importance_df['combined_score'] = (
                importance_df['mi_normalized'] + 
                importance_df['rf_normalized'] + 
                importance_df['corr_normalized']
            ) / 3
            
            importance_df = importance_df.sort_values('combined_score', ascending=False)
            
            # Generate insights
            insights = self.generate_feature_importance_insights(
                importance_df, correlations, target_col, target_type, goal
            )
            
            # Create visualizations
            graphs = self.create_feature_importance_visualizations(
                importance_df, correlations, target_col
            )
            
            return {
                'algorithm': 'Feature Importance Analysis',
                'results': {
                    'target_column': target_col,
                    'target_type': target_type,
                    'feature_columns': feature_cols,
                    'feature_importance': importance_df.to_dict('records'),
                    'correlations': correlations,
                    'n_features': len(feature_cols)
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in feature importance analysis: {e}")
            return {'error': str(e)}
    
    def perform_causal_inference(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform causal inference analysis"""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) < 3:
                raise ValueError("Need at least 3 numerical columns for causal inference")
            
            # Simple causal inference using correlation and partial correlation
            causal_relationships = []
            
            # Consider all pairs of variables
            for i, var1 in enumerate(numerical_cols):
                for j, var2 in enumerate(numerical_cols[i+1:], i+1):
                    # Direct correlation
                    direct_corr, direct_p = pearsonr(df[var1].fillna(df[var1].mean()), 
                                                   df[var2].fillna(df[var2].mean()))
                    
                    # Partial correlations (controlling for other variables)
                    other_vars = [col for col in numerical_cols if col not in [var1, var2]]
                    if len(other_vars) > 0:
                        # Use first other variable as control
                        control_var = other_vars[0]
                        partial_corr = self.calculate_partial_correlation(
                            df[var1].fillna(df[var1].mean()),
                            df[var2].fillna(df[var2].mean()),
                            df[control_var].fillna(df[control_var].mean())
                        )
                    else:
                        partial_corr = direct_corr
                    
                    # Granger causality approximation (using lagged correlation)
                    granger_score = self.approximate_granger_causality(
                        df[var1].fillna(df[var1].mean()),
                        df[var2].fillna(df[var2].mean())
                    )
                    
                    causal_relationships.append({
                        'cause': var1,
                        'effect': var2,
                        'direct_correlation': direct_corr,
                        'partial_correlation': partial_corr,
                        'granger_score': granger_score,
                        'strength': abs(direct_corr) * abs(partial_corr),
                        'p_value': direct_p
                    })
            
            # Sort by causal strength
            causal_relationships.sort(key=lambda x: x['strength'], reverse=True)
            
            # Generate insights
            insights = self.generate_causal_inference_insights(
                causal_relationships, numerical_cols, goal
            )
            
            # Create visualizations
            graphs = self.create_causal_inference_visualizations(
                causal_relationships, df, numerical_cols
            )
            
            return {
                'algorithm': 'Causal Inference',
                'results': {
                    'causal_relationships': causal_relationships,
                    'variables': numerical_cols,
                    'n_relationships': len(causal_relationships),
                    'strongest_relationship': causal_relationships[0] if causal_relationships else None
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in causal inference: {e}")
            return {'error': str(e)}
    
    def calculate_partial_correlation(self, x: pd.Series, y: pd.Series, z: pd.Series) -> float:
        """Calculate partial correlation between x and y controlling for z"""
        try:
            # Remove NaN values
            valid_indices = x.notna() & y.notna() & z.notna()
            x_clean = x[valid_indices]
            y_clean = y[valid_indices]
            z_clean = z[valid_indices]
            
            if len(x_clean) < 3:  # Need at least 3 points for correlation
                return 0.0
                
            # Check for constant series
            if x_clean.nunique() <= 1 or y_clean.nunique() <= 1 or z_clean.nunique() <= 1:
                return 0.0
            
            # Correlations
            rxy, _ = pearsonr(x_clean, y_clean)
            rxz, _ = pearsonr(x_clean, z_clean)
            ryz, _ = pearsonr(y_clean, z_clean)
            
            # Handle NaN correlations
            if pd.isna(rxy) or pd.isna(rxz) or pd.isna(ryz):
                return 0.0
            
            # Partial correlation formula
            numerator = rxy - (rxz * ryz)
            denominator = np.sqrt((1 - rxz**2) * (1 - ryz**2))
            
            if denominator == 0 or pd.isna(denominator):
                return 0.0
            
            result = numerator / denominator
            return 0.0 if pd.isna(result) else float(result)
        except:
            return 0.0
    
    def approximate_granger_causality(self, x: pd.Series, y: pd.Series, max_lag: int = 3) -> float:
        """Approximate Granger causality using lagged correlations"""
        try:
            if len(x) < max_lag + 1:
                return 0.0
            
            # Calculate lagged correlations
            lagged_corrs = []
            for lag in range(1, max_lag + 1):
                if len(x) > lag:
                    x_lagged = x.shift(lag).dropna()
                    y_aligned = y.iloc[lag:lag+len(x_lagged)]
                    
                    if len(x_lagged) > 2 and len(y_aligned) > 2:
                        # Check for constant series
                        if x_lagged.nunique() <= 1 or y_aligned.nunique() <= 1:
                            continue
                            
                        corr, _ = pearsonr(x_lagged, y_aligned)
                        if not pd.isna(corr):
                            lagged_corrs.append(abs(corr))
            
            result = np.mean(lagged_corrs) if lagged_corrs else 0.0
            return 0.0 if pd.isna(result) else float(result)
        except:
            return 0.0
    
    def perform_correlation_analysis(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform comprehensive correlation analysis"""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) < 2:
                raise ValueError("Need at least 2 numerical columns for correlation analysis")
            
            # Clean data
            df_clean = df[numerical_cols].fillna(df[numerical_cols].mean())
            
            # Check for constant columns (will produce NaN correlations)
            constant_cols = []
            for col in numerical_cols:
                if df_clean[col].nunique() <= 1:
                    constant_cols.append(col)
            
            # Remove constant columns
            if constant_cols:
                logger.warning(f"Removing constant columns: {constant_cols}")
                numerical_cols = [col for col in numerical_cols if col not in constant_cols]
                df_clean = df_clean[numerical_cols]
            
            if len(numerical_cols) < 2:
                raise ValueError("Need at least 2 non-constant numerical columns for correlation analysis")
            
            # Pearson correlation
            pearson_corr = df_clean.corr(method='pearson').fillna(0)
            
            # Spearman correlation  
            spearman_corr = df_clean.corr(method='spearman').fillna(0)
            
            # Find strong correlations
            strong_correlations = []
            for i, col1 in enumerate(numerical_cols):
                for j, col2 in enumerate(numerical_cols[i+1:], i+1):
                    pearson_val = pearson_corr.iloc[i, j]
                    spearman_val = spearman_corr.iloc[i, j]
                    
                    # Handle NaN values
                    if pd.isna(pearson_val):
                        pearson_val = 0
                    if pd.isna(spearman_val):
                        spearman_val = 0
                    
                    if abs(pearson_val) > 0.5 or abs(spearman_val) > 0.5:
                        strong_correlations.append({
                            'variable1': col1,
                            'variable2': col2,
                            'pearson': float(pearson_val),
                            'spearman': float(spearman_val),
                            'strength': max(abs(pearson_val), abs(spearman_val))
                        })
            
            # Sort by strength
            strong_correlations.sort(key=lambda x: x['strength'], reverse=True)
            
            # Generate insights
            insights = self.generate_correlation_insights(
                pearson_corr, spearman_corr, strong_correlations, numerical_cols, goal
            )
            
            # Create visualizations
            graphs = self.create_correlation_visualizations(
                pearson_corr, spearman_corr, strong_correlations, numerical_cols
            )
            
            # Clean correlation matrices for JSON serialization
            pearson_dict = {}
            spearman_dict = {}
            
            for col1 in numerical_cols:
                pearson_dict[col1] = {}
                spearman_dict[col1] = {}
                for col2 in numerical_cols:
                    p_val = pearson_corr.loc[col1, col2]
                    s_val = spearman_corr.loc[col1, col2]
                    
                    pearson_dict[col1][col2] = 0 if pd.isna(p_val) else float(p_val)
                    spearman_dict[col1][col2] = 0 if pd.isna(s_val) else float(s_val)
            
            return {
                'algorithm': 'Correlation Analysis',
                'results': {
                    'pearson_correlation': pearson_dict,
                    'spearman_correlation': spearman_dict,
                    'strong_correlations': strong_correlations,
                    'variables': numerical_cols,
                    'n_strong_correlations': len(strong_correlations),
                    'constant_columns_removed': constant_cols
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {'error': str(e)}
    
    def perform_anomaly_detection(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform anomaly detection analysis"""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) == 0:
                raise ValueError("Need at least 1 numerical column for anomaly detection")
            
            # Statistical anomaly detection
            anomalies = {}
            anomaly_scores = {}
            
            for col in numerical_cols:
                data = df[col].dropna()
                
                # Z-score method
                z_scores = np.abs(stats.zscore(data))
                z_anomalies = data[z_scores > 3].index.tolist()
                
                # IQR method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_anomalies = data[(data < lower_bound) | (data > upper_bound)].index.tolist()
                
                # Combine anomalies
                combined_anomalies = list(set(z_anomalies + iqr_anomalies))
                
                anomalies[col] = {
                    'z_score_anomalies': z_anomalies,
                    'iqr_anomalies': iqr_anomalies,
                    'combined_anomalies': combined_anomalies,
                    'anomaly_count': len(combined_anomalies),
                    'anomaly_percentage': (len(combined_anomalies) / len(data)) * 100
                }
                
                # Calculate anomaly scores for each data point
                anomaly_scores[col] = z_scores.tolist()
            
            # Overall anomaly summary
            total_anomalies = sum([len(anomalies[col]['combined_anomalies']) for col in numerical_cols])
            
            # Generate insights
            insights = self.generate_anomaly_detection_insights(
                anomalies, total_anomalies, numerical_cols, goal
            )
            
            # Create visualizations
            graphs = self.create_anomaly_detection_visualizations(
                df, anomalies, numerical_cols
            )
            
            return {
                'algorithm': 'Anomaly Detection',
                'results': {
                    'anomalies_by_column': anomalies,
                    'anomaly_scores': anomaly_scores,
                    'total_anomalies': total_anomalies,
                    'variables': numerical_cols,
                    'dataset_size': len(df)
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {'error': str(e)}
    
    def perform_root_cause_analysis(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform root cause analysis"""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(numerical_cols) == 0:
                raise ValueError("Need at least 1 numerical column for root cause analysis")
            
            # Identify potential outcome variable (last numerical column)
            outcome_col = numerical_cols[-1]
            potential_causes = numerical_cols[:-1] + categorical_cols
            
            if len(potential_causes) == 0:
                raise ValueError("Need at least 1 potential cause variable")
            
            # Analyze each potential cause
            root_causes = []
            
            for cause_col in potential_causes:
                if cause_col in numerical_cols:
                    # Numerical cause
                    cause_data = df[cause_col].fillna(df[cause_col].mean())
                    outcome_data = df[outcome_col].fillna(df[outcome_col].mean())
                    
                    # Correlation
                    corr, p_value = pearsonr(cause_data, outcome_data)
                    
                    # Segmented analysis
                    high_cause = cause_data > cause_data.median()
                    low_cause = cause_data <= cause_data.median()
                    
                    high_outcome_mean = outcome_data[high_cause].mean()
                    low_outcome_mean = outcome_data[low_cause].mean()
                    
                    impact = abs(high_outcome_mean - low_outcome_mean)
                    
                else:
                    # Categorical cause
                    cause_data = df[cause_col].fillna(df[cause_col].mode()[0])
                    outcome_data = df[outcome_col].fillna(df[outcome_col].mean())
                    
                    # ANOVA-like analysis
                    categories = cause_data.unique()
                    category_means = []
                    
                    for cat in categories:
                        cat_outcome = outcome_data[cause_data == cat]
                        if len(cat_outcome) > 0:
                            category_means.append(cat_outcome.mean())
                    
                    if len(category_means) > 1:
                        impact = max(category_means) - min(category_means)
                        corr = np.std(category_means) / np.mean(category_means) if np.mean(category_means) != 0 else 0
                        p_value = 0.05  # Placeholder
                    else:
                        impact = 0
                        corr = 0
                        p_value = 1.0
                
                root_causes.append({
                    'cause': cause_col,
                    'correlation': corr,
                    'impact': impact,
                    'p_value': p_value,
                    'significance': 'significant' if p_value < 0.05 else 'not significant',
                    'strength': abs(corr) * impact
                })
            
            # Sort by strength
            root_causes.sort(key=lambda x: x['strength'], reverse=True)
            
            # Generate insights
            insights = self.generate_root_cause_insights(
                root_causes, outcome_col, goal
            )
            
            # Create visualizations
            graphs = self.create_root_cause_visualizations(
                root_causes, df, outcome_col
            )
            
            return {
                'algorithm': 'Root Cause Analysis',
                'results': {
                    'outcome_variable': outcome_col,
                    'potential_causes': potential_causes,
                    'root_causes': root_causes,
                    'top_cause': root_causes[0] if root_causes else None,
                    'significant_causes': [rc for rc in root_causes if rc['significance'] == 'significant']
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in root cause analysis: {e}")
            return {'error': str(e)}
    
    def generate_decision_tree_insights(self, tree_rules: str, feature_importance: pd.DataFrame,
                                      analysis_type: str, performance_metric: float,
                                      target_col: str, goal: str) -> List[str]:
        """Generate insights from decision tree analysis"""
        insights = []
        
        if analysis_type == 'classification':
            insights.append(f"Decision tree achieved {performance_metric*100:.1f}% accuracy predicting {target_col}")
        else:
            insights.append(f"Decision tree explains {performance_metric*100:.1f}% of variance in {target_col}")
        
        # Top features
        top_feature = feature_importance.iloc[0]
        insights.append(f"Most important feature: {top_feature['feature']} (importance: {top_feature['importance']:.3f})")
        
        # Number of rules
        rule_count = tree_rules.count('|---')
        insights.append(f"Decision tree contains {rule_count} decision rules")
        
        # Interpretability
        insights.append("Decision tree provides interpretable if-then rules")
        insights.append("Each path from root to leaf represents a decision rule")
        
        return insights
    
    def generate_feature_importance_insights(self, importance_df: pd.DataFrame, 
                                           correlations: List[Dict], target_col: str,
                                           target_type: str, goal: str) -> List[str]:
        """Generate insights from feature importance analysis"""
        insights = []
        
        # Top feature analysis
        top_feature = importance_df.iloc[0]
        insights.append(f"🎯 Most important feature: {top_feature['feature']} (combined score: {top_feature['combined_score']:.3f})")
        
        # Feature ranking context
        insights.append(f"Analyzed {len(importance_df)} features for predicting {target_col}")
        
        # Top features analysis
        top_3_features = importance_df.head(3)
        feature_names = top_3_features['feature'].tolist()
        scores = top_3_features['combined_score'].tolist()
        
        insights.append(f"Top 3 predictive features: {', '.join([f'{name} ({score:.3f})' for name, score in zip(feature_names, scores)])}")
        
        # Feature distribution analysis
        high_importance = importance_df[importance_df['combined_score'] > 0.7]
        medium_importance = importance_df[(importance_df['combined_score'] > 0.4) & (importance_df['combined_score'] <= 0.7)]
        low_importance = importance_df[importance_df['combined_score'] <= 0.4]
        
        insights.append(f"Feature distribution: {len(high_importance)} high-impact, {len(medium_importance)} medium-impact, {len(low_importance)} low-impact features")
        
        # Correlation insights
        strong_corr = [c for c in correlations if abs(c['correlation']) > 0.7]
        if strong_corr:
            insights.append(f"Found {len(strong_corr)} features with strong correlation (>0.7) to {target_col}")
            strongest_corr = max(strong_corr, key=lambda x: abs(x['correlation']))
            insights.append(f"Strongest predictor: {strongest_corr['feature']} (correlation: {strongest_corr['correlation']:.3f})")
        
        # Feature quality assessment
        if top_feature['combined_score'] > 0.8:
            insights.append("✅ Excellent feature quality - strong predictive signals identified")
        elif top_feature['combined_score'] > 0.6:
            insights.append("✅ Good feature quality - moderate predictive signals available")
        elif top_feature['combined_score'] > 0.4:
            insights.append("⚠️ Fair feature quality - weak predictive signals, consider feature engineering")
        else:
            insights.append("❌ Poor feature quality - significant feature engineering needed")
        
        # Methodology insights
        insights.append("📊 Analysis methods: Mutual information (non-linear), Random Forest (interactions), Correlation (linear)")
        
        # Goal-specific recommendations
        if 'predict' in goal.lower() or 'forecast' in goal.lower():
            insights.append(f"💡 For prediction: Build model using top {min(5, len(importance_df))} features for optimal performance")
            if len(high_importance) > 0:
                insights.append(f"Focus on high-impact features: {', '.join(high_importance['feature'].head(3).tolist())}")
        elif 'understand' in goal.lower() or 'insight' in goal.lower():
            insights.append(f"💡 Key drivers of {target_col}: {', '.join(feature_names[:3])}")
        elif 'optimization' in goal.lower():
            insights.append(f"💡 Optimization targets: Modify {feature_names[0]} for maximum impact on {target_col}")
        
        return insights
    
    def generate_causal_inference_insights(self, causal_relationships: List[Dict],
                                         variables: List[str], goal: str) -> List[str]:
        """Generate insights from causal inference analysis"""
        insights = []
        
        if causal_relationships:
            strongest = causal_relationships[0]
            insights.append(f"🔗 Strongest causal relationship: {strongest['cause']} → {strongest['effect']} (strength: {strongest['strength']:.3f})")
            
            # Significance analysis
            significant_rels = [r for r in causal_relationships if r['p_value'] < 0.05]
            insights.append(f"📊 Statistical significance: {len(significant_rels)} of {len(causal_relationships)} relationships are statistically significant (p<0.05)")
            
            # Top causal relationships
            top_3_causal = causal_relationships[:3]
            for i, rel in enumerate(top_3_causal, 1):
                significance = "✓" if rel['p_value'] < 0.05 else "?"
                insights.append(f"{i}. {rel['cause']} → {rel['effect']} (strength: {rel['strength']:.3f}) {significance}")
            
            # Causal network analysis
            causes = list(set([r['cause'] for r in causal_relationships]))
            effects = list(set([r['effect'] for r in causal_relationships]))
            
            # Find variables that are both causes and effects (mediators)
            mediators = list(set(causes) & set(effects))
            if mediators:
                insights.append(f"🔄 Mediating variables identified: {', '.join(mediators[:3])} - these both influence and are influenced by other variables")
            
            # Find pure causes (only cause, never effect)
            pure_causes = list(set(causes) - set(effects))
            if pure_causes:
                insights.append(f"🎯 Root causes identified: {', '.join(pure_causes[:3])} - these are fundamental drivers")
            
            # Find pure effects (only effect, never cause)
            pure_effects = list(set(effects) - set(causes))
            if pure_effects:
                insights.append(f"📈 Outcome variables: {', '.join(pure_effects[:3])} - these are primarily influenced by other factors")
            
        else:
            insights.append("❌ No significant causal relationships detected - variables appear to be independent")
            insights.append("💡 Consider: Longer time series data or additional variables may reveal causal patterns")
        
        # Methodology insights
        insights.append("🔬 Causal inference based on: Direct correlation, Partial correlation (controlling for confounders), Granger causality (temporal precedence)")
        insights.append("⚠️ Note: Results suggest potential causal relationships but require experimental validation for confirmation")
        
        # Goal-specific recommendations
        if 'intervention' in goal.lower() or 'action' in goal.lower():
            if significant_rels:
                top_intervention = max(significant_rels, key=lambda x: x['strength'])
                insights.append(f"💡 Intervention target: Modify {top_intervention['cause']} to impact {top_intervention['effect']}")
        elif 'root cause' in goal.lower():
            if pure_causes:
                insights.append(f"💡 Root cause analysis: Focus on {pure_causes[0]} as a fundamental driver")
        elif 'predict' in goal.lower():
            if causal_relationships:
                insights.append(f"💡 Predictive modeling: Use causal relationships to build more robust, interpretable models")
        
        return insights
    
    def generate_correlation_insights(self, pearson_corr: pd.DataFrame, spearman_corr: pd.DataFrame,
                                    strong_correlations: List[Dict], variables: List[str],
                                    goal: str) -> List[str]:
        """Generate insights from correlation analysis"""
        insights = []
        
        # Strong correlations analysis
        if strong_correlations:
            insights.append(f"Found {len(strong_correlations)} strong correlations (>0.5) among {len(variables)} variables")
            
            strongest = strong_correlations[0]
            correlation_direction = "positive" if strongest['pearson'] > 0 else "negative"
            insights.append(f"Strongest relationship: {strongest['variable1']} and {strongest['variable2']} show a strong {correlation_direction} correlation (r={strongest['pearson']:.3f})")
            
            # Business implications based on goal
            if 'customer' in goal.lower() or 'behavior' in goal.lower():
                insights.append(f"This relationship suggests that as {strongest['variable1']} increases, {strongest['variable2']} {'increases' if strongest['pearson'] > 0 else 'decreases'} proportionally")
            elif 'sales' in goal.lower() or 'revenue' in goal.lower():
                insights.append(f"Revenue drivers identified: focus on {strongest['variable1']} to impact {strongest['variable2']}")
            elif 'predict' in goal.lower():
                insights.append(f"{strongest['variable1']} can be used as a strong predictor for {strongest['variable2']}")
        else:
            insights.append(f"No strong correlations (>0.5) found among {len(variables)} variables - variables are relatively independent")
        
        # Correlation types analysis
        linear_strong = [c for c in strong_correlations if abs(c['pearson']) > 0.7]
        nonlinear_strong = [c for c in strong_correlations if abs(c['spearman']) > 0.7 and abs(c['pearson']) < 0.7]
        
        if linear_strong:
            insights.append(f"Found {len(linear_strong)} strong linear relationships - these follow predictable patterns")
            for rel in linear_strong[:2]:  # Top 2
                insights.append(f"Linear relationship: {rel['variable1']} ↔ {rel['variable2']} (suitable for linear modeling)")
        
        if nonlinear_strong:
            insights.append(f"Found {len(nonlinear_strong)} strong non-linear relationships - these require advanced modeling")
            for rel in nonlinear_strong[:2]:  # Top 2
                insights.append(f"Non-linear relationship: {rel['variable1']} ↔ {rel['variable2']} (consider polynomial or tree-based models)")
        
        # Multicollinearity analysis
        very_strong = [c for c in strong_correlations if abs(c['pearson']) > 0.9]
        if very_strong:
            insights.append(f"⚠️ Warning: {len(very_strong)} very strong correlations (>0.9) detected - potential multicollinearity issue")
            for rel in very_strong:
                insights.append(f"Highly correlated: {rel['variable1']} and {rel['variable2']} (r={rel['pearson']:.3f}) - consider removing one for modeling")
        
        # Actionable recommendations
        if len(strong_correlations) > 0:
            insights.append(f"💡 Key insight: Focus analysis on the top {min(3, len(strong_correlations))} correlations for maximum impact")
            
            # Goal-specific recommendations
            if 'optimization' in goal.lower():
                insights.append("Use these correlations to identify leverage points for system optimization")
            elif 'segmentation' in goal.lower():
                insights.append("Strong correlations suggest natural groupings in your data for segmentation")
            elif 'forecast' in goal.lower():
                insights.append("Correlated variables can improve forecasting accuracy when used together")
        
        return insights
    
    def generate_anomaly_detection_insights(self, anomalies: Dict, total_anomalies: int,
                                          variables: List[str], goal: str) -> List[str]:
        """Generate insights from anomaly detection"""
        insights = []
        
        insights.append(f"Detected {total_anomalies} anomalies across {len(variables)} variables")
        
        # Most anomalous variable
        most_anomalous = max(anomalies.keys(), key=lambda x: anomalies[x]['anomaly_percentage'])
        insights.append(f"Most anomalous variable: {most_anomalous} ({anomalies[most_anomalous]['anomaly_percentage']:.1f}% anomalies)")
        
        # Detection methods
        insights.append("Anomalies detected using Z-score (>3) and IQR (1.5 * IQR) methods")
        
        # Data quality
        avg_anomaly_rate = np.mean([anomalies[col]['anomaly_percentage'] for col in variables])
        if avg_anomaly_rate > 5:
            insights.append(f"High anomaly rate ({avg_anomaly_rate:.1f}%) suggests data quality issues")
        else:
            insights.append(f"Low anomaly rate ({avg_anomaly_rate:.1f}%) indicates good data quality")
        
        return insights
    
    def generate_root_cause_insights(self, root_causes: List[Dict], outcome_col: str,
                                   goal: str) -> List[str]:
        """Generate insights from root cause analysis"""
        insights = []
        
        if root_causes:
            top_cause = root_causes[0]
            insights.append(f"Primary root cause: {top_cause['cause']} (impact: {top_cause['impact']:.2f})")
        
        # Significant causes
        significant_causes = [rc for rc in root_causes if rc['significance'] == 'significant']
        insights.append(f"Found {len(significant_causes)} statistically significant causes for {outcome_col}")
        
        # Top causes
        top_3_causes = [rc['cause'] for rc in root_causes[:3]]
        insights.append(f"Top 3 root causes: {', '.join(top_3_causes)}")
        
        # Actionability
        insights.append("Root cause analysis helps identify factors that drive outcomes")
        insights.append("Focus on addressing the most impactful causes first")
        
        return insights
    
    def create_decision_tree_visualizations(self, feature_importance: pd.DataFrame,
                                          tree, X_train, y_train, feature_cols: List[str],
                                          target_col: str) -> List[Dict[str, Any]]:
        """Create visualizations for decision tree analysis"""
        graphs = []
        
        # 1. Feature importance bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=feature_importance['feature'],
            y=feature_importance['importance'],
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Decision Tree Feature Importance',
            xaxis_title='Features',
            yaxis_title='Importance Score',
            showlegend=False
        )
        
        graphs.append({
            'title': 'Feature Importance',
            'type': 'bar',
            'data': fig.to_json()
        })
        
        return graphs
    
    def create_feature_importance_visualizations(self, importance_df: pd.DataFrame,
                                               correlations: List[Dict], target_col: str) -> List[Dict[str, Any]]:
        """Create visualizations for feature importance analysis"""
        graphs = []
        
        # 1. Combined importance scores
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=importance_df['feature'],
            y=importance_df['combined_score'],
            marker_color='lightgreen',
            name='Combined Score'
        ))
        
        fig.update_layout(
            title='Feature Importance: Combined Score',
            xaxis_title='Features',
            yaxis_title='Combined Importance Score',
            showlegend=False
        )
        
        graphs.append({
            'title': 'Combined Feature Importance',
            'type': 'bar',
            'data': fig.to_json()
        })
        
        # 2. Importance method comparison
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            x=importance_df['feature'],
            y=importance_df['mi_normalized'],
            name='Mutual Information',
            marker_color='lightblue'
        ))
        
        fig2.add_trace(go.Bar(
            x=importance_df['feature'],
            y=importance_df['rf_normalized'],
            name='Random Forest',
            marker_color='lightcoral'
        ))
        
        fig2.add_trace(go.Bar(
            x=importance_df['feature'],
            y=importance_df['corr_normalized'],
            name='Correlation',
            marker_color='lightgreen'
        ))
        
        fig2.update_layout(
            title='Feature Importance: Method Comparison',
            xaxis_title='Features',
            yaxis_title='Normalized Importance Score',
            barmode='group',
            showlegend=True
        )
        
        graphs.append({
            'title': 'Method Comparison',
            'type': 'bar',
            'data': fig2.to_json()
        })
        
        return graphs
    
    def create_causal_inference_visualizations(self, causal_relationships: List[Dict],
                                             df: pd.DataFrame, variables: List[str]) -> List[Dict[str, Any]]:
        """Create visualizations for causal inference"""
        graphs = []
        
        # 1. Causal strength network
        fig = go.Figure()
        
        # Create network-like visualization
        causes = [r['cause'] for r in causal_relationships[:10]]  # Top 10
        effects = [r['effect'] for r in causal_relationships[:10]]
        strengths = [r['strength'] for r in causal_relationships[:10]]
        
        # Node positions (simplified)
        unique_nodes = list(set(causes + effects))
        node_positions = {node: (i, 0) for i, node in enumerate(unique_nodes)}
        
        # Add edges
        for i, rel in enumerate(causal_relationships[:10]):
            cause_pos = node_positions[rel['cause']]
            effect_pos = node_positions[rel['effect']]
            
            fig.add_trace(go.Scatter(
                x=[cause_pos[0], effect_pos[0]],
                y=[cause_pos[1], effect_pos[1]],
                mode='lines',
                line=dict(width=strengths[i]*10, color='gray'),
                showlegend=False
            ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=[pos[0] for pos in node_positions.values()],
            y=[pos[1] for pos in node_positions.values()],
            mode='markers+text',
            text=list(node_positions.keys()),
            textposition='top center',
            marker=dict(size=20, color='lightblue'),
            name='Variables'
        ))
        
        fig.update_layout(
            title='Causal Relationship Network',
            showlegend=True,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        graphs.append({
            'title': 'Causal Network',
            'type': 'network',
            'data': fig.to_json()
        })
        
        return graphs
    
    def create_correlation_visualizations(self, pearson_corr: pd.DataFrame, spearman_corr: pd.DataFrame,
                                        strong_correlations: List[Dict], variables: List[str]) -> List[Dict[str, Any]]:
        """Create visualizations for correlation analysis"""
        graphs = []
        
        # 1. Correlation heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pearson_corr.values,
            x=variables,
            y=variables,
            colorscale='RdBu',
            zmid=0,
            showscale=True
        ))
        
        fig.update_layout(
            title='Pearson Correlation Matrix',
            xaxis_title='Variables',
            yaxis_title='Variables'
        )
        
        graphs.append({
            'title': 'Correlation Heatmap',
            'type': 'heatmap',
            'data': fig.to_json()
        })
        
        # 2. Strong correlations bar chart
        if strong_correlations:
            correlation_labels = [f"{c['variable1']} ↔ {c['variable2']}" for c in strong_correlations[:10]]
            correlation_values = [c['pearson'] for c in strong_correlations[:10]]
            
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=correlation_labels,
                y=correlation_values,
                marker_color=['red' if v < 0 else 'blue' for v in correlation_values]
            ))
            
            fig2.update_layout(
                title='Strong Correlations',
                xaxis_title='Variable Pairs',
                yaxis_title='Correlation Coefficient',
                showlegend=False
            )
            
            graphs.append({
                'title': 'Strong Correlations',
                'type': 'bar',
                'data': fig2.to_json()
            })
        
        return graphs
    
    def create_anomaly_detection_visualizations(self, df: pd.DataFrame, anomalies: Dict,
                                              variables: List[str]) -> List[Dict[str, Any]]:
        """Create visualizations for anomaly detection"""
        graphs = []
        
        # 1. Anomaly count by variable
        anomaly_counts = [anomalies[col]['anomaly_count'] for col in variables]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=variables,
            y=anomaly_counts,
            marker_color='orange'
        ))
        
        fig.update_layout(
            title='Anomaly Count by Variable',
            xaxis_title='Variables',
            yaxis_title='Number of Anomalies',
            showlegend=False
        )
        
        graphs.append({
            'title': 'Anomaly Counts',
            'type': 'bar',
            'data': fig.to_json()
        })
        
        # 2. Box plot showing anomalies for first variable
        if variables:
            first_var = variables[0]
            data = df[first_var].dropna()
            anomaly_indices = anomalies[first_var]['combined_anomalies']
            
            fig2 = go.Figure()
            fig2.add_trace(go.Box(
                y=data,
                name=first_var,
                boxpoints='outliers'
            ))
            
            fig2.update_layout(
                title=f'Anomaly Detection: {first_var}',
                yaxis_title=first_var,
                showlegend=False
            )
            
            graphs.append({
                'title': f'Anomalies in {first_var}',
                'type': 'box',
                'data': fig2.to_json()
            })
        
        return graphs
    
    def create_root_cause_visualizations(self, root_causes: List[Dict], df: pd.DataFrame,
                                       outcome_col: str) -> List[Dict[str, Any]]:
        """Create visualizations for root cause analysis"""
        graphs = []
        
        # 1. Root cause impact
        causes = [rc['cause'] for rc in root_causes[:10]]
        impacts = [rc['impact'] for rc in root_causes[:10]]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=causes,
            y=impacts,
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title='Root Cause Impact Analysis',
            xaxis_title='Potential Causes',
            yaxis_title='Impact Score',
            showlegend=False
        )
        
        graphs.append({
            'title': 'Root Cause Impact',
            'type': 'bar',
            'data': fig.to_json()
        })
        
        # 2. Correlation vs Impact scatter
        correlations = [abs(rc['correlation']) for rc in root_causes]
        impacts = [rc['impact'] for rc in root_causes]
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=correlations,
            y=impacts,
            mode='markers+text',
            text=causes,
            textposition='top center',
            marker=dict(size=10, color='blue')
        ))
        
        fig2.update_layout(
            title='Root Cause Analysis: Correlation vs Impact',
            xaxis_title='Absolute Correlation',
            yaxis_title='Impact Score',
            showlegend=False
        )
        
        graphs.append({
            'title': 'Correlation vs Impact',
            'type': 'scatter',
            'data': fig2.to_json()
        })
        
        return graphs 