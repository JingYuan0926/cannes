"""
Descriptive Analytics Module

This module implements descriptive analytics algorithms to answer "What happened?"
Includes clustering algorithms (K-Means, DBSCAN) and dimensionality reduction (PCA, t-SNE).

Created: 2025-01-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class DescriptiveAnalytics:
    """
    Descriptive analytics for understanding what happened in the data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def perform_analysis(self, df: pd.DataFrame, algorithm: str, goal: str) -> Dict[str, Any]:
        """Main entry point for descriptive analytics"""
        try:
            if algorithm == 'kmeans_clustering':
                return self.perform_kmeans_clustering(df, goal)
            elif algorithm == 'dbscan_clustering':
                return self.perform_dbscan_clustering(df, goal)
            elif algorithm == 'pca_analysis':
                return self.perform_pca_analysis(df, goal)
            elif algorithm == 'tsne_analysis':
                return self.perform_tsne_analysis(df, goal)
            else:
                return {'error': f'Unknown algorithm: {algorithm}'}
        except Exception as e:
            logger.error(f"Error in descriptive analysis {algorithm}: {e}")
            return {'error': str(e)}
    
    def prepare_data_for_clustering(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare numerical data for clustering"""
        # Select only numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            raise ValueError("Need at least 2 numerical columns for clustering")
        
        # Handle missing values
        df_clean = df[numerical_cols].fillna(df[numerical_cols].mean())
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(df_clean)
        
        return scaled_data, numerical_cols
    
    def perform_kmeans_clustering(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform K-Means clustering analysis"""
        try:
            scaled_data, feature_names = self.prepare_data_for_clustering(df)
            
            # Determine optimal number of clusters using elbow method
            inertias = []
            silhouette_scores = []
            k_range = range(2, min(11, len(df) // 2))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled_data)
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(scaled_data, cluster_labels))
            
            # Choose optimal k (highest silhouette score)
            optimal_k = k_range[np.argmax(silhouette_scores)]
            
            # Perform final clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # Add cluster labels to dataframe
            df_with_clusters = df.copy()
            df_with_clusters['Cluster'] = cluster_labels
            
            # Generate insights
            insights = self.generate_clustering_insights(df_with_clusters, feature_names, optimal_k, goal)
            
            # Create visualizations
            graphs = self.create_clustering_visualizations(
                scaled_data, cluster_labels, feature_names, optimal_k, 
                inertias, silhouette_scores, k_range
            )
            
            return {
                'algorithm': 'K-Means Clustering',
                'results': {
                    'optimal_clusters': optimal_k,
                    'silhouette_score': max(silhouette_scores),
                    'cluster_centers': kmeans.cluster_centers_.tolist(),
                    'cluster_sizes': np.bincount(cluster_labels).tolist(),
                    'feature_names': feature_names
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in K-Means clustering: {e}")
            return {'error': str(e)}
    
    def perform_dbscan_clustering(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform DBSCAN clustering analysis"""
        try:
            scaled_data, feature_names = self.prepare_data_for_clustering(df)
            
            # Determine optimal eps using knee method
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=4)
            neighbors_fit = neighbors.fit(scaled_data)
            distances, indices = neighbors_fit.kneighbors(scaled_data)
            distances = np.sort(distances, axis=0)
            distances = distances[:, 1]
            
            # Use knee point as eps (simplified approach)
            eps = np.percentile(distances, 90)
            
            # Perform DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=4)
            cluster_labels = dbscan.fit_predict(scaled_data)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            # Add cluster labels to dataframe
            df_with_clusters = df.copy()
            df_with_clusters['Cluster'] = cluster_labels
            
            # Generate insights
            insights = self.generate_dbscan_insights(df_with_clusters, feature_names, n_clusters, n_noise, goal)
            
            # Create visualizations
            graphs = self.create_dbscan_visualizations(scaled_data, cluster_labels, feature_names, eps)
            
            return {
                'algorithm': 'DBSCAN Clustering',
                'results': {
                    'n_clusters': n_clusters,
                    'n_noise_points': n_noise,
                    'eps': eps,
                    'cluster_sizes': np.bincount(cluster_labels[cluster_labels >= 0]).tolist(),
                    'feature_names': feature_names
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in DBSCAN clustering: {e}")
            return {'error': str(e)}
    
    def perform_pca_analysis(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform Principal Component Analysis"""
        try:
            scaled_data, feature_names = self.prepare_data_for_clustering(df)
            
            # Perform PCA
            pca = PCA()
            pca_result = pca.fit_transform(scaled_data)
            
            # Calculate explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Determine optimal number of components (95% variance)
            n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
            
            # Get feature contributions to first 2 components
            feature_contributions = pd.DataFrame(
                pca.components_[:2].T,
                columns=['PC1', 'PC2'],
                index=feature_names
            )
            
            # Generate insights
            insights = self.generate_pca_insights(
                explained_variance, n_components_95, feature_contributions, goal
            )
            
            # Create visualizations
            graphs = self.create_pca_visualizations(
                pca_result, explained_variance, cumulative_variance, 
                feature_contributions, feature_names
            )
            
            return {
                'algorithm': 'Principal Component Analysis',
                'results': {
                    'explained_variance_ratio': explained_variance.tolist(),
                    'cumulative_variance': cumulative_variance.tolist(),
                    'n_components_95_variance': n_components_95,
                    'feature_contributions': feature_contributions.to_dict(),
                    'feature_names': feature_names
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in PCA analysis: {e}")
            return {'error': str(e)}
    
    def perform_tsne_analysis(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform t-SNE analysis"""
        try:
            scaled_data, feature_names = self.prepare_data_for_clustering(df)
            
            # Limit data size for t-SNE (performance reasons)
            if len(scaled_data) > 1000:
                indices = np.random.choice(len(scaled_data), 1000, replace=False)
                scaled_data_sample = scaled_data[indices]
                df_sample = df.iloc[indices]
            else:
                scaled_data_sample = scaled_data
                df_sample = df
            
            # Perform t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(scaled_data_sample)-1))
            tsne_result = tsne.fit_transform(scaled_data_sample)
            
            # Generate insights
            insights = self.generate_tsne_insights(tsne_result, feature_names, goal)
            
            # Create visualizations
            graphs = self.create_tsne_visualizations(tsne_result, df_sample, feature_names)
            
            return {
                'algorithm': 't-SNE Analysis',
                'results': {
                    'tsne_coordinates': tsne_result.tolist(),
                    'feature_names': feature_names,
                    'sample_size': len(scaled_data_sample)
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in t-SNE analysis: {e}")
            return {'error': str(e)}
    
    def generate_clustering_insights(self, df: pd.DataFrame, feature_names: List[str], 
                                   n_clusters: int, goal: str) -> List[str]:
        """Generate insights from clustering analysis"""
        insights = []
        
        # Cluster size analysis
        cluster_sizes = df['Cluster'].value_counts().sort_index()
        insights.append(f"Identified {n_clusters} distinct clusters in the data")
        insights.append(f"Cluster sizes range from {cluster_sizes.min()} to {cluster_sizes.max()} data points")
        
        # Feature analysis by cluster
        for feature in feature_names[:3]:  # Top 3 features
            cluster_means = df.groupby('Cluster')[feature].mean()
            dominant_cluster = cluster_means.idxmax()
            insights.append(f"Cluster {dominant_cluster} has the highest average {feature} ({cluster_means.max():.2f})")
        
        # Goal-specific insights
        if 'segment' in goal.lower() or 'group' in goal.lower():
            insights.append("Clustering reveals natural customer/data segments for targeted strategies")
        elif 'pattern' in goal.lower():
            insights.append("Distinct patterns identified that can inform decision-making")
        
        return insights
    
    def generate_dbscan_insights(self, df: pd.DataFrame, feature_names: List[str], 
                                n_clusters: int, n_noise: int, goal: str) -> List[str]:
        """Generate insights from DBSCAN analysis"""
        insights = []
        
        insights.append(f"DBSCAN identified {n_clusters} dense clusters")
        insights.append(f"Found {n_noise} outlier points ({n_noise/len(df)*100:.1f}% of data)")
        
        if n_noise > 0:
            insights.append("Outliers may represent anomalies or special cases requiring attention")
        
        if n_clusters > 0:
            cluster_sizes = df[df['Cluster'] >= 0]['Cluster'].value_counts()
            insights.append(f"Cluster sizes vary from {cluster_sizes.min()} to {cluster_sizes.max()} points")
        
        return insights
    
    def generate_pca_insights(self, explained_variance: np.ndarray, n_components_95: int,
                             feature_contributions: pd.DataFrame, goal: str) -> List[str]:
        """Generate insights from PCA analysis"""
        insights = []
        
        insights.append(f"First component explains {explained_variance[0]*100:.1f}% of data variance")
        insights.append(f"First two components explain {sum(explained_variance[:2])*100:.1f}% of total variance")
        insights.append(f"Need {n_components_95} components to capture 95% of variance")
        
        # Feature importance in first component
        pc1_importance = feature_contributions['PC1'].abs().sort_values(ascending=False)
        top_feature = pc1_importance.index[0]
        insights.append(f"'{top_feature}' contributes most to the first principal component")
        
        # Dimensionality insights
        if n_components_95 < len(feature_contributions):
            reduction_pct = (1 - n_components_95/len(feature_contributions)) * 100
            insights.append(f"Can reduce dimensionality by {reduction_pct:.1f}% while preserving 95% of information")
        
        return insights
    
    def generate_tsne_insights(self, tsne_result: np.ndarray, feature_names: List[str], 
                              goal: str) -> List[str]:
        """Generate insights from t-SNE analysis"""
        insights = []
        
        insights.append("t-SNE reveals non-linear relationships in high-dimensional data")
        
        # Analyze spread
        x_range = tsne_result[:, 0].max() - tsne_result[:, 0].min()
        y_range = tsne_result[:, 1].max() - tsne_result[:, 1].min()
        insights.append(f"Data spreads across {x_range:.1f} x {y_range:.1f} in t-SNE space")
        
        # Density analysis
        from scipy.spatial.distance import pdist
        distances = pdist(tsne_result)
        avg_distance = np.mean(distances)
        insights.append(f"Average distance between points: {avg_distance:.2f}")
        
        if 'visualization' in goal.lower():
            insights.append("t-SNE provides excellent visualization of complex data relationships")
        
        return insights
    
    def create_clustering_visualizations(self, scaled_data: np.ndarray, cluster_labels: np.ndarray,
                                       feature_names: List[str], n_clusters: int,
                                       inertias: List[float], silhouette_scores: List[float],
                                       k_range: range) -> List[Dict[str, Any]]:
        """Create visualizations for clustering analysis"""
        graphs = []
        
        # 1. Cluster scatter plot (first 2 features)
        fig = go.Figure()
        
        for cluster in range(n_clusters):
            cluster_data = scaled_data[cluster_labels == cluster]
            fig.add_trace(go.Scatter(
                x=cluster_data[:, 0],
                y=cluster_data[:, 1],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(size=8, opacity=0.7)
            ))
        
        fig.update_layout(
            title='K-Means Clustering Results',
            xaxis_title=feature_names[0] if len(feature_names) > 0 else 'Feature 1',
            yaxis_title=feature_names[1] if len(feature_names) > 1 else 'Feature 2',
            showlegend=True
        )
        
        graphs.append({
            'title': 'K-Means Clustering Results',
            'type': 'scatter',
            'data': fig.to_json()
        })
        
        # 2. Elbow curve
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(k_range),
            y=inertias,
            mode='lines+markers',
            name='Inertia',
            line=dict(color='blue')
        ))
        
        fig2.update_layout(
            title='Elbow Method for Optimal K',
            xaxis_title='Number of Clusters (K)',
            yaxis_title='Inertia',
            showlegend=True
        )
        
        graphs.append({
            'title': 'Elbow Method for Optimal K',
            'type': 'line',
            'data': fig2.to_json()
        })
        
        # 3. Silhouette scores
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=list(k_range),
            y=silhouette_scores,
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='green')
        ))
        
        fig3.update_layout(
            title='Silhouette Score by Number of Clusters',
            xaxis_title='Number of Clusters (K)',
            yaxis_title='Silhouette Score',
            showlegend=True
        )
        
        graphs.append({
            'title': 'Silhouette Score Analysis',
            'type': 'line',
            'data': fig3.to_json()
        })
        
        return graphs
    
    def create_dbscan_visualizations(self, scaled_data: np.ndarray, cluster_labels: np.ndarray,
                                   feature_names: List[str], eps: float) -> List[Dict[str, Any]]:
        """Create visualizations for DBSCAN analysis"""
        graphs = []
        
        # DBSCAN scatter plot
        fig = go.Figure()
        
        unique_labels = set(cluster_labels)
        colors = px.colors.qualitative.Set1
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                # Noise points
                cluster_data = scaled_data[cluster_labels == label]
                fig.add_trace(go.Scatter(
                    x=cluster_data[:, 0],
                    y=cluster_data[:, 1],
                    mode='markers',
                    name='Noise',
                    marker=dict(size=6, color='black', symbol='x')
                ))
            else:
                cluster_data = scaled_data[cluster_labels == label]
                fig.add_trace(go.Scatter(
                    x=cluster_data[:, 0],
                    y=cluster_data[:, 1],
                    mode='markers',
                    name=f'Cluster {label}',
                    marker=dict(size=8, color=colors[i % len(colors)], opacity=0.7)
                ))
        
        fig.update_layout(
            title=f'DBSCAN Clustering Results (eps={eps:.3f})',
            xaxis_title=feature_names[0] if len(feature_names) > 0 else 'Feature 1',
            yaxis_title=feature_names[1] if len(feature_names) > 1 else 'Feature 2',
            showlegend=True
        )
        
        graphs.append({
            'title': 'DBSCAN Clustering Results',
            'type': 'scatter',
            'data': fig.to_json()
        })
        
        return graphs
    
    def create_pca_visualizations(self, pca_result: np.ndarray, explained_variance: np.ndarray,
                                cumulative_variance: np.ndarray, feature_contributions: pd.DataFrame,
                                feature_names: List[str]) -> List[Dict[str, Any]]:
        """Create visualizations for PCA analysis"""
        graphs = []
        
        # 1. PCA scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            mode='markers',
            name='Data Points',
            marker=dict(size=6, opacity=0.6)
        ))
        
        fig.update_layout(
            title='PCA: First Two Principal Components',
            xaxis_title=f'PC1 ({explained_variance[0]*100:.1f}% variance)',
            yaxis_title=f'PC2 ({explained_variance[1]*100:.1f}% variance)',
            showlegend=True
        )
        
        graphs.append({
            'title': 'PCA Scatter Plot',
            'type': 'scatter',
            'data': fig.to_json()
        })
        
        # 2. Explained variance
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=[f'PC{i+1}' for i in range(len(explained_variance))],
            y=explained_variance * 100,
            name='Individual Variance',
            marker_color='lightblue'
        ))
        
        fig2.add_trace(go.Scatter(
            x=[f'PC{i+1}' for i in range(len(cumulative_variance))],
            y=cumulative_variance * 100,
            mode='lines+markers',
            name='Cumulative Variance',
            line=dict(color='red'),
            yaxis='y2'
        ))
        
        fig2.update_layout(
            title='PCA Explained Variance',
            xaxis_title='Principal Components',
            yaxis_title='Individual Variance (%)',
            yaxis2=dict(title='Cumulative Variance (%)', overlaying='y', side='right'),
            showlegend=True
        )
        
        graphs.append({
            'title': 'PCA Explained Variance',
            'type': 'bar',
            'data': fig2.to_json()
        })
        
        # 3. Feature contributions biplot
        fig3 = go.Figure()
        
        # Add feature vectors
        for i, feature in enumerate(feature_names):
            fig3.add_trace(go.Scatter(
                x=[0, feature_contributions.loc[feature, 'PC1']],
                y=[0, feature_contributions.loc[feature, 'PC2']],
                mode='lines+markers+text',
                name=feature,
                text=['', feature],
                textposition='top center',
                line=dict(width=2),
                marker=dict(size=[0, 8])
            ))
        
        fig3.update_layout(
            title='PCA Feature Contributions (Biplot)',
            xaxis_title='PC1',
            yaxis_title='PC2',
            showlegend=True
        )
        
        graphs.append({
            'title': 'PCA Feature Contributions',
            'type': 'scatter',
            'data': fig3.to_json()
        })
        
        return graphs
    
    def create_tsne_visualizations(self, tsne_result: np.ndarray, df: pd.DataFrame,
                                 feature_names: List[str]) -> List[Dict[str, Any]]:
        """Create visualizations for t-SNE analysis"""
        graphs = []
        
        # t-SNE scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tsne_result[:, 0],
            y=tsne_result[:, 1],
            mode='markers',
            name='Data Points',
            marker=dict(size=6, opacity=0.6),
            text=[f'Point {i}' for i in range(len(tsne_result))],
            hovertemplate='t-SNE1: %{x}<br>t-SNE2: %{y}<br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title='t-SNE Visualization',
            xaxis_title='t-SNE Component 1',
            yaxis_title='t-SNE Component 2',
            showlegend=True
        )
        
        graphs.append({
            'title': 't-SNE Visualization',
            'type': 'scatter',
            'data': fig.to_json()
        })
        
        return graphs 