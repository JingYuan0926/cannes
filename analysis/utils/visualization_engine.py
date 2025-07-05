"""
Visualization Engine Module

Provides comprehensive visualization capabilities for business intelligence,
automatically selecting appropriate chart types and generating insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class VisualizationEngine:
    """
    Comprehensive visualization engine for business intelligence and data analysis
    """
    
    def __init__(self):
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Chart type mappings
        self.chart_type_mapping = {
            'bar': self._create_bar_chart,
            'line': self._create_line_chart,
            'scatter': self._create_scatter_plot,
            'pie': self._create_pie_chart,
            'heatmap': self._create_heatmap,
            'box': self._create_box_plot,
            'histogram': self._create_histogram,
            'area': self._create_area_chart,
            'violin': self._create_violin_plot,
            'treemap': self._create_treemap,
            'sunburst': self._create_sunburst,
            'funnel': self._create_funnel_chart,
            'gauge': self._create_gauge_chart,
            'waterfall': self._create_waterfall_chart
        }
        
    def create_visualization(self, df, config):
        """
        Create visualization based on configuration
        
        Args:
            df (pd.DataFrame): Dataset to visualize
            config (dict): Visualization configuration
            
        Returns:
            dict: Visualization data including chart and metadata
        """
        try:
            chart_type = config.get('chart_type', 'bar')
            
            if chart_type not in self.chart_type_mapping:
                chart_type = self._recommend_chart_type(df, config)
            
            # Create the visualization
            chart_func = self.chart_type_mapping[chart_type]
            chart_data = chart_func(df, config)
            
            # Add metadata
            chart_data.update({
                'chart_type': chart_type,
                'data_shape': df.shape,
                'creation_timestamp': datetime.now().isoformat(),
                'business_context': config.get('business_context', ''),
                'insights': self._generate_chart_insights(df, config, chart_type)
            })
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return {'error': str(e)}
    
    def create_dashboard(self, df, analysis_results):
        """
        Create a comprehensive dashboard with multiple visualizations
        
        Args:
            df (pd.DataFrame): Dataset
            analysis_results (dict): Results from data analysis
            
        Returns:
            dict: Dashboard with multiple visualizations
        """
        try:
            dashboard = {
                'overview': self._create_overview_charts(df),
                'trends': self._create_trend_charts(df),
                'distributions': self._create_distribution_charts(df),
                'relationships': self._create_relationship_charts(df),
                'business_metrics': self._create_business_metric_charts(df, analysis_results)
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            return {'error': str(e)}
    
    def _recommend_chart_type(self, df, config):
        """Recommend appropriate chart type based on data characteristics"""
        try:
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            
            if not x_axis or not y_axis:
                return 'bar'
            
            x_type = df[x_axis].dtype
            y_type = df[y_axis].dtype
            
            # Time series data
            if pd.api.types.is_datetime64_any_dtype(x_type):
                return 'line'
            
            # Numerical vs Numerical
            if pd.api.types.is_numeric_dtype(x_type) and pd.api.types.is_numeric_dtype(y_type):
                return 'scatter'
            
            # Categorical vs Numerical
            if not pd.api.types.is_numeric_dtype(x_type) and pd.api.types.is_numeric_dtype(y_type):
                unique_categories = df[x_axis].nunique()
                if unique_categories <= 10:
                    return 'bar'
                else:
                    return 'box'
            
            # Categorical data for pie chart
            if config.get('show_proportions') and not pd.api.types.is_numeric_dtype(x_type):
                return 'pie'
            
            return 'bar'
            
        except Exception as e:
            logger.error(f"Error recommending chart type: {str(e)}")
            return 'bar'
    
    def _create_bar_chart(self, df, config):
        """Create bar chart"""
        try:
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            group_by = config.get('group_by')
            title = config.get('title', f'{y_axis} by {x_axis}')
            
            if group_by:
                fig = px.bar(df, x=x_axis, y=y_axis, color=group_by, title=title)
            else:
                # Aggregate data if needed
                if pd.api.types.is_numeric_dtype(df[y_axis]):
                    agg_data = df.groupby(x_axis)[y_axis].sum().reset_index()
                    fig = px.bar(agg_data, x=x_axis, y=y_axis, title=title)
                else:
                    fig = px.bar(df, x=x_axis, y=y_axis, title=title)
            
            fig.update_layout(
                xaxis_title=x_axis.replace('_', ' ').title(),
                yaxis_title=y_axis.replace('_', ' ').title(),
                showlegend=True if group_by else False
            )
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'chart_type': 'bar'
            }
            
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            return {'error': str(e)}
    
    def _create_line_chart(self, df, config):
        """Create line chart"""
        try:
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            group_by = config.get('group_by')
            title = config.get('title', f'{y_axis} over {x_axis}')
            
            # Sort by x-axis for proper line chart
            df_sorted = df.sort_values(x_axis)
            
            if group_by:
                fig = px.line(df_sorted, x=x_axis, y=y_axis, color=group_by, title=title)
            else:
                # Aggregate data if needed
                if len(df_sorted) > 1000:  # Reduce data points for performance
                    agg_data = df_sorted.groupby(x_axis)[y_axis].mean().reset_index()
                    fig = px.line(agg_data, x=x_axis, y=y_axis, title=title)
                else:
                    fig = px.line(df_sorted, x=x_axis, y=y_axis, title=title)
            
            fig.update_layout(
                xaxis_title=x_axis.replace('_', ' ').title(),
                yaxis_title=y_axis.replace('_', ' ').title()
            )
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'chart_type': 'line'
            }
            
        except Exception as e:
            logger.error(f"Error creating line chart: {str(e)}")
            return {'error': str(e)}
    
    def _create_scatter_plot(self, df, config):
        """Create scatter plot"""
        try:
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            group_by = config.get('group_by')
            size_by = config.get('size_by')
            title = config.get('title', f'{y_axis} vs {x_axis}')
            
            fig = px.scatter(
                df, 
                x=x_axis, 
                y=y_axis, 
                color=group_by,
                size=size_by,
                title=title,
                trendline="ols" if not group_by else None
            )
            
            fig.update_layout(
                xaxis_title=x_axis.replace('_', ' ').title(),
                yaxis_title=y_axis.replace('_', ' ').title()
            )
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'chart_type': 'scatter'
            }
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {str(e)}")
            return {'error': str(e)}
    
    def _create_pie_chart(self, df, config):
        """Create pie chart"""
        try:
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            title = config.get('title', f'Distribution of {x_axis}')
            
            # Aggregate data for pie chart
            if y_axis and pd.api.types.is_numeric_dtype(df[y_axis]):
                pie_data = df.groupby(x_axis)[y_axis].sum().reset_index()
                fig = px.pie(pie_data, values=y_axis, names=x_axis, title=title)
            else:
                value_counts = df[x_axis].value_counts().reset_index()
                value_counts.columns = [x_axis, 'count']
                fig = px.pie(value_counts, values='count', names=x_axis, title=title)
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'chart_type': 'pie'
            }
            
        except Exception as e:
            logger.error(f"Error creating pie chart: {str(e)}")
            return {'error': str(e)}
    
    def _create_heatmap(self, df, config):
        """Create heatmap"""
        try:
            title = config.get('title', 'Correlation Heatmap')
            
            # Select numerical columns for correlation
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) < 2:
                return {'error': 'Insufficient numerical columns for heatmap'}
            
            correlation_matrix = df[numerical_cols].corr()
            
            fig = px.imshow(
                correlation_matrix,
                labels=dict(color="Correlation"),
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                color_continuous_scale='RdBu',
                title=title
            )
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'chart_type': 'heatmap'
            }
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")
            return {'error': str(e)}
    
    def _create_box_plot(self, df, config):
        """Create box plot"""
        try:
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            title = config.get('title', f'{y_axis} Distribution by {x_axis}')
            
            fig = px.box(df, x=x_axis, y=y_axis, title=title)
            
            fig.update_layout(
                xaxis_title=x_axis.replace('_', ' ').title(),
                yaxis_title=y_axis.replace('_', ' ').title()
            )
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'chart_type': 'box'
            }
            
        except Exception as e:
            logger.error(f"Error creating box plot: {str(e)}")
            return {'error': str(e)}
    
    def _create_histogram(self, df, config):
        """Create histogram"""
        try:
            x_axis = config.get('x_axis')
            group_by = config.get('group_by')
            title = config.get('title', f'Distribution of {x_axis}')
            
            fig = px.histogram(
                df, 
                x=x_axis, 
                color=group_by,
                title=title,
                nbins=30
            )
            
            fig.update_layout(
                xaxis_title=x_axis.replace('_', ' ').title(),
                yaxis_title='Frequency'
            )
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'chart_type': 'histogram'
            }
            
        except Exception as e:
            logger.error(f"Error creating histogram: {str(e)}")
            return {'error': str(e)}
    
    def _create_area_chart(self, df, config):
        """Create area chart"""
        try:
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            group_by = config.get('group_by')
            title = config.get('title', f'{y_axis} over {x_axis}')
            
            df_sorted = df.sort_values(x_axis)
            
            fig = px.area(df_sorted, x=x_axis, y=y_axis, color=group_by, title=title)
            
            fig.update_layout(
                xaxis_title=x_axis.replace('_', ' ').title(),
                yaxis_title=y_axis.replace('_', ' ').title()
            )
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'chart_type': 'area'
            }
            
        except Exception as e:
            logger.error(f"Error creating area chart: {str(e)}")
            return {'error': str(e)}
    
    def _create_violin_plot(self, df, config):
        """Create violin plot"""
        try:
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            title = config.get('title', f'{y_axis} Distribution by {x_axis}')
            
            fig = px.violin(df, x=x_axis, y=y_axis, title=title)
            
            fig.update_layout(
                xaxis_title=x_axis.replace('_', ' ').title(),
                yaxis_title=y_axis.replace('_', ' ').title()
            )
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'chart_type': 'violin'
            }
            
        except Exception as e:
            logger.error(f"Error creating violin plot: {str(e)}")
            return {'error': str(e)}
    
    def _create_treemap(self, df, config):
        """Create treemap"""
        try:
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            title = config.get('title', f'Treemap of {x_axis}')
            
            # Aggregate data
            if y_axis and pd.api.types.is_numeric_dtype(df[y_axis]):
                tree_data = df.groupby(x_axis)[y_axis].sum().reset_index()
                fig = px.treemap(tree_data, path=[x_axis], values=y_axis, title=title)
            else:
                value_counts = df[x_axis].value_counts().reset_index()
                value_counts.columns = [x_axis, 'count']
                fig = px.treemap(value_counts, path=[x_axis], values='count', title=title)
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'chart_type': 'treemap'
            }
            
        except Exception as e:
            logger.error(f"Error creating treemap: {str(e)}")
            return {'error': str(e)}
    
    def _create_sunburst(self, df, config):
        """Create sunburst chart"""
        try:
            path_columns = config.get('path_columns', [])
            values_column = config.get('values_column')
            title = config.get('title', 'Sunburst Chart')
            
            if not path_columns:
                return {'error': 'Path columns required for sunburst chart'}
            
            if values_column and pd.api.types.is_numeric_dtype(df[values_column]):
                fig = px.sunburst(df, path=path_columns, values=values_column, title=title)
            else:
                # Create count-based sunburst
                df_counts = df.groupby(path_columns).size().reset_index(name='count')
                fig = px.sunburst(df_counts, path=path_columns, values='count', title=title)
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'chart_type': 'sunburst'
            }
            
        except Exception as e:
            logger.error(f"Error creating sunburst chart: {str(e)}")
            return {'error': str(e)}
    
    def _create_funnel_chart(self, df, config):
        """Create funnel chart"""
        try:
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            title = config.get('title', 'Funnel Chart')
            
            # Aggregate data
            funnel_data = df.groupby(x_axis)[y_axis].sum().reset_index()
            funnel_data = funnel_data.sort_values(y_axis, ascending=False)
            
            fig = go.Figure(go.Funnel(
                y=funnel_data[x_axis],
                x=funnel_data[y_axis],
                textinfo="value+percent initial"
            ))
            
            fig.update_layout(title=title)
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'chart_type': 'funnel'
            }
            
        except Exception as e:
            logger.error(f"Error creating funnel chart: {str(e)}")
            return {'error': str(e)}
    
    def _create_gauge_chart(self, df, config):
        """Create gauge chart"""
        try:
            value_column = config.get('value_column')
            title = config.get('title', 'Gauge Chart')
            max_value = config.get('max_value', 100)
            
            if not value_column or not pd.api.types.is_numeric_dtype(df[value_column]):
                return {'error': 'Numeric value column required for gauge chart'}
            
            current_value = df[value_column].mean()
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=current_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': title},
                delta={'reference': max_value * 0.5},
                gauge={'axis': {'range': [None, max_value]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, max_value * 0.5], 'color': "lightgray"},
                           {'range': [max_value * 0.5, max_value], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                     'thickness': 0.75, 'value': max_value * 0.9}}))
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'chart_type': 'gauge'
            }
            
        except Exception as e:
            logger.error(f"Error creating gauge chart: {str(e)}")
            return {'error': str(e)}
    
    def _create_waterfall_chart(self, df, config):
        """Create waterfall chart"""
        try:
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            title = config.get('title', 'Waterfall Chart')
            
            # Aggregate data
            waterfall_data = df.groupby(x_axis)[y_axis].sum().reset_index()
            
            fig = go.Figure(go.Waterfall(
                name="", orientation="v",
                measure=["relative"] * len(waterfall_data),
                x=waterfall_data[x_axis],
                textposition="outside",
                text=waterfall_data[y_axis],
                y=waterfall_data[y_axis],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            fig.update_layout(title=title)
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'chart_type': 'waterfall'
            }
            
        except Exception as e:
            logger.error(f"Error creating waterfall chart: {str(e)}")
            return {'error': str(e)}
    
    def _create_overview_charts(self, df):
        """Create overview charts for dashboard"""
        try:
            charts = []
            
            # Data summary chart
            summary_data = {
                'Total Records': len(df),
                'Total Columns': len(df.columns),
                'Missing Values': df.isnull().sum().sum(),
                'Duplicate Records': df.duplicated().sum()
            }
            
            fig = px.bar(
                x=list(summary_data.keys()),
                y=list(summary_data.values()),
                title="Dataset Overview"
            )
            
            charts.append({
                'title': 'Dataset Overview',
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_type': 'overview'
            })
            
            return charts
            
        except Exception as e:
            logger.error(f"Error creating overview charts: {str(e)}")
            return []
    
    def _create_trend_charts(self, df):
        """Create trend charts for dashboard"""
        try:
            charts = []
            
            # Find datetime columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(datetime_cols) > 0 and len(numerical_cols) > 0:
                date_col = datetime_cols[0]
                
                for num_col in numerical_cols[:3]:  # Limit to first 3 numerical columns
                    df_sorted = df.sort_values(date_col)
                    
                    fig = px.line(
                        df_sorted, 
                        x=date_col, 
                        y=num_col,
                        title=f'{num_col} Trend Over Time'
                    )
                    
                    charts.append({
                        'title': f'{num_col} Trend',
                        'chart_html': fig.to_html(include_plotlyjs='cdn'),
                        'chart_type': 'trend'
                    })
            
            return charts
            
        except Exception as e:
            logger.error(f"Error creating trend charts: {str(e)}")
            return []
    
    def _create_distribution_charts(self, df):
        """Create distribution charts for dashboard"""
        try:
            charts = []
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols[:3]:  # Limit to first 3 numerical columns
                fig = px.histogram(df, x=col, title=f'Distribution of {col}')
                
                charts.append({
                    'title': f'{col} Distribution',
                    'chart_html': fig.to_html(include_plotlyjs='cdn'),
                    'chart_type': 'distribution'
                })
            
            return charts
            
        except Exception as e:
            logger.error(f"Error creating distribution charts: {str(e)}")
            return []
    
    def _create_relationship_charts(self, df):
        """Create relationship charts for dashboard"""
        try:
            charts = []
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) >= 2:
                # Correlation heatmap
                correlation_matrix = df[numerical_cols].corr()
                
                fig = px.imshow(
                    correlation_matrix,
                    title="Correlation Matrix",
                    color_continuous_scale='RdBu'
                )
                
                charts.append({
                    'title': 'Correlation Matrix',
                    'chart_html': fig.to_html(include_plotlyjs='cdn'),
                    'chart_type': 'correlation'
                })
            
            return charts
            
        except Exception as e:
            logger.error(f"Error creating relationship charts: {str(e)}")
            return []
    
    def _create_business_metric_charts(self, df, analysis_results):
        """Create business metric charts for dashboard"""
        try:
            charts = []
            
            # Data quality metrics
            if 'business_metrics' in analysis_results:
                metrics = analysis_results['business_metrics']
                
                metric_names = []
                metric_values = []
                
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        metric_names.append(key.replace('_', ' ').title())
                        metric_values.append(value)
                
                if metric_names:
                    fig = px.bar(
                        x=metric_names,
                        y=metric_values,
                        title="Business Metrics"
                    )
                    
                    charts.append({
                        'title': 'Business Metrics',
                        'chart_html': fig.to_html(include_plotlyjs='cdn'),
                        'chart_type': 'business_metrics'
                    })
            
            return charts
            
        except Exception as e:
            logger.error(f"Error creating business metric charts: {str(e)}")
            return []
    
    def _generate_chart_insights(self, df, config, chart_type):
        """Generate insights for the chart"""
        try:
            insights = []
            
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            
            if x_axis and y_axis:
                # Basic insights based on chart type
                if chart_type == 'bar':
                    if pd.api.types.is_numeric_dtype(df[y_axis]):
                        max_value = df.groupby(x_axis)[y_axis].sum().max()
                        max_category = df.groupby(x_axis)[y_axis].sum().idxmax()
                        insights.append(f"Highest value: {max_category} with {max_value:.2f}")
                
                elif chart_type == 'line':
                    if pd.api.types.is_numeric_dtype(df[y_axis]):
                        trend_corr = df[y_axis].corr(pd.to_numeric(df[x_axis], errors='coerce'))
                        if abs(trend_corr) > 0.3:
                            direction = "increasing" if trend_corr > 0 else "decreasing"
                            insights.append(f"Shows {direction} trend over time")
                
                elif chart_type == 'scatter':
                    if (pd.api.types.is_numeric_dtype(df[x_axis]) and 
                        pd.api.types.is_numeric_dtype(df[y_axis])):
                        correlation = df[x_axis].corr(df[y_axis])
                        if abs(correlation) > 0.5:
                            relationship = "positive" if correlation > 0 else "negative"
                            insights.append(f"Strong {relationship} correlation: {correlation:.3f}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating chart insights: {str(e)}")
            return [] 