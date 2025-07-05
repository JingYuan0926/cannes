"""
Visualization Engine Module

This module provides comprehensive visualization capabilities for different types of analytics:
- Descriptive Analytics: What happened? (distributions, summaries, current state)
- Diagnostic Analytics: Why did it happen? (correlations, comparisons, breakdowns)
- Predictive Analytics: What will happen? (trends, forecasts, patterns)
- Prescriptive Analytics: What should we do? (recommendations, optimizations)

Supports multiple chart types: bar, line, scatter, pie, histogram, box, heatmap, area, violin, radar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import warnings
import logging
from datetime import datetime
import base64
from io import BytesIO
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class VisualizationEngine:
    """
    Comprehensive visualization engine for business analytics
    """
    
    def __init__(self):
        """Initialize the visualization engine"""
        # Set default plotly template
        pio.templates.default = "plotly_white"
        
        # Color palettes for different analytics types
        self.color_palettes = {
            'descriptive': px.colors.qualitative.Set3,
            'diagnostic': px.colors.qualitative.Dark2,
            'predictive': px.colors.sequential.Viridis,
            'prescriptive': px.colors.qualitative.Bold
        }
    
    def create_visualization(self, df, config):
        """
        Create a visualization based on the provided configuration
        
        Args:
            df (pd.DataFrame): The dataset to visualize
            config (dict): Configuration dictionary containing:
                - chart_type: Type of chart (bar, line, scatter, etc.)
                - x_axis: Column name for x-axis
                - y_axis: Column name for y-axis
                - title: Chart title
                - category: Analytics category (descriptive, diagnostic, etc.)
                - aggregation: How to aggregate data (sum, mean, count, none)
                - description: Description of what the chart shows
        
        Returns:
            dict: Visualization result with chart data and metadata
        """
        try:
            # Validate inputs
            if not self._validate_config(df, config):
                return None
            
            chart_type = config.get('chart_type', 'bar')
            category = config.get('category', 'descriptive')
            
            # Route to appropriate chart creation method
            chart_methods = {
                'bar': self._create_bar_chart,
                'line': self._create_line_chart,
                'scatter': self._create_scatter_chart,
                'pie': self._create_pie_chart,
                'histogram': self._create_histogram,
                'box': self._create_box_plot,
                'heatmap': self._create_heatmap,
                'area': self._create_area_chart,
                'violin': self._create_violin_plot,
                'radar': self._create_radar_chart
            }
            
            if chart_type not in chart_methods:
                chart_type = self._recommend_chart_type(df, config)
            
            # Create the chart
            chart_result = chart_methods[chart_type](df, config)
            
            if chart_result:
                # Add metadata
                chart_result.update({
                    'chart_type': chart_type,
                    'category': category,
                    'config': config,
                    'insights': self._generate_insights(df, config, chart_result),
                    'timestamp': datetime.now().isoformat()
                })
            
            return chart_result
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return None
    
    def _validate_config(self, df, config):
        """Validate configuration and data"""
        if df is None or df.empty:
            logger.error("DataFrame is empty or None")
            return False
        
        x_axis = config.get('x_axis')
        y_axis = config.get('y_axis')
        
        # Check if required columns exist
        if x_axis and x_axis not in df.columns:
            logger.error(f"Column '{x_axis}' not found in DataFrame")
            return False
        
        if y_axis and y_axis not in df.columns:
            logger.error(f"Column '{y_axis}' not found in DataFrame")
            return False
        
        return True
    
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
        """Create a bar chart"""
        try:
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            title = config.get('title', f'{y_axis} by {x_axis}')
            aggregation = config.get('aggregation', 'mean')
            category = config.get('category', 'descriptive')
            
            # Prepare data based on aggregation
            if aggregation == 'sum':
                plot_data = df.groupby(x_axis)[y_axis].sum().reset_index()
            elif aggregation == 'mean':
                plot_data = df.groupby(x_axis)[y_axis].mean().reset_index()
            elif aggregation == 'count':
                plot_data = df.groupby(x_axis)[y_axis].count().reset_index()
            else:
                plot_data = df[[x_axis, y_axis]].copy()
            
            # Create plotly bar chart
            fig = px.bar(
                plot_data, 
                x=x_axis, 
                y=y_axis,
                title=title,
                color_discrete_sequence=self.color_palettes[category]
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title=x_axis.replace('_', ' ').title(),
                yaxis_title=y_axis.replace('_', ' ').title(),
                showlegend=False,
                height=500
            )
            
            # Generate insights
            insights = self._generate_bar_chart_insights(plot_data, x_axis, y_axis)
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'data_summary': plot_data.describe().to_dict(),
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            return None
    
    def _create_line_chart(self, df, config):
        """Create a line chart"""
        try:
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            title = config.get('title', f'{y_axis} over {x_axis}')
            category = config.get('category', 'predictive')
            
            # Sort by x-axis for proper line chart
            plot_data = df.sort_values(x_axis)
            
            # Create plotly line chart
            fig = px.line(
                plot_data, 
                x=x_axis, 
                y=y_axis,
                title=title,
                color_discrete_sequence=self.color_palettes[category]
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title=x_axis.replace('_', ' ').title(),
                yaxis_title=y_axis.replace('_', ' ').title(),
                height=500
            )
            
            # Add trend line if it's time series
            if pd.api.types.is_datetime64_any_dtype(plot_data[x_axis].dtype):
                fig.add_scatter(
                    x=plot_data[x_axis], 
                    y=plot_data[y_axis].rolling(window=5).mean(),
                    mode='lines',
                    name='Trend',
                    line=dict(dash='dash')
                )
            
            # Generate insights
            insights = self._generate_line_chart_insights(plot_data, x_axis, y_axis)
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'data_summary': plot_data[[x_axis, y_axis]].describe().to_dict(),
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error creating line chart: {str(e)}")
            return None
    
    def _create_scatter_chart(self, df, config):
        """Create a scatter plot"""
        try:
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            title = config.get('title', f'{y_axis} vs {x_axis}')
            category = config.get('category', 'diagnostic')
            
            # Create plotly scatter plot
            fig = px.scatter(
                df, 
                x=x_axis, 
                y=y_axis,
                title=title,
                color_discrete_sequence=self.color_palettes[category]
            )
            
            # Add trend line
            fig.add_scatter(
                x=df[x_axis], 
                y=np.poly1d(np.polyfit(df[x_axis], df[y_axis], 1))(df[x_axis]),
                mode='lines',
                name='Trend Line',
                line=dict(dash='dash')
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title=x_axis.replace('_', ' ').title(),
                yaxis_title=y_axis.replace('_', ' ').title(),
                height=500
            )
            
            # Generate insights
            insights = self._generate_scatter_chart_insights(df, x_axis, y_axis)
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'correlation': df[x_axis].corr(df[y_axis]),
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error creating scatter chart: {str(e)}")
            return None
    
    def _create_pie_chart(self, df, config):
        """Create a pie chart"""
        try:
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            title = config.get('title', f'{y_axis} distribution by {x_axis}')
            category = config.get('category', 'descriptive')
            
            # Aggregate data
            plot_data = df.groupby(x_axis)[y_axis].sum().reset_index()
            
            # Create plotly pie chart
            fig = px.pie(
                plot_data, 
                values=y_axis, 
                names=x_axis,
                title=title,
                color_discrete_sequence=self.color_palettes[category]
            )
            
            # Update layout
            fig.update_layout(height=500)
            
            # Generate insights
            insights = self._generate_pie_chart_insights(plot_data, x_axis, y_axis)
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'data_summary': plot_data.describe().to_dict(),
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error creating pie chart: {str(e)}")
            return None
    
    def _create_histogram(self, df, config):
        """Create a histogram"""
        try:
            x_axis = config.get('x_axis')
            title = config.get('title', f'Distribution of {x_axis}')
            category = config.get('category', 'descriptive')
            
            # Create plotly histogram
            fig = px.histogram(
                df, 
                x=x_axis,
                title=title,
                color_discrete_sequence=self.color_palettes[category]
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title=x_axis.replace('_', ' ').title(),
                yaxis_title='Frequency',
                height=500
            )
            
            # Generate insights
            insights = self._generate_histogram_insights(df, x_axis)
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'statistics': df[x_axis].describe().to_dict(),
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error creating histogram: {str(e)}")
            return None
    
    def _create_box_plot(self, df, config):
        """Create a box plot"""
        try:
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            title = config.get('title', f'{y_axis} distribution by {x_axis}')
            category = config.get('category', 'diagnostic')
            
            # Create plotly box plot
            fig = px.box(
                df, 
                x=x_axis, 
                y=y_axis,
                title=title,
                color_discrete_sequence=self.color_palettes[category]
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title=x_axis.replace('_', ' ').title() if x_axis else '',
                yaxis_title=y_axis.replace('_', ' ').title(),
                height=500
            )
            
            # Generate insights
            insights = self._generate_box_plot_insights(df, x_axis, y_axis)
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'statistics': df[y_axis].describe().to_dict(),
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error creating box plot: {str(e)}")
            return None
    
    def _create_heatmap(self, df, config):
        """Create a correlation heatmap"""
        try:
            title = config.get('title', 'Correlation Matrix')
            category = config.get('category', 'diagnostic')
            
            # Get numerical columns only
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) < 2:
                return None
            
            # Calculate correlation matrix
            corr_matrix = df[numerical_cols].corr()
            
            # Create plotly heatmap
            fig = px.imshow(
                corr_matrix,
                title=title,
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            
            # Update layout
            fig.update_layout(height=500)
            
            # Generate insights
            insights = self._generate_heatmap_insights(corr_matrix)
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'correlation_matrix': corr_matrix.to_dict(),
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")
            return None
    
    def _create_area_chart(self, df, config):
        """Create an area chart"""
        try:
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            title = config.get('title', f'{y_axis} over {x_axis}')
            category = config.get('category', 'predictive')
            
            # Sort by x-axis
            plot_data = df.sort_values(x_axis)
            
            # Create plotly area chart
            fig = px.area(
                plot_data, 
                x=x_axis, 
                y=y_axis,
                title=title,
                color_discrete_sequence=self.color_palettes[category]
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title=x_axis.replace('_', ' ').title(),
                yaxis_title=y_axis.replace('_', ' ').title(),
                height=500
            )
            
            # Generate insights
            insights = self._generate_area_chart_insights(plot_data, x_axis, y_axis)
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'data_summary': plot_data[[x_axis, y_axis]].describe().to_dict(),
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error creating area chart: {str(e)}")
            return None
    
    def _create_violin_plot(self, df, config):
        """Create a violin plot"""
        try:
            x_axis = config.get('x_axis')
            y_axis = config.get('y_axis')
            title = config.get('title', f'{y_axis} distribution by {x_axis}')
            category = config.get('category', 'diagnostic')
            
            # Create plotly violin plot
            fig = px.violin(
                df, 
                x=x_axis, 
                y=y_axis,
                title=title,
                color_discrete_sequence=self.color_palettes[category]
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title=x_axis.replace('_', ' ').title() if x_axis else '',
                yaxis_title=y_axis.replace('_', ' ').title(),
                height=500
            )
            
            # Generate insights
            insights = self._generate_violin_plot_insights(df, x_axis, y_axis)
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'statistics': df[y_axis].describe().to_dict(),
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error creating violin plot: {str(e)}")
            return None
    
    def _create_radar_chart(self, df, config):
        """Create a radar chart"""
        try:
            title = config.get('title', 'Radar Chart')
            category = config.get('category', 'prescriptive')
            
            # Get numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns[:6]  # Limit to 6 dimensions
            
            if len(numerical_cols) < 3:
                return None
            
            # Normalize data for radar chart
            normalized_data = df[numerical_cols].mean()
            
            # Create plotly radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_data.values,
                theta=normalized_data.index,
                fill='toself',
                name='Average Values'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, normalized_data.max()]
                    )),
                showlegend=True,
                title=title,
                height=500
            )
            
            # Generate insights
            insights = self._generate_radar_chart_insights(normalized_data)
            
            return {
                'chart_html': fig.to_html(include_plotlyjs='cdn'),
                'chart_json': fig.to_json(),
                'data_summary': normalized_data.to_dict(),
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error creating radar chart: {str(e)}")
            return None
    
    def create_dashboard(self, df, chart_configs):
        """Create a dashboard with multiple charts"""
        try:
            dashboard_charts = []
            
            for config in chart_configs:
                chart_result = self.create_visualization(df, config)
                if chart_result:
                    dashboard_charts.append(chart_result)
            
            return {
                'charts': dashboard_charts,
                'total_charts': len(dashboard_charts),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            return None
    
    def export_chart(self, chart_result, format='html'):
        """Export chart in different formats"""
        try:
            if format == 'html':
                return chart_result.get('chart_html')
            elif format == 'json':
                return chart_result.get('chart_json')
            elif format == 'png':
                # Convert to PNG (requires kaleido)
                fig = go.Figure(json.loads(chart_result.get('chart_json')))
                img_bytes = fig.to_image(format="png")
                return base64.b64encode(img_bytes).decode()
            elif format == 'svg':
                # Convert to SVG
                fig = go.Figure(json.loads(chart_result.get('chart_json')))
                return fig.to_image(format="svg").decode()
            else:
                return chart_result.get('chart_html')
                
        except Exception as e:
            logger.error(f"Error exporting chart: {str(e)}")
            return None
    
    def _generate_insights(self, df, config, chart_result):
        """Generate insights based on chart type and data"""
        chart_type = config.get('chart_type', 'bar')
        
        insight_methods = {
            'bar': lambda: self._generate_bar_chart_insights(df, config.get('x_axis'), config.get('y_axis')),
            'line': lambda: self._generate_line_chart_insights(df, config.get('x_axis'), config.get('y_axis')),
            'scatter': lambda: self._generate_scatter_chart_insights(df, config.get('x_axis'), config.get('y_axis')),
            'pie': lambda: self._generate_pie_chart_insights(df, config.get('x_axis'), config.get('y_axis')),
            'histogram': lambda: self._generate_histogram_insights(df, config.get('x_axis')),
            'box': lambda: self._generate_box_plot_insights(df, config.get('x_axis'), config.get('y_axis')),
            'heatmap': lambda: self._generate_heatmap_insights(df.select_dtypes(include=[np.number]).corr()),
            'area': lambda: self._generate_area_chart_insights(df, config.get('x_axis'), config.get('y_axis')),
            'violin': lambda: self._generate_violin_plot_insights(df, config.get('x_axis'), config.get('y_axis')),
            'radar': lambda: self._generate_radar_chart_insights(df.select_dtypes(include=[np.number]).mean())
        }
        
        try:
            return insight_methods.get(chart_type, lambda: [])()
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return []
    
    def _generate_bar_chart_insights(self, df, x_axis, y_axis):
        """Generate insights for bar charts"""
        try:
            insights = []
            
            if x_axis and y_axis:
                # Group and analyze
                grouped = df.groupby(x_axis)[y_axis].agg(['mean', 'sum', 'count'])
                
                # Highest and lowest values
                highest_category = grouped['sum'].idxmax()
                lowest_category = grouped['sum'].idxmin()
                
                insights.append(f"Highest {y_axis}: {highest_category} ({grouped.loc[highest_category, 'sum']:.2f})")
                insights.append(f"Lowest {y_axis}: {lowest_category} ({grouped.loc[lowest_category, 'sum']:.2f})")
                
                # Variation analysis
                cv = grouped['sum'].std() / grouped['sum'].mean()
                if cv > 0.5:
                    insights.append(f"High variation across {x_axis} categories (CV: {cv:.2f})")
                else:
                    insights.append(f"Moderate variation across {x_axis} categories (CV: {cv:.2f})")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating bar chart insights: {str(e)}")
            return []
    
    def _generate_line_chart_insights(self, df, x_axis, y_axis):
        """Generate insights for line charts"""
        try:
            insights = []
            
            if x_axis and y_axis:
                # Trend analysis
                if pd.api.types.is_numeric_dtype(df[x_axis]):
                    correlation = df[x_axis].corr(df[y_axis])
                    if correlation > 0.3:
                        insights.append(f"Strong positive trend: {y_axis} increases with {x_axis}")
                    elif correlation < -0.3:
                        insights.append(f"Strong negative trend: {y_axis} decreases with {x_axis}")
                    else:
                        insights.append(f"Weak or no clear trend between {x_axis} and {y_axis}")
                
                # Volatility analysis
                volatility = df[y_axis].std() / df[y_axis].mean()
                if volatility > 0.2:
                    insights.append(f"High volatility in {y_axis} (CV: {volatility:.2f})")
                else:
                    insights.append(f"Low volatility in {y_axis} (CV: {volatility:.2f})")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating line chart insights: {str(e)}")
            return []
    
    def _generate_scatter_chart_insights(self, df, x_axis, y_axis):
        """Generate insights for scatter plots"""
        try:
            insights = []
            
            if x_axis and y_axis:
                # Correlation analysis
                correlation = df[x_axis].corr(df[y_axis])
                
                if abs(correlation) > 0.7:
                    strength = "strong"
                elif abs(correlation) > 0.3:
                    strength = "moderate"
                else:
                    strength = "weak"
                
                direction = "positive" if correlation > 0 else "negative"
                insights.append(f"{strength.title()} {direction} correlation: {correlation:.3f}")
                
                # Outlier detection
                z_scores_x = np.abs((df[x_axis] - df[x_axis].mean()) / df[x_axis].std())
                z_scores_y = np.abs((df[y_axis] - df[y_axis].mean()) / df[y_axis].std())
                outliers = ((z_scores_x > 2) | (z_scores_y > 2)).sum()
                
                if outliers > 0:
                    insights.append(f"Found {outliers} potential outliers in the data")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating scatter chart insights: {str(e)}")
            return []
    
    def _generate_pie_chart_insights(self, df, x_axis, y_axis):
        """Generate insights for pie charts"""
        try:
            insights = []
            
            if x_axis and y_axis:
                # Concentration analysis
                grouped = df.groupby(x_axis)[y_axis].sum()
                total = grouped.sum()
                
                # Top contributor
                top_contributor = grouped.idxmax()
                top_percentage = (grouped.max() / total) * 100
                
                insights.append(f"Largest segment: {top_contributor} ({top_percentage:.1f}%)")
                
                # Concentration ratio
                top_3_percentage = (grouped.nlargest(3).sum() / total) * 100
                insights.append(f"Top 3 segments represent {top_3_percentage:.1f}% of total")
                
                # Diversity analysis
                if len(grouped) > 5 and top_percentage < 30:
                    insights.append("Well-distributed across categories")
                elif top_percentage > 60:
                    insights.append("Highly concentrated in one category")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating pie chart insights: {str(e)}")
            return []
    
    def _generate_histogram_insights(self, df, x_axis):
        """Generate insights for histograms"""
        try:
            insights = []
            
            if x_axis:
                data = df[x_axis].dropna()
                
                # Distribution shape
                skewness = data.skew()
                if skewness > 1:
                    insights.append(f"Right-skewed distribution (skewness: {skewness:.2f})")
                elif skewness < -1:
                    insights.append(f"Left-skewed distribution (skewness: {skewness:.2f})")
                else:
                    insights.append(f"Approximately normal distribution (skewness: {skewness:.2f})")
                
                # Central tendency
                mean_val = data.mean()
                median_val = data.median()
                
                if abs(mean_val - median_val) / data.std() > 0.5:
                    insights.append(f"Mean ({mean_val:.2f}) differs significantly from median ({median_val:.2f})")
                else:
                    insights.append(f"Mean ({mean_val:.2f}) and median ({median_val:.2f}) are similar")
                
                # Outliers
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                outliers = ((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).sum()
                
                if outliers > 0:
                    insights.append(f"Found {outliers} outliers ({outliers/len(data)*100:.1f}% of data)")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating histogram insights: {str(e)}")
            return []
    
    def _generate_box_plot_insights(self, df, x_axis, y_axis):
        """Generate insights for box plots"""
        try:
            insights = []
            
            if y_axis:
                if x_axis:
                    # Compare across categories
                    grouped = df.groupby(x_axis)[y_axis]
                    
                    # Highest and lowest medians
                    medians = grouped.median()
                    highest_median = medians.idxmax()
                    lowest_median = medians.idxmin()
                    
                    insights.append(f"Highest median {y_axis}: {highest_median}")
                    insights.append(f"Lowest median {y_axis}: {lowest_median}")
                    
                    # Variability comparison
                    stds = grouped.std()
                    most_variable = stds.idxmax()
                    least_variable = stds.idxmin()
                    
                    insights.append(f"Most variable: {most_variable}")
                    insights.append(f"Least variable: {least_variable}")
                else:
                    # Single variable analysis
                    data = df[y_axis].dropna()
                    q1 = data.quantile(0.25)
                    q3 = data.quantile(0.75)
                    iqr = q3 - q1
                    
                    insights.append(f"IQR: {iqr:.2f} (Q1: {q1:.2f}, Q3: {q3:.2f})")
                    
                    # Outliers
                    outliers = ((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).sum()
                    if outliers > 0:
                        insights.append(f"Found {outliers} outliers")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating box plot insights: {str(e)}")
            return []
    
    def _generate_heatmap_insights(self, corr_matrix):
        """Generate insights for heatmaps"""
        try:
            insights = []
            
            # Find strongest correlations
            corr_values = corr_matrix.values
            np.fill_diagonal(corr_values, 0)  # Remove self-correlations
            
            # Get indices of strongest positive and negative correlations
            max_corr_idx = np.unravel_index(np.argmax(corr_values), corr_values.shape)
            min_corr_idx = np.unravel_index(np.argmin(corr_values), corr_values.shape)
            
            max_corr = corr_values[max_corr_idx]
            min_corr = corr_values[min_corr_idx]
            
            if max_corr > 0.5:
                var1 = corr_matrix.index[max_corr_idx[0]]
                var2 = corr_matrix.columns[max_corr_idx[1]]
                insights.append(f"Strongest positive correlation: {var1} & {var2} ({max_corr:.3f})")
            
            if min_corr < -0.5:
                var1 = corr_matrix.index[min_corr_idx[0]]
                var2 = corr_matrix.columns[min_corr_idx[1]]
                insights.append(f"Strongest negative correlation: {var1} & {var2} ({min_corr:.3f})")
            
            # Overall correlation strength
            avg_abs_corr = np.mean(np.abs(corr_values))
            if avg_abs_corr > 0.3:
                insights.append(f"High overall correlation between variables (avg: {avg_abs_corr:.3f})")
            else:
                insights.append(f"Low overall correlation between variables (avg: {avg_abs_corr:.3f})")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating heatmap insights: {str(e)}")
            return []
    
    def _generate_area_chart_insights(self, df, x_axis, y_axis):
        """Generate insights for area charts"""
        try:
            insights = []
            
            if x_axis and y_axis:
                # Cumulative analysis
                cumulative_values = df[y_axis].cumsum()
                total = cumulative_values.iloc[-1]
                
                # Find where 50% and 80% of total is reached
                fifty_percent_idx = (cumulative_values >= total * 0.5).idxmax()
                eighty_percent_idx = (cumulative_values >= total * 0.8).idxmax()
                
                insights.append(f"50% of total reached at index {fifty_percent_idx}")
                insights.append(f"80% of total reached at index {eighty_percent_idx}")
                
                # Growth pattern
                growth_rate = (df[y_axis].iloc[-1] - df[y_axis].iloc[0]) / df[y_axis].iloc[0] * 100
                insights.append(f"Overall growth rate: {growth_rate:.1f}%")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating area chart insights: {str(e)}")
            return []
    
    def _generate_violin_plot_insights(self, df, x_axis, y_axis):
        """Generate insights for violin plots"""
        try:
            insights = []
            
            if y_axis:
                if x_axis:
                    # Compare distributions across categories
                    grouped = df.groupby(x_axis)[y_axis]
                    
                    # Skewness comparison
                    skewness = grouped.skew()
                    most_skewed = skewness.abs().idxmax()
                    
                    insights.append(f"Most skewed distribution: {most_skewed}")
                    
                    # Kurtosis comparison
                    kurtosis = grouped.apply(lambda x: x.kurtosis())
                    most_peaked = kurtosis.idxmax()
                    
                    insights.append(f"Most peaked distribution: {most_peaked}")
                else:
                    # Single distribution analysis
                    data = df[y_axis].dropna()
                    skewness = data.skew()
                    kurtosis = data.kurtosis()
                    
                    insights.append(f"Distribution skewness: {skewness:.2f}")
                    insights.append(f"Distribution kurtosis: {kurtosis:.2f}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating violin plot insights: {str(e)}")
            return []
    
    def _generate_radar_chart_insights(self, normalized_data):
        """Generate insights for radar charts"""
        try:
            insights = []
            
            # Highest and lowest dimensions
            highest_dim = normalized_data.idxmax()
            lowest_dim = normalized_data.idxmin()
            
            insights.append(f"Strongest dimension: {highest_dim} ({normalized_data[highest_dim]:.2f})")
            insights.append(f"Weakest dimension: {lowest_dim} ({normalized_data[lowest_dim]:.2f})")
            
            # Balance analysis
            coefficient_of_variation = normalized_data.std() / normalized_data.mean()
            if coefficient_of_variation < 0.2:
                insights.append("Well-balanced across all dimensions")
            else:
                insights.append("Significant imbalance across dimensions")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating radar chart insights: {str(e)}")
            return [] 