"""
Prescriptive Analytics Module

This module implements prescriptive analytics algorithms to answer "What should we do?"
Includes optimization algorithms (Linear Programming, Genetic Algorithm) and 
recommendation systems (Collaborative Filtering, Content-Based).

Created: 2025-01-21
"""

import pandas as pd
import numpy as np
from scipy.optimize import linprog, minimize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PrescriptiveAnalytics:
    """
    Prescriptive analytics for understanding what should be done
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    
    def perform_analysis(self, df: pd.DataFrame, algorithm: str, goal: str) -> Dict[str, Any]:
        """Main entry point for prescriptive analytics"""
        try:
            if algorithm == 'linear_programming':
                return self.perform_linear_programming(df, goal)
            elif algorithm == 'genetic_algorithm':
                return self.perform_genetic_algorithm(df, goal)
            elif algorithm == 'collaborative_filtering':
                return self.perform_collaborative_filtering(df, goal)
            elif algorithm == 'content_based_filtering':
                return self.perform_content_based_filtering(df, goal)
            elif algorithm == 'portfolio_optimization':
                return self.perform_portfolio_optimization(df, goal)
            elif algorithm == 'resource_allocation':
                return self.perform_resource_allocation(df, goal)
            else:
                return {'error': f'Unknown algorithm: {algorithm}'}
        except Exception as e:
            logger.error(f"Error in prescriptive analysis {algorithm}: {e}")
            return {'error': str(e)}
    
    def perform_linear_programming(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform linear programming optimization"""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) < 2:
                raise ValueError("Need at least 2 numerical columns for linear programming")
            
            # Use first column as objective function coefficients
            # Use second column as constraint coefficients
            c = df[numerical_cols[0]].values[:10]  # Limit to first 10 for demo
            A_ub = df[numerical_cols[1]].values[:10].reshape(1, -1)
            b_ub = [df[numerical_cols[1]].sum() * 0.8]  # 80% of total as constraint
            
            # Bounds for variables (non-negative)
            bounds = [(0, None) for _ in range(len(c))]
            
            # Solve linear programming problem
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if result.success:
                optimal_solution = result.x
                optimal_value = result.fun
                
                # Generate insights
                insights = self.generate_linear_programming_insights(
                    optimal_solution, optimal_value, numerical_cols, goal
                )
                
                # Create visualizations
                graphs = self.create_linear_programming_visualizations(
                    c, optimal_solution, numerical_cols
                )
                
                return {
                    'algorithm': 'Linear Programming',
                    'results': {
                        'optimal_solution': optimal_solution.tolist(),
                        'optimal_value': optimal_value,
                        'success': True,
                        'objective_coefficients': c.tolist(),
                        'variables': [f'x{i+1}' for i in range(len(c))],
                        'constraint_columns': numerical_cols[:2]
                    },
                    'insights': insights,
                    'graphs': graphs,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'algorithm': 'Linear Programming',
                    'error': 'Optimization failed to converge',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in linear programming: {e}")
            return {'error': str(e)}
    
    def perform_genetic_algorithm(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform genetic algorithm optimization"""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) < 1:
                raise ValueError("Need at least 1 numerical column for genetic algorithm")
            
            # Define objective function (maximize sum of weighted values)
            weights = df[numerical_cols[0]].values[:10]  # Limit to first 10
            
            def objective_function(x):
                return -np.sum(weights * x)  # Negative for maximization
            
            # Define constraints
            def constraint(x):
                return 1.0 - np.sum(x)  # Sum of x should be <= 1
            
            # Run genetic algorithm (simplified implementation)
            best_solution, best_fitness = self.simple_genetic_algorithm(
                objective_function, len(weights), constraint
            )
            
            # Generate insights
            insights = self.generate_genetic_algorithm_insights(
                best_solution, best_fitness, numerical_cols, goal
            )
            
            # Create visualizations
            graphs = self.create_genetic_algorithm_visualizations(
                weights, best_solution, numerical_cols
            )
            
            return {
                'algorithm': 'Genetic Algorithm',
                'results': {
                    'best_solution': best_solution.tolist(),
                    'best_fitness': -best_fitness,  # Convert back to positive
                    'weights': weights.tolist(),
                    'variables': [f'x{i+1}' for i in range(len(weights))],
                    'optimization_column': numerical_cols[0]
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in genetic algorithm: {e}")
            return {'error': str(e)}
    
    def simple_genetic_algorithm(self, objective_func, n_vars, constraint, 
                                pop_size=50, n_generations=100):
        """Simple genetic algorithm implementation"""
        # Initialize population
        population = np.random.random((pop_size, n_vars))
        
        # Normalize to satisfy constraint
        for i in range(pop_size):
            population[i] = population[i] / np.sum(population[i])
        
        best_solution = None
        best_fitness = float('inf')
        
        for generation in range(n_generations):
            # Evaluate fitness
            fitness = np.array([objective_func(ind) for ind in population])
            
            # Track best solution
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_fitness = fitness[min_idx]
                best_solution = population[min_idx].copy()
            
            # Selection (tournament selection)
            new_population = []
            for _ in range(pop_size):
                # Tournament selection
                tournament_size = 3
                tournament_indices = np.random.choice(pop_size, tournament_size, replace=False)
                tournament_fitness = fitness[tournament_indices]
                winner_idx = tournament_indices[np.argmin(tournament_fitness)]
                new_population.append(population[winner_idx].copy())
            
            population = np.array(new_population)
            
            # Mutation
            mutation_rate = 0.1
            for i in range(pop_size):
                if np.random.random() < mutation_rate:
                    mutation_strength = 0.1
                    population[i] += np.random.normal(0, mutation_strength, n_vars)
                    population[i] = np.abs(population[i])  # Keep positive
                    population[i] = population[i] / np.sum(population[i])  # Normalize
        
        return best_solution, best_fitness
    
    def perform_collaborative_filtering(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform collaborative filtering recommendations"""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) < 3:
                raise ValueError("Need at least 3 numerical columns for collaborative filtering")
            
            # Create user-item matrix (assuming first 3 numerical columns)
            user_item_matrix = df[numerical_cols[:3]].fillna(0).values
            
            # Compute user-user similarity
            user_similarity = cosine_similarity(user_item_matrix)
            
            # Generate recommendations for each user
            recommendations = []
            for user_idx in range(min(5, len(user_item_matrix))):  # Limit to first 5 users
                user_recommendations = self.get_user_recommendations(
                    user_idx, user_item_matrix, user_similarity
                )
                recommendations.append({
                    'user_id': user_idx,
                    'recommendations': user_recommendations
                })
            
            # Generate insights
            insights = self.generate_collaborative_filtering_insights(
                user_similarity, recommendations, numerical_cols, goal
            )
            
            # Create visualizations
            graphs = self.create_collaborative_filtering_visualizations(
                user_similarity, user_item_matrix, numerical_cols
            )
            
            return {
                'algorithm': 'Collaborative Filtering',
                'results': {
                    'recommendations': recommendations,
                    'user_similarity_matrix': user_similarity.tolist(),
                    'item_columns': numerical_cols[:3],
                    'n_users': len(user_item_matrix),
                    'n_items': len(numerical_cols[:3])
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in collaborative filtering: {e}")
            return {'error': str(e)}
    
    def get_user_recommendations(self, user_idx, user_item_matrix, user_similarity):
        """Get recommendations for a specific user"""
        # Get similar users
        similar_users = np.argsort(user_similarity[user_idx])[::-1][1:4]  # Top 3 similar users
        
        # Get items this user hasn't rated
        user_ratings = user_item_matrix[user_idx]
        unrated_items = np.where(user_ratings == 0)[0]
        
        # Calculate predicted ratings for unrated items
        recommendations = []
        for item_idx in unrated_items:
            predicted_rating = 0
            similarity_sum = 0
            
            for similar_user in similar_users:
                if user_item_matrix[similar_user, item_idx] > 0:
                    predicted_rating += (user_similarity[user_idx, similar_user] * 
                                       user_item_matrix[similar_user, item_idx])
                    similarity_sum += user_similarity[user_idx, similar_user]
            
            if similarity_sum > 0:
                predicted_rating /= similarity_sum
                recommendations.append({
                    'item_id': item_idx,
                    'predicted_rating': predicted_rating
                })
        
        # Sort by predicted rating
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return recommendations[:3]  # Top 3 recommendations
    
    def perform_content_based_filtering(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform content-based filtering recommendations"""
        try:
            # Look for text columns for content-based filtering
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(text_cols) == 0 and len(numerical_cols) < 2:
                raise ValueError("Need text columns or at least 2 numerical columns for content-based filtering")
            
            if len(text_cols) > 0:
                # Use text-based similarity
                text_data = df[text_cols[0]].fillna('').astype(str)
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_data)
                content_similarity = cosine_similarity(tfidf_matrix)
            else:
                # Use numerical features
                numerical_data = df[numerical_cols].fillna(0)
                scaled_data = self.scaler.fit_transform(numerical_data)
                content_similarity = cosine_similarity(scaled_data)
            
            # Generate recommendations
            recommendations = []
            for item_idx in range(min(5, len(content_similarity))):
                similar_items = np.argsort(content_similarity[item_idx])[::-1][1:4]
                recommendations.append({
                    'item_id': item_idx,
                    'similar_items': similar_items.tolist(),
                    'similarity_scores': content_similarity[item_idx][similar_items].tolist()
                })
            
            # Generate insights
            insights = self.generate_content_based_insights(
                content_similarity, recommendations, text_cols, numerical_cols, goal
            )
            
            # Create visualizations
            graphs = self.create_content_based_visualizations(
                content_similarity, text_cols, numerical_cols
            )
            
            return {
                'algorithm': 'Content-Based Filtering',
                'results': {
                    'recommendations': recommendations,
                    'content_similarity_matrix': content_similarity.tolist(),
                    'text_columns': text_cols,
                    'numerical_columns': numerical_cols,
                    'n_items': len(content_similarity)
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in content-based filtering: {e}")
            return {'error': str(e)}
    
    def perform_portfolio_optimization(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform portfolio optimization"""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) < 2:
                raise ValueError("Need at least 2 numerical columns for portfolio optimization")
            
            # Calculate returns and covariance matrix
            returns = df[numerical_cols].pct_change().dropna()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Optimize portfolio (minimize risk for given return)
            n_assets = len(numerical_cols)
            
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            def portfolio_return(weights):
                return np.sum(mean_returns * weights)
            
            # Constraints: weights sum to 1
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # Bounds: weights between 0 and 1
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess
            x0 = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(portfolio_variance, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                optimal_return = portfolio_return(optimal_weights)
                optimal_risk = np.sqrt(result.fun)
                
                # Generate insights
                insights = self.generate_portfolio_optimization_insights(
                    optimal_weights, optimal_return, optimal_risk, numerical_cols, goal
                )
                
                # Create visualizations
                graphs = self.create_portfolio_optimization_visualizations(
                    optimal_weights, mean_returns, cov_matrix, numerical_cols
                )
                
                return {
                    'algorithm': 'Portfolio Optimization',
                    'results': {
                        'optimal_weights': optimal_weights.tolist(),
                        'expected_return': optimal_return,
                        'risk_volatility': optimal_risk,
                        'assets': numerical_cols,
                        'mean_returns': mean_returns.tolist(),
                        'covariance_matrix': cov_matrix.values.tolist()
                    },
                    'insights': insights,
                    'graphs': graphs,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'algorithm': 'Portfolio Optimization',
                    'error': 'Optimization failed to converge',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return {'error': str(e)}
    
    def perform_resource_allocation(self, df: pd.DataFrame, goal: str) -> Dict[str, Any]:
        """Perform resource allocation optimization"""
        try:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) < 2:
                raise ValueError("Need at least 2 numerical columns for resource allocation")
            
            # Use first column as benefits, second as costs
            benefits = df[numerical_cols[0]].values
            costs = df[numerical_cols[1]].values
            
            # Calculate benefit-to-cost ratio
            benefit_cost_ratio = benefits / (costs + 1e-6)  # Add small value to avoid division by zero
            
            # Sort by benefit-to-cost ratio
            sorted_indices = np.argsort(benefit_cost_ratio)[::-1]
            
            # Allocate resources based on available budget
            total_budget = costs.sum() * 0.7  # Assume 70% of total cost as budget
            allocated_resources = []
            remaining_budget = total_budget
            
            for idx in sorted_indices:
                if costs[idx] <= remaining_budget:
                    allocated_resources.append({
                        'resource_id': idx,
                        'benefit': benefits[idx],
                        'cost': costs[idx],
                        'ratio': benefit_cost_ratio[idx]
                    })
                    remaining_budget -= costs[idx]
            
            total_allocated_benefit = sum([r['benefit'] for r in allocated_resources])
            total_allocated_cost = sum([r['cost'] for r in allocated_resources])
            
            # Generate insights
            insights = self.generate_resource_allocation_insights(
                allocated_resources, total_allocated_benefit, total_allocated_cost, 
                total_budget, numerical_cols, goal
            )
            
            # Create visualizations
            graphs = self.create_resource_allocation_visualizations(
                benefits, costs, benefit_cost_ratio, allocated_resources, numerical_cols
            )
            
            return {
                'algorithm': 'Resource Allocation',
                'results': {
                    'allocated_resources': allocated_resources,
                    'total_benefit': total_allocated_benefit,
                    'total_cost': total_allocated_cost,
                    'budget_utilized': total_allocated_cost / total_budget,
                    'benefit_columns': numerical_cols[:2],
                    'n_resources_allocated': len(allocated_resources),
                    'n_total_resources': len(benefits)
                },
                'insights': insights,
                'graphs': graphs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in resource allocation: {e}")
            return {'error': str(e)}
    
    def generate_linear_programming_insights(self, optimal_solution: np.ndarray, 
                                           optimal_value: float, columns: List[str], 
                                           goal: str) -> List[str]:
        """Generate insights from linear programming optimization"""
        insights = []
        
        insights.append(f"Optimal objective value: {optimal_value:.2f}")
        
        # Top variables in solution
        top_vars = np.argsort(optimal_solution)[::-1][:3]
        insights.append(f"Top variables: {', '.join([f'x{i+1}={optimal_solution[i]:.2f}' for i in top_vars])}")
        
        # Solution characteristics
        non_zero_vars = np.sum(optimal_solution > 1e-6)
        insights.append(f"Number of non-zero variables: {non_zero_vars}")
        
        if 'maximize' in goal.lower():
            insights.append("Solution maximizes the objective function")
        elif 'minimize' in goal.lower():
            insights.append("Solution minimizes the objective function")
        else:
            insights.append("Solution optimizes the objective function")
        
        insights.append("Linear programming provides exact optimal solution")
        
        return insights
    
    def generate_genetic_algorithm_insights(self, best_solution: np.ndarray, 
                                          best_fitness: float, columns: List[str], 
                                          goal: str) -> List[str]:
        """Generate insights from genetic algorithm optimization"""
        insights = []
        
        insights.append(f"Best fitness value: {best_fitness:.2f}")
        
        # Top variables in solution
        top_vars = np.argsort(best_solution)[::-1][:3]
        insights.append(f"Top variables: {', '.join([f'x{i+1}={best_solution[i]:.3f}' for i in top_vars])}")
        
        # Solution diversity
        solution_entropy = -np.sum(best_solution * np.log(best_solution + 1e-6))
        insights.append(f"Solution diversity (entropy): {solution_entropy:.2f}")
        
        insights.append("Genetic algorithm provides near-optimal solution")
        insights.append("Solution evolved through natural selection principles")
        
        return insights
    
    def generate_collaborative_filtering_insights(self, user_similarity: np.ndarray,
                                                recommendations: List[Dict], columns: List[str],
                                                goal: str) -> List[str]:
        """Generate insights from collaborative filtering"""
        insights = []
        
        # Average similarity
        avg_similarity = np.mean(user_similarity[np.triu_indices_from(user_similarity, k=1)])
        insights.append(f"Average user similarity: {avg_similarity:.3f}")
        
        # Number of recommendations
        total_recommendations = sum([len(r['recommendations']) for r in recommendations])
        insights.append(f"Total recommendations generated: {total_recommendations}")
        
        # Most similar users
        max_similarity = np.max(user_similarity[np.triu_indices_from(user_similarity, k=1)])
        insights.append(f"Highest user similarity: {max_similarity:.3f}")
        
        insights.append("Collaborative filtering leverages user behavior patterns")
        insights.append("Recommendations based on similar users' preferences")
        
        return insights
    
    def generate_content_based_insights(self, content_similarity: np.ndarray,
                                      recommendations: List[Dict], text_cols: List[str],
                                      numerical_cols: List[str], goal: str) -> List[str]:
        """Generate insights from content-based filtering"""
        insights = []
        
        # Average similarity
        avg_similarity = np.mean(content_similarity[np.triu_indices_from(content_similarity, k=1)])
        insights.append(f"Average content similarity: {avg_similarity:.3f}")
        
        # Number of recommendations
        total_recommendations = sum([len(r['similar_items']) for r in recommendations])
        insights.append(f"Total item recommendations: {total_recommendations}")
        
        # Feature type used
        if text_cols:
            insights.append(f"Content analysis based on text features: {text_cols[0]}")
        else:
            insights.append(f"Content analysis based on numerical features: {', '.join(numerical_cols[:2])}")
        
        insights.append("Content-based filtering uses item characteristics")
        insights.append("Recommendations based on item similarity")
        
        return insights
    
    def generate_portfolio_optimization_insights(self, optimal_weights: np.ndarray,
                                               optimal_return: float, optimal_risk: float,
                                               assets: List[str], goal: str) -> List[str]:
        """Generate insights from portfolio optimization"""
        insights = []
        
        insights.append(f"Expected portfolio return: {optimal_return*100:.2f}%")
        insights.append(f"Portfolio risk (volatility): {optimal_risk*100:.2f}%")
        
        # Sharpe ratio approximation
        sharpe_ratio = optimal_return / optimal_risk if optimal_risk > 0 else 0
        insights.append(f"Risk-adjusted return (Sharpe ratio): {sharpe_ratio:.2f}")
        
        # Top holdings
        top_holdings = np.argsort(optimal_weights)[::-1][:3]
        insights.append(f"Top holdings: {', '.join([f'{assets[i]} ({optimal_weights[i]*100:.1f}%)' for i in top_holdings])}")
        
        # Diversification
        diversification_ratio = 1 / np.sum(optimal_weights**2)
        insights.append(f"Diversification ratio: {diversification_ratio:.2f}")
        
        insights.append("Portfolio optimized for risk-return trade-off")
        
        return insights
    
    def generate_resource_allocation_insights(self, allocated_resources: List[Dict],
                                            total_benefit: float, total_cost: float,
                                            budget: float, columns: List[str], 
                                            goal: str) -> List[str]:
        """Generate insights from resource allocation"""
        insights = []
        
        insights.append(f"Total allocated benefit: {total_benefit:.2f}")
        insights.append(f"Total allocated cost: {total_cost:.2f}")
        insights.append(f"Budget utilization: {(total_cost/budget)*100:.1f}%")
        
        # Efficiency metrics
        if total_cost > 0:
            efficiency = total_benefit / total_cost
            insights.append(f"Resource efficiency (benefit/cost): {efficiency:.2f}")
        
        # Best resource
        if allocated_resources:
            best_resource = max(allocated_resources, key=lambda x: x['ratio'])
            insights.append(f"Best resource: ID {best_resource['resource_id']} (ratio: {best_resource['ratio']:.2f})")
        
        insights.append(f"Allocated {len(allocated_resources)} resources optimally")
        insights.append("Resource allocation maximizes benefit within budget constraints")
        
        return insights
    
    def create_linear_programming_visualizations(self, c: np.ndarray, optimal_solution: np.ndarray,
                                               columns: List[str]) -> List[Dict[str, Any]]:
        """Create visualizations for linear programming"""
        graphs = []
        
        # 1. Objective coefficients vs optimal solution
        variables = [f'x{i+1}' for i in range(len(c))]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=variables,
            y=c,
            name='Objective Coefficients',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Scatter(
            x=variables,
            y=optimal_solution,
            mode='markers+lines',
            name='Optimal Solution',
            marker=dict(size=10, color='red'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Linear Programming: Coefficients vs Optimal Solution',
            xaxis_title='Variables',
            yaxis_title='Objective Coefficients',
            yaxis2=dict(title='Optimal Values', overlaying='y', side='right'),
            showlegend=True
        )
        
        graphs.append({
            'title': 'Optimization Results',
            'type': 'mixed',
            'data': fig.to_json()
        })
        
        # 2. Solution distribution pie chart
        fig2 = go.Figure(data=[go.Pie(
            labels=variables,
            values=optimal_solution,
            hole=0.3
        )])
        
        fig2.update_layout(
            title='Optimal Solution Distribution',
            showlegend=True
        )
        
        graphs.append({
            'title': 'Solution Distribution',
            'type': 'pie',
            'data': fig2.to_json()
        })
        
        return graphs
    
    def create_genetic_algorithm_visualizations(self, weights: np.ndarray, best_solution: np.ndarray,
                                              columns: List[str]) -> List[Dict[str, Any]]:
        """Create visualizations for genetic algorithm"""
        graphs = []
        
        # 1. Weights vs solution
        variables = [f'x{i+1}' for i in range(len(weights))]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=variables,
            y=weights,
            name='Weights',
            marker_color='lightgreen'
        ))
        
        fig.add_trace(go.Scatter(
            x=variables,
            y=best_solution,
            mode='markers+lines',
            name='Best Solution',
            marker=dict(size=10, color='red'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Genetic Algorithm: Weights vs Best Solution',
            xaxis_title='Variables',
            yaxis_title='Weights',
            yaxis2=dict(title='Solution Values', overlaying='y', side='right'),
            showlegend=True
        )
        
        graphs.append({
            'title': 'GA Optimization Results',
            'type': 'mixed',
            'data': fig.to_json()
        })
        
        return graphs
    
    def create_collaborative_filtering_visualizations(self, user_similarity: np.ndarray,
                                                    user_item_matrix: np.ndarray,
                                                    columns: List[str]) -> List[Dict[str, Any]]:
        """Create visualizations for collaborative filtering"""
        graphs = []
        
        # 1. User similarity heatmap
        fig = go.Figure(data=go.Heatmap(
            z=user_similarity,
            colorscale='Viridis',
            showscale=True
        ))
        
        fig.update_layout(
            title='User Similarity Matrix',
            xaxis_title='Users',
            yaxis_title='Users'
        )
        
        graphs.append({
            'title': 'User Similarity Heatmap',
            'type': 'heatmap',
            'data': fig.to_json()
        })
        
        # 2. User-item matrix heatmap
        fig2 = go.Figure(data=go.Heatmap(
            z=user_item_matrix,
            colorscale='Blues',
            showscale=True
        ))
        
        fig2.update_layout(
            title='User-Item Interaction Matrix',
            xaxis_title='Items',
            yaxis_title='Users'
        )
        
        graphs.append({
            'title': 'User-Item Matrix',
            'type': 'heatmap',
            'data': fig2.to_json()
        })
        
        return graphs
    
    def create_content_based_visualizations(self, content_similarity: np.ndarray,
                                          text_cols: List[str], numerical_cols: List[str]) -> List[Dict[str, Any]]:
        """Create visualizations for content-based filtering"""
        graphs = []
        
        # 1. Content similarity heatmap
        fig = go.Figure(data=go.Heatmap(
            z=content_similarity,
            colorscale='Oranges',
            showscale=True
        ))
        
        fig.update_layout(
            title='Content Similarity Matrix',
            xaxis_title='Items',
            yaxis_title='Items'
        )
        
        graphs.append({
            'title': 'Content Similarity Heatmap',
            'type': 'heatmap',
            'data': fig.to_json()
        })
        
        # 2. Similarity distribution
        similarity_values = content_similarity[np.triu_indices_from(content_similarity, k=1)]
        
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=similarity_values,
            nbinsx=20,
            name='Similarity Distribution',
            marker_color='orange'
        ))
        
        fig2.update_layout(
            title='Content Similarity Distribution',
            xaxis_title='Similarity Score',
            yaxis_title='Frequency'
        )
        
        graphs.append({
            'title': 'Similarity Distribution',
            'type': 'histogram',
            'data': fig2.to_json()
        })
        
        return graphs
    
    def create_portfolio_optimization_visualizations(self, optimal_weights: np.ndarray,
                                                   mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                                                   assets: List[str]) -> List[Dict[str, Any]]:
        """Create visualizations for portfolio optimization"""
        graphs = []
        
        # 1. Portfolio weights pie chart
        fig = go.Figure(data=[go.Pie(
            labels=assets,
            values=optimal_weights,
            hole=0.3
        )])
        
        fig.update_layout(
            title='Optimal Portfolio Weights',
            showlegend=True
        )
        
        graphs.append({
            'title': 'Portfolio Allocation',
            'type': 'pie',
            'data': fig.to_json()
        })
        
        # 2. Risk-return scatter plot
        fig2 = go.Figure()
        
        # Individual assets
        individual_risks = np.sqrt(np.diag(cov_matrix))
        fig2.add_trace(go.Scatter(
            x=individual_risks,
            y=mean_returns,
            mode='markers+text',
            text=assets,
            textposition='top center',
            name='Individual Assets',
            marker=dict(size=10, color='blue')
        ))
        
        # Portfolio
        portfolio_return = np.sum(mean_returns * optimal_weights)
        portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        
        fig2.add_trace(go.Scatter(
            x=[portfolio_risk],
            y=[portfolio_return],
            mode='markers',
            name='Optimal Portfolio',
            marker=dict(size=15, color='red', symbol='star')
        ))
        
        fig2.update_layout(
            title='Risk-Return Analysis',
            xaxis_title='Risk (Volatility)',
            yaxis_title='Expected Return',
            showlegend=True
        )
        
        graphs.append({
            'title': 'Risk-Return Profile',
            'type': 'scatter',
            'data': fig2.to_json()
        })
        
        return graphs
    
    def create_resource_allocation_visualizations(self, benefits: np.ndarray, costs: np.ndarray,
                                                benefit_cost_ratio: np.ndarray, 
                                                allocated_resources: List[Dict],
                                                columns: List[str]) -> List[Dict[str, Any]]:
        """Create visualizations for resource allocation"""
        graphs = []
        
        # 1. Benefit vs Cost scatter plot
        fig = go.Figure()
        
        # All resources
        colors = ['red' if i in [r['resource_id'] for r in allocated_resources] else 'blue' 
                 for i in range(len(benefits))]
        
        fig.add_trace(go.Scatter(
            x=costs,
            y=benefits,
            mode='markers',
            marker=dict(size=8, color=colors),
            text=[f'Resource {i}' for i in range(len(benefits))],
            name='Resources'
        ))
        
        fig.update_layout(
            title='Resource Allocation: Benefits vs Costs',
            xaxis_title=f'Cost ({columns[1]})',
            yaxis_title=f'Benefit ({columns[0]})',
            showlegend=True
        )
        
        graphs.append({
            'title': 'Resource Allocation Map',
            'type': 'scatter',
            'data': fig.to_json()
        })
        
        # 2. Benefit-to-cost ratio bar chart
        resource_ids = list(range(len(benefit_cost_ratio)))
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=resource_ids,
            y=benefit_cost_ratio,
            marker_color=['red' if i in [r['resource_id'] for r in allocated_resources] else 'lightblue' 
                         for i in resource_ids],
            name='Benefit/Cost Ratio'
        ))
        
        fig2.update_layout(
            title='Benefit-to-Cost Ratio by Resource',
            xaxis_title='Resource ID',
            yaxis_title='Benefit/Cost Ratio',
            showlegend=True
        )
        
        graphs.append({
            'title': 'Resource Efficiency',
            'type': 'bar',
            'data': fig2.to_json()
        })
        
        return graphs 