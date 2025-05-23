"""
Risk assessment module for stock portfolios
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

class PortfolioRiskAnalyzer:
    """
    Class for analyzing portfolio risk metrics
    """
    
    def __init__(self):
        """Initialize risk analyzer"""
        # Risk-free rate (US 10-year treasury yield as approximation)
        self.risk_free_rate = 0.03  # 3% as default
    
    def calculate_portfolio_volatility(self, symbols, weights, period="1y"):
        """
        Calculate portfolio volatility based on historical data
        
        Parameters:
            symbols (list): List of stock symbols
            weights (list): List of weights for each symbol
            period (str): Historical period (default: "1y")
            
        Returns:
            float: Portfolio volatility (annualized)
        """
        # Get historical data
        end_date = datetime.now()
        data = yf.download(symbols, period=period)['Adj Close']
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Calculate covariance matrix
        cov_matrix = returns.cov() * 252  # Annualized
        
        # Calculate portfolio variance
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        
        # Calculate portfolio volatility
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        return portfolio_volatility
    
    def calculate_correlation_matrix(self, symbols, period="1y"):
        """
        Calculate correlation matrix for stock symbols
        
        Parameters:
            symbols (list): List of stock symbols
            period (str): Historical period (default: "1y")
            
        Returns:
            DataFrame: Correlation matrix
        """
        # Get historical data
        data = yf.download(symbols, period=period)['Adj Close']
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        
        return correlation_matrix
    
    def calculate_sharpe_ratio(self, symbols, weights, period="1y"):
        """
        Calculate Sharpe ratio for the portfolio
        
        Parameters:
            symbols (list): List of stock symbols
            weights (list): List of weights for each symbol
            period (str): Historical period (default: "1y")
            
        Returns:
            float: Sharpe ratio
        """
        # Get historical data
        data = yf.download(symbols, period=period)['Adj Close']
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)
        
        # Calculate expected return (annualized)
        expected_return = portfolio_returns.mean() * 252
        
        # Calculate portfolio volatility (annualized)
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio
        sharpe_ratio = (expected_return - self.risk_free_rate) / portfolio_volatility
        
        return sharpe_ratio
    
    def calculate_value_at_risk(self, symbols, weights, confidence_level=0.95, period="1y"):
        """
        Calculate Value at Risk (VaR) for the portfolio
        
        Parameters:
            symbols (list): List of stock symbols
            weights (list): List of weights for each symbol
            confidence_level (float): Confidence level (default: 0.95)
            period (str): Historical period (default: "1y")
            
        Returns:
            float: Value at Risk
        """
        # Get historical data
        data = yf.download(symbols, period=period)['Adj Close']
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)
        
        # Calculate VaR
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        return abs(var)
    
    def create_correlation_heatmap(self, correlation_matrix):
        """
        Create correlation heatmap
        
        Parameters:
            correlation_matrix (DataFrame): Correlation matrix
            
        Returns:
            Figure: Plotly figure
        """
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            aspect="auto"
        )
        
        fig.update_layout(
            title="Stock Correlation Matrix",
            height=600,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        return fig
    
    def create_risk_return_scatter(self, symbols, period="1y"):
        """
        Create risk-return scatter plot
        
        Parameters:
            symbols (list): List of stock symbols
            period (str): Historical period (default: "1y")
            
        Returns:
            Figure: Plotly figure
        """
        # Get historical data
        data = yf.download(symbols, period=period)['Adj Close']
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Calculate expected return and volatility for each stock
        expected_returns = returns.mean() * 252  # Annualized
        volatilities = returns.std() * np.sqrt(252)  # Annualized
        
        # Create risk-return dataframe
        risk_return_df = pd.DataFrame({
            'Symbol': symbols,
            'Expected Return': expected_returns.values,
            'Volatility': volatilities.values
        })
        
        # Create scatter plot
        fig = px.scatter(
            risk_return_df,
            x='Volatility',
            y='Expected Return',
            text='Symbol',
            color='Volatility',
            color_continuous_scale='Viridis',
            title="Risk-Return Profile"
        )
        
        fig.update_traces(textposition='top center')
        
        fig.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title="Risk (Volatility)",
            yaxis_title="Expected Return",
            coloraxis_showscale=False
        )
        
        return fig
    
    def assess_portfolio_risk(self, symbols, weights, portfolio_value, period="1y"):
        """
        Comprehensive portfolio risk assessment
        
        Parameters:
            symbols (list): List of stock symbols
            weights (list): List of weights for each symbol
            portfolio_value (float): Total portfolio value
            period (str): Historical period (default: "1y")
            
        Returns:
            dict: Risk assessment metrics and visualizations
        """
        # Calculate risk metrics
        try:
            volatility = self.calculate_portfolio_volatility(symbols, weights, period)
            sharpe_ratio = self.calculate_sharpe_ratio(symbols, weights, period)
            correlation_matrix = self.calculate_correlation_matrix(symbols, period)
            var_95 = self.calculate_value_at_risk(symbols, weights, 0.95, period)
            var_99 = self.calculate_value_at_risk(symbols, weights, 0.99, period)
            
            # Calculate diversification score based on correlation
            avg_correlation = correlation_matrix.values.sum() / (correlation_matrix.shape[0] ** 2)
            diversification_score = 1 - abs(avg_correlation)
            
            # Create visualizations
            correlation_heatmap = self.create_correlation_heatmap(correlation_matrix)
            risk_return_scatter = self.create_risk_return_scatter(symbols, period)
            
            # Determine risk level
            if volatility < 0.15:
                risk_level = "Low"
                risk_description = "Your portfolio has relatively low volatility, indicating conservative investment choices."
            elif volatility < 0.25:
                risk_level = "Moderate"
                risk_description = "Your portfolio has a balanced risk profile with moderate volatility."
            else:
                risk_level = "High"
                risk_description = "Your portfolio has high volatility, indicating aggressive investment choices."
            
            # Determine Sharpe ratio interpretation
            if sharpe_ratio < 0.5:
                sharpe_description = "Poor risk-adjusted returns. Consider rebalancing your portfolio."
            elif sharpe_ratio < 1.0:
                sharpe_description = "Below-average risk-adjusted returns. There may be room for improvement."
            elif sharpe_ratio < 1.5:
                sharpe_description = "Good risk-adjusted returns. Your portfolio is performing reasonably well."
            else:
                sharpe_description = "Excellent risk-adjusted returns. Your portfolio is effectively balancing risk and return."
            
            # Determine diversification recommendation
            if diversification_score < 0.3:
                diversification_recommendation = "Your portfolio lacks diversification. Consider adding assets with lower correlation."
            elif diversification_score < 0.6:
                diversification_recommendation = "Your portfolio has moderate diversification. You could improve by adding more varied investments."
            else:
                diversification_recommendation = "Your portfolio is well-diversified. Continue to monitor correlations over time."
            
            # Calculate potential loss amount
            potential_loss_95 = var_95 * portfolio_value
            potential_loss_99 = var_99 * portfolio_value
            
            # Package results
            result = {
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "correlation_matrix": correlation_matrix,
                "diversification_score": diversification_score,
                "var_95": var_95,
                "var_99": var_99,
                "potential_loss_95": potential_loss_95,
                "potential_loss_99": potential_loss_99,
                "risk_level": risk_level,
                "risk_description": risk_description,
                "sharpe_description": sharpe_description,
                "diversification_recommendation": diversification_recommendation,
                "correlation_heatmap": correlation_heatmap,
                "risk_return_scatter": risk_return_scatter
            }
            
            return result
        
        except Exception as e:
            return {
                "error": str(e)
            }