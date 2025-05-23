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
        try:
            # 确保符号列表非空
            if not symbols or len(symbols) == 0:
                return 0.0
                
            # 获取历史数据
            data = yf.download(symbols, period=period)
            
            # 确保能够获取到数据
            if data.empty:
                return 0.0
                
            # 获取价格数据
            price_data = None
            
            # 处理数据结构 - 多只股票情况
            if isinstance(data, pd.DataFrame) and len(data.columns) > 0:
                if 'Adj Close' in data.columns:
                    price_data = data['Adj Close']
                elif 'Close' in data.columns:
                    price_data = data['Close']
                
                # 单只股票时数据结构可能不同
                elif len(symbols) == 1:
                    # 通过索引检查所有可能的列
                    for col in data.columns:
                        if 'Adj Close' in col:
                            price_data = data[col]
                            break
                        elif 'Close' in col:
                            price_data = data[col]
                            break
            
            # 如果仍然无法获取价格数据，则尝试其他方法
            if price_data is None:
                # 手动为每个股票构建收盘价数据
                price_data = pd.DataFrame()
                for symbol in symbols:
                    try:
                        stock_data = yf.download(symbol, period=period)
                        if not stock_data.empty:
                            if 'Adj Close' in stock_data.columns:
                                price_data[symbol] = stock_data['Adj Close']
                            else:
                                price_data[symbol] = stock_data['Close']
                    except Exception:
                        # 如果无法获取数据，填充零值
                        price_data[symbol] = 0
            
            # 如果仍然没有数据，返回零
            if price_data is None or price_data.empty:
                return 0.0
                
            # 计算收益率
            returns = price_data.pct_change().dropna()
            
            # 如果收益率数据不足，返回零
            if returns.empty or len(returns) < 2:
                return 0.0
                
            # 计算协方差矩阵
            cov_matrix = returns.cov() * 252  # 年化
            
            # 确保权重和协方差矩阵尺寸匹配
            if len(weights) != cov_matrix.shape[0]:
                weights = [1.0/cov_matrix.shape[0]] * cov_matrix.shape[0]
                
            # 计算投资组合方差
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            
            # 计算投资组合波动率
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            return portfolio_volatility
            
        except Exception as e:
            # 发生任何错误时返回零
            return 0.0
    
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
        data = yf.download(symbols, period=period)
        
        # 确保能够获取到数据
        if data.empty:
            raise Exception("无法获取股票历史数据")
            
        # 确保有收盘价数据
        if 'Adj Close' in data.columns:
            price_data = data['Adj Close']
        elif 'Close' in data.columns:
            price_data = data['Close']
        else:
            # 如果是单一股票，数据结构可能不同
            if len(symbols) == 1 and 'Adj Close' in data:
                price_data = data['Adj Close']
            elif len(symbols) == 1 and 'Close' in data:
                price_data = data['Close']
            else:
                raise Exception("无法获取股票价格数据")
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
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
        try:
            # 確保符號列表非空
            if not symbols or len(symbols) == 0:
                return 0.0
                
            # 獲取歷史數據
            data = yf.download(symbols, period=period)
            
            # 確保能夠獲取到數據
            if data.empty:
                return 0.0
                
            # 獲取價格數據
            price_data = None
            
            # 處理數據結構 - 多隻股票情況
            if isinstance(data, pd.DataFrame) and len(data.columns) > 0:
                if 'Adj Close' in data.columns:
                    price_data = data['Adj Close']
                elif 'Close' in data.columns:
                    price_data = data['Close']
                
                # 單隻股票時數據結構可能不同
                elif len(symbols) == 1:
                    # 通過索引檢查所有可能的列
                    for col in data.columns:
                        if 'Adj Close' in col:
                            price_data = data[col]
                            break
                        elif 'Close' in col:
                            price_data = data[col]
                            break
            
            # 如果仍然無法獲取價格數據，則嘗試其他方法
            if price_data is None:
                # 手動為每個股票構建收盤價數據
                price_data = pd.DataFrame()
                for symbol in symbols:
                    try:
                        stock_data = yf.download(symbol, period=period)
                        if not stock_data.empty:
                            if 'Adj Close' in stock_data.columns:
                                price_data[symbol] = stock_data['Adj Close']
                            else:
                                price_data[symbol] = stock_data['Close']
                    except Exception:
                        # 如果無法獲取數據，填充零值
                        price_data[symbol] = 0
            
            # 如果仍然沒有數據，返回零
            if price_data is None or price_data.empty:
                return 0.0
                
            # 計算收益率
            returns = price_data.pct_change().dropna()
            
            # 如果收益率數據不足，返回零
            if returns.empty or len(returns) < 2:
                return 0.0
            
            # 確保權重和收益率尺寸匹配
            if len(weights) != returns.shape[1]:
                weights = [1.0/returns.shape[1]] * returns.shape[1]
            
            # 計算投資組合收益率
            portfolio_returns = returns.dot(weights)
            
            # 計算預期收益率（年化）
            expected_return = portfolio_returns.mean() * 252
            
            # 計算投資組合波動率（年化）
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
            
            # 避免除以零
            if portfolio_volatility == 0 or pd.isna(portfolio_volatility):
                return 0.0
            
            # 計算夏普比率
            sharpe_ratio = (expected_return - self.risk_free_rate) / portfolio_volatility
            
            return 0.0 if pd.isna(sharpe_ratio) else sharpe_ratio
        
        except Exception as e:
            # 發生任何錯誤時返回零
            return 0.0
    
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
        try:
            # 確保符號列表非空
            if not symbols or len(symbols) == 0:
                return 0.05  # 返回默認值
                
            # 獲取歷史數據
            data = yf.download(symbols, period=period)
            
            # 確保能夠獲取到數據
            if data.empty:
                return 0.05
                
            # 獲取價格數據
            price_data = None
            
            # 處理數據結構 - 多隻股票情況
            if isinstance(data, pd.DataFrame) and len(data.columns) > 0:
                if 'Adj Close' in data.columns:
                    price_data = data['Adj Close']
                elif 'Close' in data.columns:
                    price_data = data['Close']
                
                # 單隻股票時數據結構可能不同
                elif len(symbols) == 1:
                    # 通過索引檢查所有可能的列
                    for col in data.columns:
                        if 'Adj Close' in col:
                            price_data = data[col]
                            break
                        elif 'Close' in col:
                            price_data = data[col]
                            break
            
            # 如果仍然無法獲取價格數據，則嘗試其他方法
            if price_data is None:
                # 手動為每個股票構建收盤價數據
                price_data = pd.DataFrame()
                for symbol in symbols:
                    try:
                        stock_data = yf.download(symbol, period=period)
                        if not stock_data.empty:
                            if 'Adj Close' in stock_data.columns:
                                price_data[symbol] = stock_data['Adj Close']
                            else:
                                price_data[symbol] = stock_data['Close']
                    except Exception:
                        # 如果無法獲取數據，填充零值
                        price_data[symbol] = 0
            
            # 如果仍然沒有數據，返回默認值
            if price_data is None or price_data.empty:
                return 0.05
                
            # 計算收益率
            returns = price_data.pct_change().dropna()
            
            # 如果收益率數據不足，返回默認值
            if returns.empty or len(returns) < 2:
                return 0.05
            
            # 確保權重和收益率尺寸匹配
            if len(weights) != returns.shape[1]:
                weights = [1.0/returns.shape[1]] * returns.shape[1]
            
            # 計算投資組合收益率
            portfolio_returns = returns.dot(weights)
            
            # 計算VaR
            var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            
            # 確保結果有效
            if pd.isna(var):
                return 0.05
                
            return abs(var)
        
        except Exception as e:
            # 發生任何錯誤時返回默認值
            return 0.05 if confidence_level == 0.95 else 0.08
    
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
        try:
            # 確保符號列表非空
            if not symbols or len(symbols) == 0:
                # 創建空的圖表
                fig = go.Figure()
                fig.update_layout(
                    title="無法創建風險回報圖：沒有股票數據",
                    height=500,
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                return fig
                
            # 獲取歷史數據
            data = yf.download(symbols, period=period)
            
            # 確保能夠獲取到數據
            if data.empty:
                fig = go.Figure()
                fig.update_layout(
                    title="無法創建風險回報圖：無法獲取股票數據",
                    height=500,
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                return fig
                
            # 獲取價格數據
            price_data = None
            
            # 處理數據結構 - 多隻股票情況
            if isinstance(data, pd.DataFrame) and len(data.columns) > 0:
                if 'Adj Close' in data.columns:
                    price_data = data['Adj Close']
                elif 'Close' in data.columns:
                    price_data = data['Close']
                
                # 如果單隻股票數據結構不同
                elif len(symbols) == 1:
                    # 通過索引檢查所有可能的列
                    for col in data.columns:
                        if 'Adj Close' in col:
                            price_data = data[col]
                            break
                        elif 'Close' in col:
                            price_data = data[col]
                            break
            
            # 如果仍然無法獲取價格數據
            if price_data is None or price_data.empty:
                fig = go.Figure()
                fig.update_layout(
                    title="無法創建風險回報圖：價格數據無效",
                    height=500,
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                return fig
                
            # 計算收益率
            returns = price_data.pct_change().dropna()
            
            # 如果收益率數據不足
            if returns.empty or len(returns) < 2:
                fig = go.Figure()
                fig.update_layout(
                    title="無法創建風險回報圖：收益率數據不足",
                    height=500,
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                return fig
            
            # 計算預期收益和波動率
            expected_returns = []
            volatilities = []
            valid_symbols = []
            
            # 如果是多隻股票，處理每隻股票
            if len(symbols) > 1:
                for symbol in symbols:
                    if symbol in returns.columns:
                        symbol_returns = returns[symbol].dropna()
                        if len(symbol_returns) > 1:
                            er = symbol_returns.mean() * 252  # 年化
                            vol = symbol_returns.std() * np.sqrt(252)  # 年化
                            
                            if not pd.isna(er) and not pd.isna(vol):
                                expected_returns.append(er)
                                volatilities.append(vol)
                                valid_symbols.append(symbol)
            else:
                # 單隻股票的情況
                er = returns.mean() * 252
                vol = returns.std() * np.sqrt(252)
                
                if not pd.isna(er) and not pd.isna(vol) and isinstance(er, (int, float)) and isinstance(vol, (int, float)):
                    expected_returns.append(er)
                    volatilities.append(vol)
                    valid_symbols.append(symbols[0])
            
            # 如果無有效數據
            if not valid_symbols:
                fig = go.Figure()
                fig.update_layout(
                    title="無法創建風險回報圖：無有效數據",
                    height=500,
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                return fig
            
            # 創建風險回報數據框
            risk_return_df = pd.DataFrame({
                'Symbol': valid_symbols,
                'Expected Return': expected_returns,
                'Volatility': volatilities
            })
            
            # 創建散點圖
            fig = px.scatter(
                risk_return_df,
                x='Volatility',
                y='Expected Return',
                text='Symbol',
                color='Volatility',
                color_continuous_scale='Viridis',
                title="風險回報分析"
            )
            
            fig.update_traces(textposition='top center')
            
            fig.update_layout(
                height=500,
                margin=dict(l=10, r=10, t=50, b=10),
                xaxis_title="風險（波動率）",
                yaxis_title="預期收益",
                coloraxis_showscale=False
            )
            
            return fig
            
        except Exception as e:
            # 發生錯誤時創建空圖表
            fig = go.Figure()
            fig.update_layout(
                title=f"創建風險回報圖時出錯",
                height=500,
                margin=dict(l=10, r=10, t=50, b=10)
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
        try:
            # 確保符號列表非空
            if not symbols or len(symbols) == 0:
                return {"error": "沒有投資組合股票數據"}
                
            # 確保權重有效
            if not weights or len(weights) == 0:
                weights = [1.0/len(symbols)] * len(symbols)
                
            # 計算風險指標
            volatility = self.calculate_portfolio_volatility(symbols, weights, period)
            sharpe_ratio = self.calculate_sharpe_ratio(symbols, weights, period)
            var_95 = self.calculate_value_at_risk(symbols, weights, 0.95, period)
            var_99 = self.calculate_value_at_risk(symbols, weights, 0.99, period)
            
            # 嘗試計算相關性矩陣和分散化分數
            try:
                correlation_matrix = self.calculate_correlation_matrix(symbols, period)
                avg_correlation = correlation_matrix.values.sum() / (correlation_matrix.shape[0] ** 2)
                diversification_score = 1 - abs(avg_correlation)
                correlation_heatmap = self.create_correlation_heatmap(correlation_matrix)
            except Exception:
                correlation_matrix = pd.DataFrame()
                diversification_score = 0.5
                correlation_heatmap = None
            
            # 創建風險回報散點圖
            risk_return_scatter = self.create_risk_return_scatter(symbols, period)
            
            # 確定風險水平
            if volatility < 0.15:
                risk_level = "低"
                risk_description = "您的投資組合波動率較低，表明保守的投資選擇。"
            elif volatility < 0.25:
                risk_level = "中等"
                risk_description = "您的投資組合具有平衡的風險狀況，波動率適中。"
            else:
                risk_level = "高"
                risk_description = "您的投資組合波動率高，表明激進的投資選擇。"
            
            # 確定夏普比率解釋
            if sharpe_ratio < 0.5:
                sharpe_description = "風險調整後回報不佳。考慮重新平衡您的投資組合。"
            elif sharpe_ratio < 1.0:
                sharpe_description = "風險調整後回報低於平均水平。可能有改進空間。"
            elif sharpe_ratio < 1.5:
                sharpe_description = "風險調整後回報良好。您的投資組合表現還算不錯。"
            else:
                sharpe_description = "風險調整後回報優秀。您的投資組合有效平衡了風險和回報。"
            
            # 確定多元化建議
            if diversification_score < 0.3:
                diversification_recommendation = "您的投資組合缺乏多元化。考慮添加相關性較低的資產。"
            elif diversification_score < 0.6:
                diversification_recommendation = "您的投資組合具有中等多元化。可以通過添加更多樣化的投資來改進。"
            else:
                diversification_recommendation = "您的投資組合已經很好地多元化。繼續監控資產間的相關性。"
            
            # 計算潛在損失金額
            potential_loss_95 = var_95 * portfolio_value
            potential_loss_99 = var_99 * portfolio_value
            
            # 打包結果
            result = {
                "volatility": float(volatility) if not pd.isna(volatility) else 0.0,
                "sharpe_ratio": float(sharpe_ratio) if not pd.isna(sharpe_ratio) else 0.0,
                "correlation_matrix": correlation_matrix,
                "diversification_score": float(diversification_score) if not pd.isna(diversification_score) else 0.5,
                "var_95": float(var_95) if not pd.isna(var_95) else 0.05,
                "var_99": float(var_99) if not pd.isna(var_99) else 0.08,
                "potential_loss_95": float(potential_loss_95) if not pd.isna(potential_loss_95) else portfolio_value * 0.05,
                "potential_loss_99": float(potential_loss_99) if not pd.isna(potential_loss_99) else portfolio_value * 0.08,
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