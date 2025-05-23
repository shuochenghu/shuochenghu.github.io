import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

def calculate_technical_indicators(df, sma_period=None, ema_period=None, rsi_period=None, 
                                   macd_params=None, bollinger_params=None):
    """
    Calculate various technical indicators for the given DataFrame
    
    Parameters:
        df (pd.DataFrame): DataFrame with OHLC price data
        sma_period (int): Period for Simple Moving Average
        ema_period (int): Period for Exponential Moving Average
        rsi_period (int): Period for Relative Strength Index
        macd_params (tuple): Fast, Slow, and Signal periods for MACD
        bollinger_params (tuple): Period and standard deviation for Bollinger Bands
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Simple Moving Average (SMA)
    if sma_period:
        df_copy['SMA'] = df_copy['Close'].rolling(window=sma_period).mean()
    
    # Exponential Moving Average (EMA)
    if ema_period:
        df_copy['EMA'] = df_copy['Close'].ewm(span=ema_period, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    if rsi_period:
        # Calculate price changes
        delta = df_copy['Close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        # Calculate relative strength
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        df_copy['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    if macd_params:
        fast_period, slow_period, signal_period = macd_params
        
        # Calculate fast and slow EMAs
        fast_ema = df_copy['Close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df_copy['Close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        df_copy['MACD'] = fast_ema - slow_ema
        
        # Calculate signal line
        df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        df_copy['MACD_Hist'] = df_copy['MACD'] - df_copy['MACD_Signal']
    
    # Bollinger Bands
    if bollinger_params:
        period, std_dev = bollinger_params
        
        # Calculate middle band (SMA)
        df_copy['BB_Middle'] = df_copy['Close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        rolling_std = df_copy['Close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        df_copy['BB_Upper'] = df_copy['BB_Middle'] + (rolling_std * std_dev)
        df_copy['BB_Lower'] = df_copy['BB_Middle'] - (rolling_std * std_dev)
    
    return df_copy

def format_large_number(num):
    """
    Format large numbers to K, M, B, T
    """
    if num is None:
        return "N/A"
    
    if isinstance(num, str):
        return num
        
    if num >= 1_000_000_000_000:
        return f"${num / 1_000_000_000_000:.2f}T"
    elif num >= 1_000_000_000:
        return f"${num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"${num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"${num / 1_000:.2f}K"
    else:
        return f"${num:.2f}"

def get_company_info(ticker):
    """
    Extract key company information from a ticker object
    
    Parameters:
        ticker (yfinance.Ticker): Ticker object
        
    Returns:
        dict: Dictionary with key financial metrics
    """
    try:
        # Get company information
        info = ticker.info
        
        # Create a dictionary with key financial metrics
        company_info = {
            "Company Name": info.get("longName", info.get("shortName", "N/A")),
            "Market Cap": format_large_number(info.get("marketCap")),
            "P/E Ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else "N/A",
            "Forward P/E": round(info.get("forwardPE", 0), 2) if info.get("forwardPE") else "N/A",
            "PEG Ratio": round(info.get("pegRatio", 0), 2) if info.get("pegRatio") else "N/A",
            "Dividend Yield (%)": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else "N/A",
            "52 Week High": f"${info.get('fiftyTwoWeekHigh', 0):.2f}" if info.get("fiftyTwoWeekHigh") else "N/A",
            "52 Week Low": f"${info.get('fiftyTwoWeekLow', 0):.2f}" if info.get("fiftyTwoWeekLow") else "N/A",
            "50-Day MA": f"${info.get('fiftyDayAverage', 0):.2f}" if info.get("fiftyDayAverage") else "N/A",
            "200-Day MA": f"${info.get('twoHundredDayAverage', 0):.2f}" if info.get("twoHundredDayAverage") else "N/A",
            "Revenue": format_large_number(info.get("totalRevenue")),
            "Profit Margin (%)": round(info.get("profitMargins", 0) * 100, 2) if info.get("profitMargins") else "N/A",
            "Return on Equity (%)": round(info.get("returnOnEquity", 0) * 100, 2) if info.get("returnOnEquity") else "N/A",
            "Return on Assets (%)": round(info.get("returnOnAssets", 0) * 100, 2) if info.get("returnOnAssets") else "N/A",
            "Total Debt": format_large_number(info.get("totalDebt")),
            "Total Cash": format_large_number(info.get("totalCash")),
            "Free Cash Flow": format_large_number(info.get("freeCashflow")),
        }
        
        return company_info
    
    except Exception as e:
        # Return a minimal set of information if there's an error
        return {
            "Company Name": ticker.ticker,
            "Error": f"Could not retrieve complete information: {str(e)}"
        }
