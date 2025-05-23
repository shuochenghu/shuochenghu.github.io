"""
AI-powered analysis module for stock portfolio and market insights
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
import openai
from openai import OpenAI

# 初始化OpenAI客戶端
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def analyze_portfolio_risk(portfolio_data, risk_metrics):
    """
    使用OpenAI分析投資組合風險並提供建議
    
    Parameters:
        portfolio_data (dict): 投資組合股票數據
        risk_metrics (dict): 風險評估指標
        
    Returns:
        dict: AI分析結果和建議
    """
    try:
        # 構建提示詞
        stocks_info = []
        for stock in portfolio_data:
            stock_info = f"股票代碼: {stock['symbol']}, 持股量: {stock['shares']}, 目前價格: ${stock['current_price']:.2f}, 市值: ${stock['market_value']:.2f}"
            stocks_info.append(stock_info)
        
        stock_details = "\n".join(stocks_info)
        
        # 投資組合風險指標
        risk_info = (
            f"波動率(年化): {risk_metrics.get('volatility', 0)*100:.2f}%\n"
            f"夏普比率: {risk_metrics.get('sharpe_ratio', 0):.2f}\n"
            f"多元化分數: {risk_metrics.get('diversification_score', 0):.2f}\n"
            f"風險水平: {risk_metrics.get('risk_level', '中等')}\n"
            f"95%置信區間VaR: {risk_metrics.get('var_95', 0)*100:.2f}%\n"
        )
        
        # 創建提示詞
        prompt = f"""
        你是一位專業的投資風險分析師，需要對以下投資組合進行分析並提供建議。請基於以下數據提供深入的風險評估和投資建議。

        ## 投資組合詳情
        {stock_details}

        ## 風險指標
        {risk_info}

        請提供以下格式的分析:
        1. 風險評估摘要 (3-4句話概述)
        2. 主要風險點 (列出2-3個具體風險)
        3. 優化建議 (提供2-3個具體的優化建議)
        4. 市場洞察 (基於目前市場情況提供見解)

        請使用專業但容易理解的語言，避免過於技術性的術語。分析應該有實用性和可操作性。
        """
        
        # 調用OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o", # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            messages=[
                {"role": "system", "content": "你是一位精通繁體中文的金融風險分析專家，專門為台灣和香港的投資者提供投資組合分析。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        analysis = response.choices[0].message.content
        
        return {
            "status": "success",
            "analysis": analysis
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"AI分析失敗: {str(e)}"
        }

def get_stock_prediction_explanation(ticker_symbol, prediction_data):
    """
    使用OpenAI解釋股票價格預測結果
    
    Parameters:
        ticker_symbol (str): 股票代碼
        prediction_data (dict): 預測數據
        
    Returns:
        str: 預測解釋
    """
    try:
        # 獲取預測數據
        prophet_prediction = prediction_data.get('prophet_prediction', {})
        linear_prediction = prediction_data.get('linear_prediction', {})
        
        # 如果沒有預測數據，返回錯誤
        if not prophet_prediction or not linear_prediction:
            return "無法提供預測分析：預測數據不完整"
        
        # 計算預測變化百分比
        prophet_last = prophet_prediction.get('forecast', [])[-1] if prophet_prediction.get('forecast') else 0
        linear_last = linear_prediction.get('forecast', [])[-1] if linear_prediction.get('forecast') else 0
        
        current_price = prediction_data.get('last_price', 0)
        
        prophet_change = (prophet_last - current_price) / current_price * 100 if current_price else 0
        linear_change = (linear_last - current_price) / current_price * 100 if current_price else 0
        
        # 計算平均預測變化
        avg_change = (prophet_change + linear_change) / 2
        
        # 構建提示詞
        prompt = f"""
        請基於以下股票預測數據，提供專業的股票前景分析和投資建議:

        股票代碼: {ticker_symbol}
        當前價格: ${current_price:.2f}
        
        Prophet模型預測:
        - 預測價格: ${prophet_last:.2f}
        - 預測變化: {prophet_change:.2f}%
        
        線性回歸模型預測:
        - 預測價格: ${linear_last:.2f}
        - 預測變化: {linear_change:.2f}%
        
        平均預測變化: {avg_change:.2f}%

        請包括:
        1. 預測趨勢解讀 (1-2句話)
        2. 模型差異分析 (1-2句話解釋為什麼兩個模型可能有不同結果)
        3. 投資建議 (1-2個簡短建議)
        4. 風險提示 (1個關於預測模型局限性的提示)

        回答應簡潔明了，總長度不超過250字。
        """
        
        # 調用OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o", # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            messages=[
                {"role": "system", "content": "你是一位精通繁體中文的金融分析師，專門為台灣和香港的投資者提供股票預測分析。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=350
        )
        
        explanation = response.choices[0].message.content
        
        return explanation
    
    except Exception as e:
        return f"無法產生預測解釋: {str(e)}"