import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import uuid
from utils import calculate_technical_indicators, get_company_info, format_large_number
from simplified_database import (
    save_stock_data, get_cached_stock_data, 
    save_user_preference, get_user_preference,
    save_favorite_stock, remove_favorite_stock, get_favorite_stocks,
    create_portfolio, get_portfolios, delete_portfolio,
    add_portfolio_item, remove_portfolio_item, get_portfolio_items, get_portfolio_by_id
)
from session_state import get_user_id
from translations import get_translation
from simplified_prediction import SimpleStockPredictor
from risk_assessment import PortfolioRiskAnalyzer

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Platform",
    page_icon="📈",
    layout="wide"
)

# Get user ID for database operations
user_id = get_user_id()

# Language options
language_options = {
    'English': 'en',
    '繁體中文': 'zh-tw'
}

# Get saved language preference or default to English
if 'language' not in st.session_state:
    default_language = get_user_preference(user_id, 'language', 'en')
    st.session_state['language'] = default_language

# Function to translate text based on current language
def t(key):
    return get_translation(key, language=st.session_state['language'])

# Application title
st.title(t('app_title'))
st.markdown(t('app_subtitle'))

# Sidebar for inputs
with st.sidebar:
    st.header(t('settings'))
    
    # Language selector
    selected_language_name = st.selectbox(
        t('language'),
        options=list(language_options.keys()),
        index=list(language_options.values()).index(st.session_state['language']) if st.session_state['language'] in language_options.values() else 0
    )
    
    # Navigation
    st.markdown("---")
    if 'page' not in st.session_state:
        st.session_state['page'] = 'stock_analysis'

    pages = {
        'stock_analysis': t('price_comparison'),
        'predictions': t('predictions'), 
        'portfolio': t('portfolio')
    }
    
    selected_page = st.radio(
        "Navigation",
        options=list(pages.keys()),
        format_func=lambda x: pages[x],
        index=list(pages.keys()).index(st.session_state['page']) if st.session_state['page'] in pages.keys() else 0
    )
    
    st.session_state['page'] = selected_page
    
    # Update language if changed
    selected_language = language_options[selected_language_name]
    if selected_language != st.session_state['language']:
        st.session_state['language'] = selected_language
        save_user_preference(user_id, 'language', selected_language)
        st.rerun()
        
    # Stock market selector
    st.subheader(t('stock_market'))
    if 'market' not in st.session_state:
        st.session_state['market'] = 'us'  # Default to US market
        
    market_options = {
        t('us_market'): 'us',
        t('tw_market'): 'tw'
    }
    
    selected_market_name = st.radio(
        t('market_selection'),
        options=list(market_options.keys())
    )
    
    # Update market if changed
    selected_market = market_options[selected_market_name]
    if selected_market != st.session_state['market']:
        st.session_state['market'] = selected_market
        save_user_preference(user_id, 'market', selected_market)
    
    # Input for stock symbols with market-specific defaults
    default_symbols = "AAPL,MSFT,GOOGL" if st.session_state['market'] == 'us' else "2330.TW,3008.TW"
    stock_input = st.text_input(
        t('enter_symbols'),
        value=default_symbols,
        help=t('symbols_help')
    )
    
    # Time period selection
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "Max": "max"
    }
    
    selected_period = st.selectbox(
        t('time_period'), 
        list(period_options.keys()),
        index=3  # Default to 1 Year
    )
    
    period = period_options[selected_period]
    
    # Interval selection
    interval_options = {
        "1 Day": "1d",
        "1 Week": "1wk",
        "1 Month": "1mo"
    }
    
    selected_interval = st.selectbox(
        t('interval'), 
        list(interval_options.keys()),
        index=0  # Default to 1 Day
    )
    
    interval = interval_options[selected_interval]
    
    # Technical indicator selection
    st.subheader(t('tech_indicators'))
    show_sma = st.checkbox(t('sma'), value=True)
    show_ema = st.checkbox(t('ema'), value=False)
    show_rsi = st.checkbox(t('rsi'), value=True)
    show_macd = st.checkbox(t('macd'), value=False)
    show_bollinger = st.checkbox(t('bollinger'), value=False)
    
    # Parameters for indicators
    if show_sma:
        sma_period = st.slider(f"{t('sma')} {t('period')}", 5, 200, 50)
    
    if show_ema:
        ema_period = st.slider(f"{t('ema')} {t('period')}", 5, 200, 20)
        
    if show_rsi:
        rsi_period = st.slider(f"{t('rsi')} {t('period')}", 5, 30, 14)
        
    if show_macd:
        macd_fast = st.slider(f"{t('macd')} Fast {t('period')}", 5, 20, 12)
        macd_slow = st.slider(f"{t('macd')} Slow {t('period')}", 10, 40, 26)
        macd_signal = st.slider(f"{t('macd')} Signal {t('period')}", 5, 15, 9)
        
    if show_bollinger:
        bollinger_period = st.slider(f"{t('bollinger')} {t('period')}", 5, 50, 20)
        bollinger_std = st.slider(f"{t('bollinger')} {t('std_dev')}", 1, 4, 2)

# Main content based on selected page
if st.session_state['page'] == 'stock_analysis':
    # Stock Analysis Page
    # Process stock symbols
    if stock_input:
        # Clean up input
        stocks = [symbol.strip().upper() for symbol in stock_input.split(",")]
        
        # Check for valid input
        if len(stocks) > 0:
            # Create tabs for different analysis views
            tabs = st.tabs([t('price_comparison'), t('financial_data'), t('tech_analysis'), t('raw_data')])
            
            # Dictionary to store data for each stock
            stock_data = {}
            all_data_df = pd.DataFrame()
            
            # Flag to track if any data was retrieved successfully
            valid_data = False
        
        # Loop through each stock and fetch data
        for stock in stocks:
            with st.spinner(f"Loading data for {stock}..."):
                try:
                    # Get stock data
                    ticker = yf.Ticker(stock)
                    hist = ticker.history(period=period, interval=interval)
                    
                    if hist.empty:
                        st.warning(f"No data available for {stock}. Please check the symbol.")
                        continue
                    
                    # Calculate technical indicators
                    hist = calculate_technical_indicators(
                        hist, 
                        sma_period=sma_period if show_sma else None,
                        ema_period=ema_period if show_ema else None,
                        rsi_period=rsi_period if show_rsi else None,
                        macd_params=(macd_fast, macd_slow, macd_signal) if show_macd else None,
                        bollinger_params=(bollinger_period, bollinger_std) if show_bollinger else None
                    )
                    
                    # Get company info
                    info = get_company_info(ticker)
                    
                    # Store data
                    stock_data[stock] = {
                        "price_data": hist,
                        "info": info
                    }
                    
                    # Prepare data for download
                    temp_df = hist.copy()
                    temp_df['Symbol'] = stock
                    all_data_df = pd.concat([all_data_df, temp_df])
                    
                    valid_data = True
                    
                except Exception as e:
                    st.error(f"Error fetching data for {stock}: {str(e)}")
        
        # Only proceed if we have valid data
        if valid_data:
            # Create color map for consistent colors across charts
            colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]
            color_map = {stock: colors[i % len(colors)] for i, stock in enumerate(stock_data.keys())}
            
            # TAB 1: PRICE COMPARISON
            with tabs[0]:
                st.header("Price Comparison")
                
                # Create figure for price comparison
                fig = make_subplots(rows=1, cols=1)
                
                for i, (stock, data) in enumerate(stock_data.items()):
                    df = data["price_data"]
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['Close'],
                            name=f"{stock} Close",
                            line=dict(color=color_map[stock]),
                        )
                    )
                
                # Get correct currency based on selected market
                currency = t('currency_usd') if st.session_state['market'] == 'us' else t('currency_twd')
                
                # Update layout
                fig.update_layout(
                    title=t('compare_stocks'),
                    xaxis_title=t('date'),
                    yaxis_title=t('price_currency').format(currency),
                    legend_title=t('stocks'),
                    height=600,
                    hovermode="x unified"
                )
                
                # Add range slider
                fig.update_layout(
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=[
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ]
                        ),
                        rangeslider=dict(visible=True),
                        type="date"
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate percentage change
                st.subheader("Percentage Change")
                
                # Create figure for percentage change
                fig_pct = make_subplots(rows=1, cols=1)
                
                for i, (stock, data) in enumerate(stock_data.items()):
                    df = data["price_data"].copy()
                    df['Pct_Change'] = (df['Close'] / df['Close'].iloc[0] - 1) * 100
                    
                    fig_pct.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['Pct_Change'],
                            name=f"{stock}",
                            line=dict(color=color_map[stock]),
                        )
                    )
                
                # Update layout
                fig_pct.update_layout(
                    title=t('pct_change_period'),
                    xaxis_title=t('date'),
                    yaxis_title=t('pct_change'),
                    legend_title=t('stocks'),
                    height=400,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_pct, use_container_width=True)
            
            # TAB 2: FINANCIAL DATA
            with tabs[1]:
                st.header(t('financial_comparison'))
                
                # Create a comparison table
                comparison_data = {}
                for stock, data in stock_data.items():
                    info = data["info"]
                    comparison_data[stock] = info
                
                # Set up the metrics to display
                metrics = [
                    "Market Cap", "P/E Ratio", "Forward P/E", "PEG Ratio", 
                    "Dividend Yield (%)", "52 Week High", "52 Week Low", 
                    "50-Day MA", "200-Day MA", "Revenue", "Profit Margin (%)",
                    "Return on Equity (%)", "Return on Assets (%)", "Total Debt", 
                    "Total Cash", "Free Cash Flow"
                ]
                
                # Display key metrics
                cols = st.columns(len(stock_data))
                
                for i, (stock, data) in enumerate(comparison_data.items()):
                    with cols[i]:
                        st.subheader(stock)
                        if "Company Name" in data:
                            st.caption(data["Company Name"])
                            
                        # Display current price
                        if stock in stock_data and not stock_data[stock]["price_data"].empty:
                            current_price = stock_data[stock]["price_data"]["Close"].iloc[-1]
                            change = stock_data[stock]["price_data"]["Close"].iloc[-1] - stock_data[stock]["price_data"]["Close"].iloc[-2]
                            pct_change = (change / stock_data[stock]["price_data"]["Close"].iloc[-2]) * 100
                            
                            # Display with correct currency symbol based on market
                            currency_symbol = "$" if st.session_state['market'] == 'us' else "NT$"
                            
                            price_color = "green" if change >= 0 else "red"
                            st.markdown(f"**{t('current_price')}:** {currency_symbol}{current_price:.2f}")
                            st.markdown(f"**{t('change')}:** <span style='color:{price_color}'>{change:.2f} ({pct_change:.2f}%)</span>", unsafe_allow_html=True)
                            
                        st.markdown("---")
                        
                        # Display other metrics
                        for metric in metrics:
                            if metric in data and data[metric] is not None:
                                st.markdown(f"**{metric}:** {data[metric]}")
            
            # TAB 3: TECHNICAL ANALYSIS
            with tabs[2]:
                st.header(t('tech_analysis'))
                
                # Stock selector for technical analysis
                tech_stock = st.selectbox(
                    t('select_stock'),
                    list(stock_data.keys()),
                    key="tech_stock_select"
                )
                
                if tech_stock in stock_data:
                    df = stock_data[tech_stock]["price_data"]
                    
                    # Get correct currency based on selected market
                    currency = t('currency_usd') if st.session_state['market'] == 'us' else t('currency_twd')
                    
                    # Create figure for price and indicators
                    fig = make_subplots(
                        rows=2, 
                        cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        row_heights=[0.7, 0.3],
                        subplot_titles=(t('price_with_indicators').format(tech_stock), t('volume'))
                    )
                    
                    # Add price candlestick
                    fig.add_trace(
                        go.Candlestick(
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name=tech_stock
                        ),
                        row=1, col=1
                    )
                    
                    # Add volume
                    fig.add_trace(
                        go.Bar(
                            x=df.index,
                            y=df['Volume'],
                            name="Volume",
                            marker=dict(color='rgba(0, 0, 255, 0.5)')
                        ),
                        row=2, col=1
                    )
                    
                    # Add selected technical indicators
                    if show_sma and 'SMA' in df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['SMA'],
                                name=f"SMA ({sma_period})",
                                line=dict(color='rgba(255, 165, 0, 0.7)')
                            ),
                            row=1, col=1
                        )
                    
                    if show_ema and 'EMA' in df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['EMA'],
                                name=f"EMA ({ema_period})",
                                line=dict(color='rgba(128, 0, 128, 0.7)')
                            ),
                            row=1, col=1
                        )
                    
                    if show_bollinger:
                        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=df.index,
                                    y=df['BB_Upper'],
                                    name="Bollinger Upper",
                                    line=dict(color='rgba(0, 128, 0, 0.3)')
                                ),
                                row=1, col=1
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=df.index,
                                    y=df['BB_Lower'],
                                    name="Bollinger Lower",
                                    line=dict(color='rgba(0, 128, 0, 0.3)'),
                                    fill='tonexty',
                                    fillcolor='rgba(0, 128, 0, 0.1)'
                                ),
                                row=1, col=1
                            )
                    
                    # Update layout with correct currency
                    currency = t('currency_usd') if st.session_state['market'] == 'us' else t('currency_twd')
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{tech_stock} {t('tech_analysis')}",
                        xaxis_title=t('date'),
                        yaxis_title=t('price_currency').format(currency),
                        xaxis_rangeslider_visible=False,
                        height=700,
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display RSI and MACD in separate charts if enabled
                    if show_rsi or show_macd:
                        indicator_cols = st.columns(2)
                        
                        # RSI Chart
                        if show_rsi and 'RSI' in df.columns:
                            with indicator_cols[0]:
                                fig_rsi = go.Figure()
                                fig_rsi.add_trace(
                                    go.Scatter(
                                        x=df.index,
                                        y=df['RSI'],
                                        name="RSI",
                                        line=dict(color='blue')
                                    )
                                )
                                
                                # Add overbought/oversold lines
                                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                                
                                fig_rsi.update_layout(
                                    title=f"Relative Strength Index (RSI-{rsi_period})",
                                    xaxis_title="Date",
                                    yaxis_title="RSI",
                                    height=400
                                )
                                
                                st.plotly_chart(fig_rsi, use_container_width=True)
                        
                        # MACD Chart
                        if show_macd and 'MACD' in df.columns:
                            with indicator_cols[1]:
                                fig_macd = make_subplots(rows=1, cols=1)
                                
                                fig_macd.add_trace(
                                    go.Scatter(
                                        x=df.index,
                                        y=df['MACD'],
                                        name="MACD",
                                        line=dict(color='blue')
                                    )
                                )
                                
                                fig_macd.add_trace(
                                    go.Scatter(
                                        x=df.index,
                                        y=df['MACD_Signal'],
                                        name="Signal",
                                        line=dict(color='red')
                                    )
                                )
                                
                                fig_macd.add_trace(
                                    go.Bar(
                                        x=df.index,
                                        y=df['MACD_Hist'],
                                        name="Histogram",
                                        marker=dict(color='rgba(0, 128, 0, 0.5)')
                                    )
                                )
                                
                                fig_macd.update_layout(
                                    title=f"MACD ({macd_fast}, {macd_slow}, {macd_signal})",
                                    xaxis_title="Date",
                                    yaxis_title="MACD",
                                    height=400
                                )
                                
                                st.plotly_chart(fig_macd, use_container_width=True)
            
            # TAB 4: RAW DATA
            with tabs[3]:
                st.header(t('raw_data'))
                
                # Stock selector for raw data
                raw_stock = st.selectbox(
                    t('select_stock'),
                    list(stock_data.keys()),
                    key="raw_stock_select"
                )
                
                if raw_stock in stock_data:
                    df = stock_data[raw_stock]["price_data"]
                    
                    # Display the raw data
                    st.dataframe(df.round(2))
                    
                    # Download button for CSV
                    csv = df.to_csv().encode('utf-8')
                    st.download_button(
                        label=t('download').format(raw_stock),
                        data=csv,
                        file_name=f"{raw_stock}_stock_data.csv",
                        mime='text/csv',
                    )
                
                # Option to download all data
                st.subheader(t('download_data'))
                
                all_csv = all_data_df.to_csv().encode('utf-8')
                st.download_button(
                    label=t('download_all'),
                    data=all_csv,
                    file_name="all_stock_data.csv",
                    mime='text/csv',
                )
    else:
        st.warning(t('valid_symbol'))
elif st.session_state['page'] == 'predictions':
    st.header(t('predictions'))
    st.subheader(t('prediction_title'))
    
    # Stock input for prediction
    prediction_stock_input = st.text_input(
        t('enter_symbols'),
        value=default_symbols.split(',')[0],  # Default to first stock
        help=t('symbols_help')
    )
    
    # Prediction period options
    prediction_period = st.selectbox(
        t('time_period'),
        ["1 Month", "3 Months", "6 Months", "1 Year"],
        index=3  # Default to 1 Year
    )
    
    period_map = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y"
    }
    
    # When user clicks the predict button
    if st.button(t('predict_button')):
        if prediction_stock_input:
            # Clean stock symbol
            stock_symbol = prediction_stock_input.strip().upper()
            
            # Show loading message
            with st.spinner(t('prediction_loading')):
                # Initialize predictor
                predictor = SimpleStockPredictor()
                
                # Get prediction
                result = predictor.predict_stock(
                    stock_symbol, 
                    period=period_map[prediction_period],
                    prophet_days=30,
                    linear_days=30
                )
                
                if result['success']:
                    # Display prediction chart
                    st.plotly_chart(result['prediction_chart'], use_container_width=True)
                    
                    # Display prediction summary
                    summary = result['prediction_summary']
                    
                    # Determine currency symbol
                    currency_symbol = "NT$" if stock_symbol.upper().endswith('.TW') else "$"
                    
                    # Create columns for the summary
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            t('current_price'), 
                            f"{currency_symbol}{summary['current_price']:.2f}",
                            None
                        )
                    
                    # Show Prophet prediction
                    with col2:
                        st.metric(
                            t('prophet_prediction'), 
                            f"{currency_symbol}{summary['prophet_prediction']['price']:.2f}",
                            f"{summary['prophet_prediction']['change_percent']:.2f}%"
                        )
                    
                    # Show Linear Regression prediction
                    with col3:
                        st.metric(
                            t('linear_prediction'), 
                            f"{currency_symbol}{summary['linear_prediction']['price']:.2f}",
                            f"{summary['linear_prediction']['change_percent']:.2f}%"
                        )
                    
                    # Show average prediction and trend
                    st.subheader(t('avg_prediction'))
                    
                    avg_col1, avg_col2, avg_col3 = st.columns(3)
                    
                    with avg_col1:
                        st.metric(
                            t('predicted_price'), 
                            f"{currency_symbol}{summary['average_prediction']['price']:.2f}",
                            f"{summary['average_prediction']['change_percent']:.2f}%"
                        )
                    
                    with avg_col2:
                        # Determine trend text and color
                        if summary['predicted_trend'] == 'up':
                            trend_text = t('trend_up')
                            trend_color = 'green'
                        elif summary['predicted_trend'] == 'down':
                            trend_text = t('trend_down')
                            trend_color = 'red'
                        else:
                            trend_text = t('trend_sideways')
                            trend_color = 'gray'
                            
                        st.markdown(f"**{t('predicted_trend')}**: <span style='color:{trend_color}'>{trend_text}</span>", unsafe_allow_html=True)
                    
                    with avg_col3:
                        st.metric(t('prediction_confidence'), f"{summary['confidence_score']:.0f}%")
                    
                    # Disclaimer
                    st.info("⚠️ This prediction is based on historical data and should not be considered as financial advice. Always consult with a financial advisor before making investment decisions.")
                    
                else:
                    st.error(t('prediction_error').format(result['error']))
        else:
            st.warning(t('valid_symbol'))

# Portfolio Dashboard Page
elif st.session_state['page'] == 'portfolio':
    st.header(t('portfolio_dashboard'))
    
    # 获取用户ID
    user_id = get_user_id()
    
    # 创建标签页：我的投资组合 和 创建投资组合
    portfolio_tabs = st.tabs([t('my_portfolios'), t('create_portfolio')])
    
    # 检查是否在查看某个特定投资组合
    viewing_portfolio = st.session_state.get('current_portfolio_id', None)
    
    # 第一个标签：我的投资组合
    with portfolio_tabs[0]:
        
        # 如果正在查看特定投资组合
        if viewing_portfolio:
            # 获取投资组合信息
            all_portfolios = get_portfolios(user_id)
            current_portfolio = None
            
            for pid, pname, pdesc in all_portfolios:
                if pid == viewing_portfolio:
                    current_portfolio = (pid, pname, pdesc)
                    break
            
            if current_portfolio:
                # 显示投资组合名称和描述
                st.subheader(f"{current_portfolio[1]}")
                if current_portfolio[2]:
                    st.markdown(f"*{current_portfolio[2]}*")
                
                # 返回按钮
                if st.button(t('back_to_portfolios')):
                    st.session_state.pop('current_portfolio_id', None)
                    st.rerun()
                
                # 获取投资组合项目
                items = get_portfolio_items(viewing_portfolio)
                
                # 添加股票按钮
                if st.button(t('add_stock')):
                    st.session_state[f'add_stock_{viewing_portfolio}'] = True
                    st.rerun()
                
                # 处理添加股票表单
                if st.session_state.get(f'add_stock_{viewing_portfolio}', False):
                    with st.form(key=f"add_stock_form"):
                        st.subheader(t('add_stock'))
                        new_symbol = st.text_input(t('symbol')).strip().upper()
                        new_shares = st.number_input(t('shares'), min_value=0.01, step=0.01)
                        new_price = st.number_input(t('purchase_price'), min_value=0.01, step=0.01)
                        new_date = st.date_input(t('purchase_date'), value=datetime.now())
                        new_notes = st.text_area(t('notes'), "")
                        
                        # 提交按钮
                        submitted = st.form_submit_button(t('save'))
                        
                        if submitted:
                            if new_symbol and new_shares > 0:
                                # 添加股票到投资组合
                                add_portfolio_item(
                                    viewing_portfolio, 
                                    new_symbol, 
                                    new_shares, 
                                    new_price, 
                                    new_date,
                                    new_notes
                                )
                                st.success(f"Added {new_symbol} to portfolio.")
                                st.session_state[f'add_stock_{viewing_portfolio}'] = False
                                st.rerun()
                            else:
                                st.error("Symbol and shares are required.")
                    
                    # 取消按钮
                    if st.button(t('cancel')):
                        st.session_state[f'add_stock_{viewing_portfolio}'] = False
                        st.rerun()
                
                # 显示投资组合内容
                if items:
                    # 处理股票数据
                    stock_data = []
                    total_value = 0
                    total_cost = 0
                    
                    for symbol, shares, price, date, notes in items:
                        try:
                            # 获取当前股票数据
                            ticker = yf.Ticker(symbol)
                            history = ticker.history(period="1d")
                            
                            if not history.empty:
                                current_price = history['Close'].iloc[-1]
                                market_value = shares * current_price
                                cost_basis = shares * (price if price else current_price)
                                profit_loss = market_value - cost_basis
                                profit_loss_pct = (profit_loss / cost_basis * 100) if cost_basis > 0 else 0
                                
                                stock_data.append({
                                    'symbol': symbol,
                                    'shares': shares,
                                    'purchase_price': price,
                                    'purchase_date': date,
                                    'current_price': current_price,
                                    'market_value': market_value,
                                    'cost_basis': cost_basis,
                                    'profit_loss': profit_loss,
                                    'profit_loss_pct': profit_loss_pct,
                                    'notes': notes
                                })
                                
                                total_value += market_value
                                total_cost += cost_basis
                        except Exception as e:
                            st.error(f"Error fetching data for {symbol}: {str(e)}")
                    
                    if stock_data:
                        # 当前持股表格
                        st.subheader(t('current_holdings'))
                        
                        # 创建数据框
                        df = pd.DataFrame(stock_data)
                        
                        # 确定货币符号
                        first_symbol = stock_data[0]['symbol'] if stock_data else ""
                        currency_symbol = "NT$" if first_symbol.endswith('.TW') else "$"
                        
                        # 格式化数据框
                        formatted_df = pd.DataFrame({
                            t('symbol'): df['symbol'],
                            t('shares'): df['shares'].map('{:.2f}'.format),
                            t('purchase_price'): df['purchase_price'].map(lambda x: f"{currency_symbol}{x:.2f}" if x else "N/A"),
                            t('current_price'): df['current_price'].map(f"{currency_symbol}{{:.2f}}".format),
                            t('market_value'): df['market_value'].map(f"{currency_symbol}{{:.2f}}".format),
                            t('profit_loss'): df['profit_loss'].map(lambda x: f"{currency_symbol}{x:.2f}"),
                            t('profit_loss_percent'): df['profit_loss_pct'].map(lambda x: f"{x:.2f}%")
                        })
                        
                        st.dataframe(formatted_df)
                        
                        # 显示投资组合总值和盈亏
                        st.subheader(t('total_value'))
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                t('total_value'), 
                                f"{currency_symbol}{total_value:.2f}"
                            )
                        
                        with col2:
                            st.metric(
                                t('total_cost'), 
                                f"{currency_symbol}{total_cost:.2f}"
                            )
                        
                        with col3:
                            total_profit_loss = total_value - total_cost
                            total_profit_loss_pct = (total_profit_loss / total_cost * 100) if total_cost > 0 else 0
                            
                            st.metric(
                                t('total_profit_loss'), 
                                f"{currency_symbol}{total_profit_loss:.2f}",
                                f"{total_profit_loss_pct:.2f}%"
                            )
                        
                        # 投资组合配置图表
                        st.subheader(t('allocation'))
                        
                        # 创建饼图
                        fig = go.Figure(data=[go.Pie(
                            labels=df['symbol'],
                            values=df['market_value'],
                            textinfo='label+percent',
                            insidetextorientation='radial'
                        )])
                        
                        fig.update_layout(
                            height=500,
                            margin=dict(l=20, r=20, t=30, b=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 添加风险评估部分
                        st.subheader(t('risk_assessment'))
                        risk_col1, risk_col2 = st.columns([1, 3])
                        
                        with risk_col1:
                            st.write(t('analyze_portfolio_risk'))
                            analyze_button = st.button(t('analyze_risk'))
                            
                            if analyze_button:
                                st.session_state[f'analyze_risk_{viewing_portfolio}'] = True
                                st.rerun()
                        
                        # 如果用户点击了风险分析按钮
                        if st.session_state.get(f'analyze_risk_{viewing_portfolio}', False):
                            with st.spinner(t('loading')):
                                # 准备风险分析所需的数据
                                symbols = [item['symbol'] for item in stock_data]
                                weights = [item['market_value'] / total_value for item in stock_data]
                                
                                # 创建风险分析器并分析风险
                                risk_analyzer = PortfolioRiskAnalyzer()
                                risk_analysis = risk_analyzer.assess_portfolio_risk(symbols, weights, total_value)
                                
                                if 'error' in risk_analysis:
                                    st.error(f"Error in risk analysis: {risk_analysis['error']}")
                                else:
                                    # 显示风险指标
                                    st.subheader(t('risk_metrics'))
                                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                                    
                                    with metrics_col1:
                                        st.metric(
                                            t('risk_level'), 
                                            risk_analysis['risk_level']
                                        )
                                        st.metric(
                                            t('volatility'), 
                                            f"{risk_analysis['volatility']*100:.2f}%"
                                        )
                                    
                                    with metrics_col2:
                                        st.metric(
                                            t('sharpe_ratio'), 
                                            f"{risk_analysis['sharpe_ratio']:.2f}"
                                        )
                                        st.metric(
                                            t('diversification'), 
                                            f"{risk_analysis['diversification_score']*100:.2f}%"
                                        )
                                    
                                    with metrics_col3:
                                        st.metric(
                                            t('value_at_risk'), 
                                            f"{risk_analysis['var_95']*100:.2f}%"
                                        )
                                        st.metric(
                                            t('potential_loss'), 
                                            f"{currency_symbol}{risk_analysis['potential_loss_95']:.2f}"
                                        )
                                    
                                    # 显示风险描述
                                    st.markdown(f"**{t('risk_level')}**: {risk_analysis['risk_description']}")
                                    st.markdown(f"**{t('risk_adjusted_return')}**: {risk_analysis['sharpe_description']}")
                                    st.markdown(f"**{t('diversification')}**: {risk_analysis['diversification_recommendation']}")
                                    
                                    # 显示相关性热图
                                    st.subheader(t('correlation_heatmap'))
                                    st.plotly_chart(risk_analysis['correlation_heatmap'], use_container_width=True)
                                    
                                    # 显示风险回报散点图
                                    st.subheader(t('risk_return_profile'))
                                    st.plotly_chart(risk_analysis['risk_return_scatter'], use_container_width=True)
                            
                            # 添加关闭按钮
                            if st.button(t('close')):
                                st.session_state.pop(f'analyze_risk_{viewing_portfolio}', None)
                                st.rerun()
                        
                        # 个别股票操作
                        st.subheader(t('stock_actions'))
                        
                        for i, stock in enumerate(stock_data):
                            with st.expander(f"{stock['symbol']} - {stock['shares']} shares"):
                                # 股票详情
                                st.markdown(f"**{t('purchase_price')}:** {currency_symbol}{stock['purchase_price']:.2f}" if stock['purchase_price'] else f"**{t('purchase_price')}:** N/A")
                                st.markdown(f"**{t('purchase_date')}:** {stock['purchase_date'].strftime('%Y-%m-%d') if isinstance(stock['purchase_date'], datetime) else stock['purchase_date']}")
                                st.markdown(f"**{t('notes')}:** {stock['notes']}" if stock['notes'] else f"**{t('notes')}:** None")
                                
                                # 操作按钮
                                action_col1, action_col2 = st.columns([1, 1])
                                
                                # 在操作列中添加股票编辑和删除按钮
                                with action_col1:
                                    if st.button(t('edit_stock'), key=f"edit_{stock['symbol']}"):
                                        st.session_state[f'edit_stock_{viewing_portfolio}_{stock["symbol"]}'] = True
                                        st.rerun()
                                
                                with action_col2:
                                    if st.button(t('remove_stock'), key=f"remove_{stock['symbol']}"):
                                        # 从投资组合中移除股票
                                        remove_portfolio_item(viewing_portfolio, stock['symbol'])
                                        st.success(f"Removed {stock['symbol']} from portfolio.")
                                        st.rerun()
                                
                                # 处理编辑股票表单
                                if st.session_state.get(f'edit_stock_{viewing_portfolio}_{stock["symbol"]}', False):
                                    with st.form(key=f"edit_stock_form_{stock['symbol']}"):
                                        st.subheader(f"{t('edit_stock')} - {stock['symbol']}")
                                        edit_shares = st.number_input(t('shares'), min_value=0.01, value=float(stock['shares']), step=0.01)
                                        edit_price = st.number_input(t('purchase_price'), min_value=0.01, value=float(stock['purchase_price']) if stock['purchase_price'] else 0.01, step=0.01)
                                        edit_date = st.date_input(t('purchase_date'), value=stock['purchase_date'] if isinstance(stock['purchase_date'], datetime) else datetime.now())
                                        edit_notes = st.text_area(t('notes'), stock['notes'] if stock['notes'] else "")
                                        
                                        # 提交按钮
                                        submitted = st.form_submit_button(t('save'))
                                        
                                        if submitted:
                                            if edit_shares > 0:
                                                # 更新投资组合中的股票
                                                add_portfolio_item(
                                                    viewing_portfolio, 
                                                    stock['symbol'], 
                                                    edit_shares, 
                                                    edit_price, 
                                                    edit_date,
                                                    edit_notes
                                                )
                                                st.success(f"Updated {stock['symbol']} in portfolio.")
                                                st.session_state[f'edit_stock_{viewing_portfolio}_{stock["symbol"]}'] = False
                                                st.rerun()
                                            else:
                                                st.error("Shares must be greater than 0.")
                                    
                                    # 取消按钮
                                    if st.button(t('cancel'), key=f"cancel_edit_{stock['symbol']}"):
                                        st.session_state[f'edit_stock_{viewing_portfolio}_{stock["symbol"]}'] = False
                                        st.rerun()
                    else:
                        st.info(t('no_stocks'))
                else:
                    st.info(t('no_stocks'))
            else:
                st.error("Portfolio not found.")
                st.session_state.pop('current_portfolio_id', None)
                st.rerun()
        
        # 如果没有查看特定投资组合，显示投资组合列表
        else:
            # 获取用户投资组合
            portfolios = get_portfolios(user_id)
            
            if portfolios:
                # 显示用户投资组合
                st.subheader(t('my_portfolios'))
                
                # 为每个投资组合创建卡片
                for i, (portfolio_id, name, description) in enumerate(portfolios):
                    with st.expander(name, expanded=True if i == 0 else False):
                        # 如果有描述，显示描述
                        if description:
                            st.markdown(f"*{description}*")
                        
                        # 创建投资组合操作按钮
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            if st.button(t('view'), key=f"view_{portfolio_id}"):
                                # 将投资组合ID存储在会话状态中
                                st.session_state['current_portfolio_id'] = portfolio_id
                                st.rerun()
                        
                        with col2:
                            if st.button(t('delete'), key=f"delete_{portfolio_id}"):
                                # 显示确认对话框
                                st.session_state[f'confirm_delete_{portfolio_id}'] = True
                                st.rerun()
                        
                        # 处理删除确认
                        if st.session_state.get(f'confirm_delete_{portfolio_id}', False):
                            st.warning(t('confirm_delete'))
                            conf_col1, conf_col2 = st.columns([1, 1])
                            
                            with conf_col1:
                                if st.button(t('yes'), key=f"confirm_yes_{portfolio_id}"):
                                    # 删除投资组合
                                    delete_portfolio(portfolio_id)
                                    # 重置确认状态
                                    st.session_state[f'confirm_delete_{portfolio_id}'] = False
                                    st.success(f"Portfolio '{name}' deleted.")
                                    st.rerun()
                            
                            with conf_col2:
                                if st.button(t('no'), key=f"confirm_no_{portfolio_id}"):
                                    # 重置确认状态
                                    st.session_state[f'confirm_delete_{portfolio_id}'] = False
                                    st.rerun()
            else:
                st.info(t('no_portfolios'))
    
    # 第二个标签：创建投资组合
    with portfolio_tabs[1]:
        st.subheader(t('create_portfolio'))
        
        # Form to create a new portfolio
        with st.form(key="create_portfolio_form"):
            portfolio_name = st.text_input(t('portfolio_name'))
            portfolio_description = st.text_area(t('portfolio_description'))
            
            # Submit button
            submitted = st.form_submit_button(t('create'))
            
            if submitted:
                if portfolio_name:
                    # Create portfolio
                    portfolio_id = create_portfolio(user_id, portfolio_name, portfolio_description)
                    
                    if portfolio_id:
                        st.success(f"Portfolio '{portfolio_name}' created.")
                        # Set as current portfolio
                        st.session_state['current_portfolio_id'] = portfolio_id
                        st.rerun()
                    else:
                        st.error("Failed to create portfolio. Please try again.")
                else:
                    st.error("Portfolio name is required.")
