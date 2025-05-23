import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
    else:
        sma_period = 50
        
    if show_ema:
        ema_period = st.slider(f"{t('ema')} {t('period')}", 5, 200, 20)
    else:
        ema_period = 20
        
    if show_rsi:
        rsi_period = st.slider(f"{t('rsi')} {t('period')}", 5, 30, 14)
    else:
        rsi_period = 14
        
    if show_macd:
        macd_fast = st.slider(f"{t('macd')} Fast {t('period')}", 5, 20, 12)
        macd_slow = st.slider(f"{t('macd')} Slow {t('period')}", 10, 40, 26)
        macd_signal = st.slider(f"{t('macd')} Signal {t('period')}", 5, 15, 9)
    else:
        macd_fast = 12
        macd_slow = 26
        macd_signal = 9
        
    if show_bollinger:
        bollinger_period = st.slider(f"{t('bollinger')} {t('period')}", 5, 50, 20)
        bollinger_std = st.slider(f"{t('bollinger')} {t('std_dev')}", 1, 4, 2)
    else:
        bollinger_period = 20
        bollinger_std = 2

# Main content based on selected page
if st.session_state['page'] == 'stock_analysis':
    # Stock Analysis Page
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
                                name=t('price'),
                                increasing_line_color='green',
                                decreasing_line_color='red'
                            ),
                            row=1, col=1
                        )
                        
                        # Add volume
                        fig.add_trace(
                            go.Bar(
                                x=df.index,
                                y=df['Volume'],
                                name=t('volume'),
                                marker_color='rgba(0, 0, 255, 0.5)'
                            ),
                            row=2, col=1
                        )
                        
                        # Add SMA
                        if show_sma and 'SMA' in df.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=df.index,
                                    y=df['SMA'],
                                    name=f"SMA ({sma_period})",
                                    line=dict(color='blue', width=1)
                                ),
                                row=1, col=1
                            )
                        
                        # Add EMA
                        if show_ema and 'EMA' in df.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=df.index,
                                    y=df['EMA'],
                                    name=f"EMA ({ema_period})",
                                    line=dict(color='orange', width=1)
                                ),
                                row=1, col=1
                            )
                        
                        # Add Bollinger Bands
                        if show_bollinger and 'Bollinger_Upper' in df.columns:
                            # Upper band
                            fig.add_trace(
                                go.Scatter(
                                    x=df.index,
                                    y=df['Bollinger_Upper'],
                                    name=t('bollinger_upper'),
                                    line=dict(color='rgba(250, 0, 0, 0.5)', width=1),
                                    hoverinfo='none'
                                ),
                                row=1, col=1
                            )
                            
                            # Lower band
                            fig.add_trace(
                                go.Scatter(
                                    x=df.index,
                                    y=df['Bollinger_Lower'],
                                    name=t('bollinger_lower'),
                                    line=dict(color='rgba(250, 0, 0, 0.5)', width=1),
                                    fill='tonexty',
                                    fillcolor='rgba(250, 0, 0, 0.05)',
                                    hoverinfo='none'
                                ),
                                row=1, col=1
                            )
                        
                        # Add RSI in a subplot
                        if show_rsi and 'RSI' in df.columns:
                            # Add RSI indicator
                            fig.add_trace(
                                go.Scatter(
                                    x=df.index,
                                    y=df['RSI'],
                                    name=f"RSI ({rsi_period})",
                                    line=dict(color='purple', width=1)
                                ),
                                row=1, col=1
                            )
                            
                            # Create RSI subplot
                            fig_rsi = make_subplots(rows=1, cols=1)
                            
                            # Add RSI line
                            fig_rsi.add_trace(
                                go.Scatter(
                                    x=df.index,
                                    y=df['RSI'],
                                    name=f"RSI ({rsi_period})",
                                    line=dict(color='purple', width=1)
                                )
                            )
                            
                            # Add overbought line (70)
                            fig_rsi.add_trace(
                                go.Scatter(
                                    x=[df.index[0], df.index[-1]],
                                    y=[70, 70],
                                    name=t('overbought'),
                                    line=dict(color='red', width=1, dash='dash'),
                                    hoverinfo='none'
                                )
                            )
                            
                            # Add oversold line (30)
                            fig_rsi.add_trace(
                                go.Scatter(
                                    x=[df.index[0], df.index[-1]],
                                    y=[30, 30],
                                    name=t('oversold'),
                                    line=dict(color='green', width=1, dash='dash'),
                                    hoverinfo='none'
                                )
                            )
                            
                            # Update layout
                            fig_rsi.update_layout(
                                title=f"RSI ({rsi_period})",
                                yaxis_title=t('rsi'),
                                height=300,
                                hovermode="x unified",
                                yaxis=dict(range=[0, 100])
                            )
                            
                            st.plotly_chart(fig_rsi, use_container_width=True)
                        
                        # Add MACD in a subplot
                        if show_macd and 'MACD' in df.columns:
                            # Create MACD subplot
                            fig_macd = make_subplots(rows=1, cols=1)
                            
                            # Add MACD line
                            fig_macd.add_trace(
                                go.Scatter(
                                    x=df.index,
                                    y=df['MACD'],
                                    name=f"MACD ({macd_fast},{macd_slow},{macd_signal})",
                                    line=dict(color='blue', width=1)
                                )
                            )
                            
                            # Add Signal line
                            fig_macd.add_trace(
                                go.Scatter(
                                    x=df.index,
                                    y=df['MACD_Signal'],
                                    name=t('macd_signal'),
                                    line=dict(color='red', width=1)
                                )
                            )
                            
                            # Add Histogram
                            fig_macd.add_trace(
                                go.Bar(
                                    x=df.index,
                                    y=df['MACD_Hist'],
                                    name=t('macd_hist'),
                                    marker_color=np.where(df['MACD_Hist'] >= 0, 'green', 'red')
                                )
                            )
                            
                            # Update layout
                            fig_macd.update_layout(
                                title=f"MACD ({macd_fast}, {macd_slow}, {macd_signal})",
                                yaxis_title=t('macd'),
                                height=300,
                                hovermode="x unified"
                            )
                            
                            st.plotly_chart(fig_macd, use_container_width=True)
                        
                        # Update layout for main chart
                        fig.update_layout(
                            title=t('price_chart').format(tech_stock),
                            yaxis_title=t('price_currency').format(currency),
                            xaxis_title=t('date'),
                            height=600,
                            hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
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
                        
                        # Display raw data table
                        st.dataframe(df, use_container_width=True)
                        
                        # Download button for CSV
                        st.download_button(
                            label=t('download_csv'),
                            data=df.to_csv(),
                            file_name=f"{raw_stock}_data.csv",
                            mime='text/csv',
                        )
                    
                    # Download button for all data
                    st.subheader(t('download_all_data'))
                    all_csv = all_data_df.to_csv()
                    st.download_button(
                        label=t('download_all_csv'),
                        data=all_csv,
                        file_name="all_stock_data.csv",
                        mime='text/csv',
                    )
            else:
                st.warning(t('valid_symbol'))
        else:
            st.warning(t('valid_symbol'))
    else:
        st.info(t('enter_to_start'))

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
                    st.info("⚠️ " + t('prediction_disclaimer'))
                    
                else:
                    st.error(t('prediction_error').format(result['error']))
        else:
            st.warning(t('valid_symbol'))

elif st.session_state['page'] == 'portfolio':
    st.header(t('portfolio_dashboard'))
    
    # Create tabs for portfolio management and viewing
    portfolio_tabs = st.tabs([t('my_portfolios'), t('create_portfolio')])
    
    # Tab 1: My Portfolios
    with portfolio_tabs[0]:
        # Get user's portfolios
        portfolios = get_portfolios(user_id)
        
        if portfolios:
            # Show user's portfolios
            st.subheader(t('my_portfolios'))
            
            # Create a card for each portfolio
            for i, (portfolio_id, name, description) in enumerate(portfolios):
                with st.expander(name, expanded=True if i == 0 else False):
                    # If description exists, show it
                    if description:
                        st.markdown(f"*{description}*")
                    
                    # Create buttons for portfolio actions
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        view_btn = st.button(t('view'), key=f"view_{portfolio_id}")
                        if view_btn:
                            # Store portfolio ID in session state and force rerun
                            st.session_state['current_portfolio_id'] = portfolio_id
                            # Add debug print
                            st.write(f"设置当前投资组合ID: {portfolio_id}")
                            st.rerun()
                    
                    with col2:
                        if st.button(t('delete'), key=f"delete_{portfolio_id}"):
                            # Show confirmation dialog
                            st.session_state[f'confirm_delete_{portfolio_id}'] = True
                            st.rerun()
                    
                    # Handle delete confirmation
                    if st.session_state.get(f'confirm_delete_{portfolio_id}', False):
                        st.warning(t('confirm_delete'))
                        conf_col1, conf_col2 = st.columns([1, 1])
                        
                        with conf_col1:
                            if st.button(t('yes'), key=f"confirm_yes_{portfolio_id}"):
                                # Delete portfolio
                                delete_portfolio(portfolio_id)
                                # Reset confirmation state
                                st.session_state[f'confirm_delete_{portfolio_id}'] = False
                                st.success(f"Portfolio '{name}' deleted.")
                                st.rerun()
                        
                        with conf_col2:
                            if st.button(t('no'), key=f"confirm_no_{portfolio_id}"):
                                # Reset confirmation state
                                st.session_state[f'confirm_delete_{portfolio_id}'] = False
                                st.rerun()
                    
                    # Show portfolio details if selected
                    if st.session_state.get('current_portfolio_id') == portfolio_id:
                        # Show portfolio details
                        st.subheader(t('portfolio_details'))
                        
                        # Get portfolio items
                        items = get_portfolio_items(portfolio_id)
                        
                        if items:
                            # Show add stock button
                            if st.button(t('add_stock'), key=f"add_stock_{portfolio_id}"):
                                st.session_state[f'add_stock_{portfolio_id}'] = True
                                st.rerun()
                            
                            # Handle add stock form
                            if st.session_state.get(f'add_stock_{portfolio_id}', False):
                                with st.form(key=f"add_stock_form_{portfolio_id}"):
                                    st.subheader(t('add_stock'))
                                    new_symbol = st.text_input(t('symbol')).strip().upper()
                                    new_shares = st.number_input(t('shares'), min_value=0.01, step=0.01)
                                    new_price = st.number_input(t('purchase_price'), min_value=0.01, step=0.01)
                                    new_date = st.date_input(t('purchase_date'), value=datetime.now())
                                    new_notes = st.text_area(t('notes'), "")
                                    
                                    # Submit button
                                    submitted = st.form_submit_button(t('save'))
                                    
                                    if submitted:
                                        if new_symbol and new_shares > 0:
                                            # Add stock to portfolio
                                            add_portfolio_item(
                                                portfolio_id, 
                                                new_symbol, 
                                                new_shares, 
                                                new_price, 
                                                new_date,
                                                new_notes
                                            )
                                            st.success(f"Added {new_symbol} to portfolio.")
                                            st.session_state[f'add_stock_{portfolio_id}'] = False
                                            st.rerun()
                                        else:
                                            st.error("Symbol and shares are required.")
                                
                                # Cancel button
                                if st.button(t('cancel'), key=f"cancel_add_{portfolio_id}"):
                                    st.session_state[f'add_stock_{portfolio_id}'] = False
                                    st.rerun()
                            
                            # Create a table of holdings
                            stock_data = []
                            total_value = 0
                            total_cost = 0
                            
                            for symbol, shares, price, date, notes in items:
                                try:
                                    # Get current stock data
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
                                # Current holdings table
                                st.subheader(t('current_holdings'))
                                
                                # Create a dataframe
                                df = pd.DataFrame(stock_data)
                                
                                # Determine currency symbol (assuming all stocks in same market)
                                first_symbol = stock_data[0]['symbol'] if stock_data else ""
                                currency_symbol = "NT$" if first_symbol.endswith('.TW') else "$"
                                
                                # Format dataframe
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
                                
                                # Show total portfolio value and profit/loss
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
                                
                                # Portfolio allocation chart
                                st.subheader(t('allocation'))
                                
                                # Create pie chart of portfolio allocation
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
                                
                                # Individual stock actions
                                st.subheader("Stock Actions")
                                
                                for i, stock in enumerate(stock_data):
                                    with st.expander(f"{stock['symbol']} - {stock['shares']} shares"):
                                        # Stock details
                                        st.markdown(f"**Purchase Price:** {currency_symbol}{stock['purchase_price']:.2f}" if stock['purchase_price'] else "**Purchase Price:** N/A")
                                        st.markdown(f"**Purchase Date:** {stock['purchase_date'].strftime('%Y-%m-%d') if isinstance(stock['purchase_date'], datetime) else stock['purchase_date']}")
                                        st.markdown(f"**Notes:** {stock['notes']}" if stock['notes'] else "**Notes:** None")
                                        
                                        # Action buttons
                                        action_col1, action_col2 = st.columns([1, 1])
                                        
                                        with action_col1:
                                            if st.button(t('edit_stock'), key=f"edit_{portfolio_id}_{stock['symbol']}"):
                                                st.session_state[f'edit_stock_{portfolio_id}_{stock["symbol"]}'] = True
                                                st.rerun()
                                        
                                        with action_col2:
                                            if st.button(t('remove_stock'), key=f"remove_{portfolio_id}_{stock['symbol']}"):
                                                # Remove stock
                                                remove_portfolio_item(portfolio_id, stock['symbol'])
                                                st.success(f"Removed {stock['symbol']} from portfolio.")
                                                st.rerun()
                                        
                                        # Handle edit stock form
                                        if st.session_state.get(f'edit_stock_{portfolio_id}_{stock["symbol"]}', False):
                                            with st.form(key=f"edit_stock_form_{portfolio_id}_{stock['symbol']}"):
                                                st.subheader(f"{t('edit_stock')} - {stock['symbol']}")
                                                edit_shares = st.number_input(t('shares'), min_value=0.01, value=float(stock['shares']), step=0.01)
                                                edit_price = st.number_input(t('purchase_price'), min_value=0.01, value=float(stock['purchase_price']) if stock['purchase_price'] else 0.01, step=0.01)
                                                edit_date = st.date_input(t('purchase_date'), value=stock['purchase_date'] if isinstance(stock['purchase_date'], datetime) else datetime.now())
                                                edit_notes = st.text_area(t('notes'), stock['notes'] if stock['notes'] else "")
                                                
                                                # Submit button
                                                submitted = st.form_submit_button(t('save'))
                                                
                                                if submitted:
                                                    if edit_shares > 0:
                                                        # Update stock in portfolio
                                                        add_portfolio_item(
                                                            portfolio_id, 
                                                            stock['symbol'], 
                                                            edit_shares, 
                                                            edit_price, 
                                                            edit_date,
                                                            edit_notes
                                                        )
                                                        st.success(f"Updated {stock['symbol']} in portfolio.")
                                                        st.session_state[f'edit_stock_{portfolio_id}_{stock["symbol"]}'] = False
                                                        st.rerun()
                                                    else:
                                                        st.error("Shares must be greater than 0.")
                                            
                                            # Cancel button
                                            if st.button(t('cancel'), key=f"cancel_edit_{portfolio_id}_{stock['symbol']}"):
                                                st.session_state[f'edit_stock_{portfolio_id}_{stock["symbol"]}'] = False
                                                st.rerun()
                            else:
                                st.info(t('no_stocks'))
                                
                                # Add stock button
                                if st.button(t('add_stock'), key=f"first_add_{portfolio_id}"):
                                    st.session_state[f'add_stock_{portfolio_id}'] = True
                                    st.rerun()
                        
                        # Back button
                        if st.button(t('back_to_portfolios')):
                            # Clear current portfolio ID
                            st.session_state.pop('current_portfolio_id', None)
                            st.rerun()
        else:
            st.info(t('no_portfolios'))
    
    # Tab 2: Create Portfolio
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