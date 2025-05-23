import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from utils import calculate_technical_indicators, get_company_info, format_large_number

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Platform",
    page_icon="📈",
    layout="wide"
)

# Application title
st.title("📈 Stock Analysis Platform")
st.markdown("Compare multiple stocks with charts, financial data, and technical indicators")

# Sidebar for inputs
with st.sidebar:
    st.header("Settings")
    
    # Input for stock symbols
    stock_input = st.text_input(
        "Enter stock symbols (comma-separated)",
        value="AAPL,MSFT,GOOGL",
        help="Example: AAPL,MSFT,GOOGL or 2330.TW,3008.TW for Taiwan stocks"
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
        "Select time period", 
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
        "Select interval", 
        list(interval_options.keys()),
        index=0  # Default to 1 Day
    )
    
    interval = interval_options[selected_interval]
    
    # Technical indicator selection
    st.subheader("Technical Indicators")
    show_sma = st.checkbox("Simple Moving Average (SMA)", value=True)
    show_ema = st.checkbox("Exponential Moving Average (EMA)", value=False)
    show_rsi = st.checkbox("Relative Strength Index (RSI)", value=True)
    show_macd = st.checkbox("MACD", value=False)
    show_bollinger = st.checkbox("Bollinger Bands", value=False)
    
    # Parameters for indicators
    if show_sma:
        sma_period = st.slider("SMA Period", 5, 200, 50)
    
    if show_ema:
        ema_period = st.slider("EMA Period", 5, 200, 20)
        
    if show_rsi:
        rsi_period = st.slider("RSI Period", 5, 30, 14)
        
    if show_macd:
        macd_fast = st.slider("MACD Fast Period", 5, 20, 12)
        macd_slow = st.slider("MACD Slow Period", 10, 40, 26)
        macd_signal = st.slider("MACD Signal Period", 5, 15, 9)
        
    if show_bollinger:
        bollinger_period = st.slider("Bollinger Period", 5, 50, 20)
        bollinger_std = st.slider("Bollinger Standard Deviation", 1, 4, 2)

# Process stock symbols
if stock_input:
    # Clean up input
    stocks = [symbol.strip().upper() for symbol in stock_input.split(",")]
    
    # Check for valid input
    if len(stocks) > 0:
        # Create tabs for different analysis views
        tabs = st.tabs(["Price Comparison", "Financial Data", "Technical Analysis", "Raw Data"])
        
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
                
                # Update layout
                fig.update_layout(
                    title="Stock Price Comparison",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    legend_title="Stocks",
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
                    title="Percentage Change Since Start of Period",
                    xaxis_title="Date",
                    yaxis_title="Change (%)",
                    legend_title="Stocks",
                    height=400,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_pct, use_container_width=True)
            
            # TAB 2: FINANCIAL DATA
            with tabs[1]:
                st.header("Financial Data Comparison")
                
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
                            
                            price_color = "green" if change >= 0 else "red"
                            st.markdown(f"**Current Price:** ${current_price:.2f}")
                            st.markdown(f"**Change:** <span style='color:{price_color}'>{change:.2f} ({pct_change:.2f}%)</span>", unsafe_allow_html=True)
                            
                        st.markdown("---")
                        
                        # Display other metrics
                        for metric in metrics:
                            if metric in data and data[metric] is not None:
                                st.markdown(f"**{metric}:** {data[metric]}")
            
            # TAB 3: TECHNICAL ANALYSIS
            with tabs[2]:
                st.header("Technical Analysis")
                
                # Stock selector for technical analysis
                tech_stock = st.selectbox(
                    "Select stock for technical analysis",
                    list(stock_data.keys()),
                    key="tech_stock_select"
                )
                
                if tech_stock in stock_data:
                    df = stock_data[tech_stock]["price_data"]
                    
                    # Create figure for price and indicators
                    fig = make_subplots(
                        rows=2, 
                        cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f"{tech_stock} Price with Indicators", "Volume")
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
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{tech_stock} Technical Analysis",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
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
                st.header("Raw Data")
                
                # Stock selector for raw data
                raw_stock = st.selectbox(
                    "Select stock to view raw data",
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
                        label=f"Download {raw_stock} data as CSV",
                        data=csv,
                        file_name=f"{raw_stock}_stock_data.csv",
                        mime='text/csv',
                    )
                
                # Option to download all data
                st.subheader("Download All Data")
                
                all_csv = all_data_df.to_csv().encode('utf-8')
                st.download_button(
                    label="Download All Stock Data as CSV",
                    data=all_csv,
                    file_name="all_stock_data.csv",
                    mime='text/csv',
                )
    else:
        st.warning("Please enter at least one valid stock symbol.")
else:
    st.info("Enter stock symbols in the sidebar to get started.")
